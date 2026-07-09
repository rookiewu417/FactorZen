"""Agent 闭环主循环：只调度，业务逻辑在 nodes。"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from factorzen.agents.nodes import (
    node_critic,
    node_evaluate,
    node_generate,
    node_guardrails,
    node_reflect,
)
from factorzen.agents.state import AgentState
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.client import LLMClientError
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import split_holdout
from factorzen.validation.multiple_testing import TrialLedger

_LOG = logging.getLogger(__name__)


@dataclass
class AgentResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int


def run_llm_agent(daily, llm_fn: LLMFn, *, n_rounds: int, seed: int, top_k: int = 5,
                  holdout_ratio: float = 0.2, human_review: bool = False,
                  patience: int | None = None,
                  heal_rounds: int = 2,
                  on_round_end: Callable[[AgentResult], None] | None = None,
                  llm_failure_patience: int = 3) -> AgentResult:
    """跑 n_rounds 轮 Agent 挖掘闭环。

    ``on_round_end``：每个**成功**轮次结束时以当前累积结果回调，供调用方增量落盘。
    没有它，进程在第 N 轮崩溃会让前 N-1 轮的候选全部丢失（manifest 只在返回后才写）。

    ``llm_failure_patience``：连续多少轮 LLM 不可用即提前终止。单轮的 ``LLMClientError``
    （client 层重试已耗尽，或 422 这类不可重试错误）只跳过该轮，不崩整个 session；
    但 LLM 持续不可用时空转跑满 n_rounds 毫无意义。计数器在成功轮重置，
    否则零散抖动会被累计成「持续不可用」。

    只吞 ``LLMClientError``。其余异常（代码 bug、磁盘满）照常冒泡——静默吞掉它们
    会把真实缺陷伪装成「LLM 抖动」。
    """
    rng = np.random.default_rng(seed)  # noqa: F841 预留给未来随机选择，保证可复现入口
    mining_df, holdout_df, _hstart = split_holdout(daily, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)        # Agent 只见 mining 段
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    feedback = ""
    no_improve = 0
    last_cand_count = 0
    llm_failures = 0
    for round_i in range(n_rounds):
        # 自适应早停：连续 patience 轮无新 passed 候选则停（patience=None → 跑满，零回归）
        if patience is not None and round_i > 0:
            no_improve = 0 if len(state.candidates) > last_cand_count else no_improve + 1
            if no_improve >= patience:
                break
        last_cand_count = len(state.candidates)
        try:
            state = node_generate(state, llm_fn, daily=mining_df, bundle=bundle,
                                  feedback=feedback, heal_rounds=heal_rounds)
            state = node_evaluate(state, daily=mining_df, bundle=bundle)
            state = node_guardrails(state, daily=mining_df, holdout_df=holdout_df,
                                    bundle=bundle, ledger=ledger, top_k=top_k)
            state = node_critic(state, llm_fn)
        except LLMClientError as exc:
            llm_failures += 1
            # 丢弃本轮未评估的暂存表达式；node_reflect 未执行，故此处补推进 iteration
            state._pending = []  # type: ignore[attr-defined]
            state.iteration += 1
            _LOG.warning("第 %d 轮 LLM 不可用（连续第 %d 次），跳过本轮: %s",
                         round_i, llm_failures, exc)
            if llm_failures >= llm_failure_patience:
                _LOG.error("连续 %d 轮 LLM 不可用，提前终止挖掘（已产出 %d 个候选）",
                           llm_failures, len(state.candidates))
                break
            continue
        llm_failures = 0
        if human_review:
            _human_gate(state)  # 打印候选 + 等输入（非交互/CI 跳过）
        state = node_reflect(state)
        feedback = _summarize_feedback(state)
        if on_round_end is not None:
            on_round_end(AgentResult(state=state, candidates=state.candidates,
                                     n_trials=ledger.n_trials))
    return AgentResult(state=state, candidates=state.candidates, n_trials=ledger.n_trials)


def _summarize_feedback(state: AgentState) -> str:
    """把上一轮结果压成喂给下一轮 prompt 的反馈。

    「最佳」= 上一轮 |train_IC| 最大的**可评估** attempt。三个必须守住的点：
    只看上一轮（`state.iteration - 1`，node_reflect 已把 iteration +1）——上一轮颗粒无收时
    不许回退去报更早轮次的战绩；按 |IC| 取最佳——反向因子同样有效；排除 ic_train=None
    的编译失败项——否则「上轮最佳 train_IC=None」会被原样喂给 LLM。
    """
    if not state.attempts:
        return ""
    n_seen = len(state.seen_expressions)
    prev = state.iteration - 1
    scored = [a for a in state.attempts if a.iteration == prev and a.ic_train is not None]
    if not scored:
        return f"上一轮无可评估表达式（编译或求值全部失败）。已试 {n_seen} 个表达式。"
    best = max(scored, key=lambda a: abs(a.ic_train or 0.0))
    return (f"上一轮最佳: {best.expression} train_IC={best.ic_train:.4f} "
            f"(过护栏={best.passed_guardrails}); 已试 {n_seen} 个表达式。")


def _human_gate(state: AgentState) -> None:
    import sys
    if not sys.stdin.isatty():   # 非交互（CI/管道）跳过
        return
    print(f"[agent] 本轮候选 {len(state.candidates)} 个，回车继续...")
    with contextlib.suppress(EOFError):
        input()
