"""Agent 闭环主循环：只调度，业务逻辑在 nodes。"""
from __future__ import annotations

import contextlib
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
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import split_holdout
from factorzen.validation.multiple_testing import TrialLedger


@dataclass
class AgentResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int


def run_llm_agent(daily, llm_fn: LLMFn, *, n_rounds: int, seed: int, top_k: int = 5,
                  holdout_ratio: float = 0.2, human_review: bool = False,
                  patience: int | None = None) -> AgentResult:
    rng = np.random.default_rng(seed)  # noqa: F841 预留给未来随机选择，保证可复现入口
    mining_df, holdout_df, _hstart = split_holdout(daily, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)        # Agent 只见 mining 段
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    feedback = ""
    no_improve = 0
    last_cand_count = 0
    for round_i in range(n_rounds):
        # 自适应早停：连续 patience 轮无新 passed 候选则停（patience=None → 跑满，零回归）
        if patience is not None and round_i > 0:
            no_improve = 0 if len(state.candidates) > last_cand_count else no_improve + 1
            if no_improve >= patience:
                break
        last_cand_count = len(state.candidates)
        state = node_generate(state, llm_fn, daily=mining_df, bundle=bundle, feedback=feedback)
        state = node_evaluate(state, daily=mining_df, bundle=bundle)
        state = node_guardrails(state, daily=mining_df, holdout_df=holdout_df,
                                bundle=bundle, ledger=ledger, top_k=top_k)
        state = node_critic(state, llm_fn)
        if human_review:
            _human_gate(state)  # 打印候选 + 等输入（非交互/CI 跳过）
        state = node_reflect(state)
        feedback = _summarize_feedback(state)
    return AgentResult(state=state, candidates=state.candidates, n_trials=ledger.n_trials)


def _summarize_feedback(state: AgentState) -> str:
    if not state.attempts:
        return ""
    last = state.attempts[-1]
    return f"上轮最佳 train_IC={last.ic_train}; 已试 {len(state.seen_expressions)} 个表达式。"


def _human_gate(state: AgentState) -> None:
    import sys
    if not sys.stdin.isatty():   # 非交互（CI/管道）跳过
        return
    print(f"[agent] 本轮候选 {len(state.candidates)} 个，回车继续...")
    with contextlib.suppress(EOFError):
        input()
