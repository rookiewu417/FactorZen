"""Agent 闭环主循环：只调度，业务逻辑在 nodes。"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from factorzen.agents.nodes import (
    _print_rejections,
    node_critic,
    node_evaluate,
    node_finalize_guardrails,
    node_generate,
    node_guardrails,
    node_reflect,
)
from factorzen.agents.state import AgentState
from factorzen.agents.team_orchestrator import _prepare_segments, _to_date
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.client import LLMClientError
from factorzen.llm.generation import LLMFn
from factorzen.validation.multiple_testing import TrialLedger

_LOG = logging.getLogger(__name__)


def _step(msg: str) -> None:
    """过程提示 → stdout。挖掘由 CLI 触发，用户要看实时进度；不走 logging 免被默认级别吞掉。"""
    print(f"[mine-agent] {msg}", flush=True)


@dataclass
class AgentResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int
    # deflation 基准的尺度。缺了它，光凭 n_trials 复算不出候选的 dsr_pvalue
    # （`expected_max_sharpe ∝ sqrt(sharpe_variance)`）——manifest 就无法自证。
    # 默认 nan：中途的 `on_round_end` 检查点尚无最终 basis，写 null 比写一个假值诚实。
    sharpe_variance: float = float("nan")


def run_llm_agent(daily, llm_fn: LLMFn, *, n_rounds: int, seed: int, top_k: int = 5,
                  holdout_ratio: float = 0.2, human_review: bool = False,
                  patience: int | None = None,
                  heal_rounds: int = 2,
                  on_round_end: Callable[[AgentResult], None] | None = None,
                  llm_failure_patience: int = 3,
                  eval_start: str | None = None, profile=None,
                  library_orthogonal: bool = True,
                  library_root: str | None = None) -> AgentResult:
    """跑 n_rounds 轮 Agent 挖掘闭环。

    ``on_round_end``：每个**成功**轮次结束时以当前累积结果回调，供调用方增量落盘。
    没有它，进程在第 N 轮崩溃会让前 N-1 轮的候选全部丢失（manifest 只在返回后才写）。

    ``llm_failure_patience``：连续多少轮 LLM 不可用即提前终止。单轮的 ``LLMClientError``
    （client 层重试已耗尽，或 422 这类不可重试错误）只跳过该轮，不崩整个 session；
    但 LLM 持续不可用时空转跑满 n_rounds 毫无意义。计数器在成功轮重置，
    否则零散抖动会被累计成「持续不可用」。

    只吞 ``LLMClientError``。其余异常（代码 bug、磁盘满）照常冒泡——静默吞掉它们
    会把真实缺陷伪装成「LLM 抖动」。

    ``eval_start``：``"YYYYMMDD"``，训练段的干净起点（预热段的边界）。``daily`` 先按它裁
    （`_prepare_segments`，与 team 路径共用）再 split holdout，`mining_df`/`holdout_df`/
    `bundle` 全部建在干净样本上；完整的 ``daily`` 只作为求值时的预热前缀。``None``
    （默认）时退化为旧行为，对现有调用方零回归。
    """
    rng = np.random.default_rng(seed)  # noqa: F841 预留给未来随机选择，保证可复现入口
    mining_df, holdout_df, holdout_start = _prepare_segments(
        daily, eval_start=eval_start, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)        # Agent 只见 mining 段
    _step(f"数据切分 ▸ 训练 {mining_df['trade_date'].n_unique()} 天 / "
          f"holdout {holdout_df['trade_date'].n_unique()} 天")
    _eval_start_date = _to_date(eval_start) if eval_start is not None else None
    _eval_end_date = mining_df["trade_date"].max() if eval_start is not None else None
    # 叶子历史预算（算一次，逐轮复用）：在含预热前缀的完整帧上算，只留短于预热前缀
    # （AGENT_WARMUP_LOOKBACK）的叶子回灌 LLM。须与预热门同一套 _preprocess_daily 帧算，
    # 才能与 have 判定逐值一致（见 leaf_warmup_budgets）。eval_start=None → None，零回归。
    # 市场上下文（profile=None → A 股默认）：叶子集/映射/市场名，供 budgets 与各 node 透传。
    from factorzen.agents.nodes import AgentContext
    ctx = AgentContext.from_profile(profile)
    # 开局摘死叶（与 team / M1 同口径；prep 帧与求值一致，避免派生列误摘）
    from factorzen.agents.evaluation import _preprocess_daily
    from factorzen.discovery.leaf_health import (
        apply_leaf_exclusion,
        filter_leaves_by_holdout_coverage,
        log_excluded_leaves,
    )
    _kept, excluded_leaves = filter_leaves_by_holdout_coverage(
        _preprocess_daily(daily, profile), list(ctx.leaf_names), holdout_start,
        leaf_map=ctx.leaf_map,
    )
    log_excluded_leaves(excluded_leaves, prefix="mine-agent")
    ctx.leaf_names, ctx.leaf_map = apply_leaf_exclusion(
        list(ctx.leaf_names), ctx.leaf_map, excluded_leaves,
    )
    leaf_budgets: dict[str, int] | None = None
    if _eval_start_date is not None:
        from factorzen.agents.evaluation import _preprocess_daily
        from factorzen.discovery.expression import leaf_warmup_budgets
        from factorzen.pipelines.factor_mine import AGENT_WARMUP_LOOKBACK
        _all_budgets = leaf_warmup_budgets(
            _preprocess_daily(daily, profile), _eval_start_date, ctx.leaf_names,
            leaf_map=ctx.leaf_map)
        leaf_budgets = {k: v for k, v in _all_budgets.items() if v < AGENT_WARMUP_LOOKBACK}
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    feedback = ""
    no_improve = 0
    last_cand_count = 0
    llm_failures = 0

    # 库级正交：session 开始物化一次（与 node_guardrails 同帧 = holdout）。空库 → 零回归。
    lib_pool: dict = {}
    library_covered: list[str] | None = None
    if library_orthogonal:
        try:

            from factorzen.agents.evaluation import _preprocess_daily
            from factorzen.discovery.factor_library import (
                DEFAULT_ROOT,
                build_library_pool,
                library_covered_expressions,
            )
            market = getattr(profile, "name", None) or "ashare"
            lib_root = library_root or DEFAULT_ROOT
            _prepped = _preprocess_daily(daily, profile)
            _hold_start = holdout_df["trade_date"].min()
            lib_pool = build_library_pool(
                market, _prepped, ctx.leaf_map, root=lib_root, eval_start=_hold_start,
            )
            covered = library_covered_expressions(market, k=10, root=lib_root)
            library_covered = covered or None
            state.library_pool_size = len(lib_pool)
            if lib_pool:
                _step(f"库级正交 ▸ 物化 {len(lib_pool)} 个 active 库因子")
        except Exception as exc:
            _LOG.warning("库池物化失败，本 session 跳过库级正交: %s: %s",
                         type(exc).__name__, exc)
            lib_pool, library_covered = {}, None

    for round_i in range(n_rounds):
        # 自适应早停：连续 patience 轮无新 passed 候选则停（patience=None → 跑满，零回归）
        if patience is not None and round_i > 0:
            no_improve = 0 if len(state.candidates) > last_cand_count else no_improve + 1
            if no_improve >= patience:
                _step(f"连续 {patience} 轮无新候选 → 提前早停（已跑 {round_i} 轮）")
                break
        last_cand_count = len(state.candidates)
        _step(f"── 第 {round_i + 1}/{n_rounds} 轮 " + "─" * 40)
        try:
            _step("  ① 生成假设 + 表达式")
            # leaf_guidance=None：M5 无跨 session index；注入函数与 team 共用，
            # 有 guidance 时由调用方/扩展接线传入。ctx 透传以尊重开局摘死叶。
            state = node_generate(state, llm_fn, daily=mining_df, bundle=bundle,
                                  feedback=feedback, heal_rounds=heal_rounds,
                                  leaf_budgets=leaf_budgets, profile=profile,
                                  library_covered=library_covered, ctx=ctx)
            _step(f"  ② 评估 {len(getattr(state, '_pending', []))} 个候选表达式")
            # None-gating：eval_start=None（旧调用方默认）时 daily/eval_start/eval_end
            # 的组合与之前逐字节相同的裸调用；非 None 时在完整帧 daily 上求值，裁剪到
            # [eval_start, eval_end]（mining_df 此时已被 _prepare_segments 提前裁到
            # eval_start，不能拿它的起点当判据——见 task-1.4 CORRECTION）。
            state = node_evaluate(state, daily=mining_df, bundle=bundle,
                                  eval_start=_eval_start_date, eval_end=_eval_end_date,
                                  warmup_daily=daily, profile=profile)
            _step("  ③ 防过拟合护栏（DSR / holdout / CI / 去相关 / 库级正交）")
            state = node_guardrails(state, daily=mining_df, holdout_df=holdout_df,
                                    bundle=bundle, ledger=ledger, top_k=top_k,
                                    warmup_daily=daily,   # holdout 扩窗预热用完整帧
                                    eval_start=_eval_start_date,  # 池级 PBO 的 None-gating
                                    profile=profile, lib_pool=lib_pool)
            _print_rejections("mine-agent", state)
            _step("  ④ Critic 审计")
            state = node_critic(state, llm_fn)
        except LLMClientError as exc:
            llm_failures += 1
            # 丢弃本轮未评估的暂存表达式；node_reflect 未执行，故此处补推进 iteration
            state._pending = []  # type: ignore[attr-defined]
            state.iteration += 1
            _step(f"  ⚠ LLM 不可用（连续第 {llm_failures} 次），跳过本轮")
            _LOG.warning("第 %d 轮 LLM 不可用（连续第 %d 次），跳过本轮: %s",
                         round_i, llm_failures, exc)
            if llm_failures >= llm_failure_patience:
                _step(f"  ✖ 连续 {llm_failures} 轮 LLM 不可用 → 提前终止"
                      f"（已产出 {len(state.candidates)} 个候选）")
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
    # 收尾复核：早轮候选此前按「截至当轮」的 N 定 p，门槛偏松。用最终 basis 统一重判。
    _step("收尾复核：以最终 N 统一重判候选 DSR")
    basis = node_finalize_guardrails(state, daily=mining_df, bundle=bundle, profile=profile)
    return AgentResult(state=state, candidates=state.candidates, n_trials=ledger.n_trials,
                       sharpe_variance=basis.sharpe_variance)


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
