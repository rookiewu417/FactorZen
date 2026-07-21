"""多角色团队编排：Librarian→Hypothesis→Coder→Evaluator→Critic 流水线 + 否决回路。"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.manifest import dump_manifest, json_safe_float
from factorzen.agents.nodes import (
    AgentContext,
    _print_rejections,
    node_finalize_guardrails,
    node_guardrails,
)
from factorzen.agents.roles.coder import (
    decompose_tasks,
    revise_expressions,
    revise_from_error,
    write_expressions,
)
from factorzen.agents.roles.critic import CriticVerdict, critique
from factorzen.agents.roles.hypothesis import (
    format_structured,
    propose_hypotheses,
    propose_structured,
)
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
from factorzen.core.experiment import get_git_sha
from factorzen.discovery.evaluation import evaluate_expressions, make_health_check
from factorzen.discovery.expression import clamp_window_literals, parse_expr, to_expr_string
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.client import LLMClientError
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import holdout_boundary, split_holdout
from factorzen.validation.multiple_testing import TrialLedger

_LOG = logging.getLogger(__name__)

_T = TypeVar("_T")


def _step(msg: str) -> None:
    """过程提示 → stdout。挖掘由 CLI 触发，用户要看实时进度；不走 logging 免被默认级别吞掉。"""
    print(f"[mine-team] {msg}", flush=True)


def _llm_map(callables: list[Callable[[], _T]], workers: int) -> list[_T]:
    """执行一组零参可调用，**按提交序**返回结果。

    ``workers <= 1``：纯串行列表推导，**不**实例化 ``ThreadPoolExecutor``——既有有状态
    scripted ``llm_fn`` 依赖调用序且非线程安全，API 缺省必须零回归。

    ``workers > 1``：线程池并发；worker 内捕获异常回传，装配阶段按提交序遇到的**第一个**
    异常重新抛出（保持 round 级 ``except LLMClientError`` / llm_failure_patience 语义）。
    生成阶段不得写共享 state——调用方保证 callables 只产纯结果。
    """
    if not callables:
        return []
    if workers <= 1:
        return [fn() for fn in callables]

    def _capture(fn: Callable[[], _T]) -> tuple[bool, _T | BaseException]:
        try:
            return True, fn()
        except BaseException as exc:  # 契约：装配期按序重抛
            return False, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_capture, fn) for fn in callables]
        # 按提交序收集（不用 as_completed）——产物与完成序无关
        outcomes = [fut.result() for fut in futures]

    results: list[_T] = []
    for ok, val in outcomes:
        if not ok:
            raise val  # type: ignore[misc]
        results.append(val)  # type: ignore[arg-type]
    return results


@dataclass
class TeamResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int
    rounds_log: list[dict] = field(default_factory=list)
    # deflation 基准的尺度；缺了它光凭 n_trials 复算不出 dsr_pvalue。
    # 默认 nan：中途的 on_round_end 检查点尚无最终 basis，写 null 比写假值诚实。
    sharpe_variance: float = float("nan")
    # holdout 覆盖不足被摘除的叶子 → {leaf: coverage}；manifest 可审计。
    excluded_leaves: dict[str, float] = field(default_factory=dict)
    # session 末 lift 钩子产物（partial 检查点保持默认空）
    n_lift_queue: int = 0
    lift_group: dict | None = None
    lift_results: list = field(default_factory=list)
    lift_admissions: dict = field(default_factory=lambda: {
        "added_active": 0, "added_probation": 0,
    })
    n_lift_evaluated: int = 0
    lift_dropped_coverage: list = field(default_factory=list)
    lift_error: str | None = None
    # campaign trial family：跨 session DSR N 累计（partial 检查点保持默认）
    campaign_id: str | None = None
    prior_n_trials: int = 0
    prior_n_sessions: int = 0
    # finalize 所用 basis 的 n_trials（prior ∪ session 唯一）；无 prior 时 ≈ 本 session
    n_trials_family: int = 0
    # 日内 Feature Scout 审计块（flag-off 时 None，manifest 不写）
    intraday_scout: dict | None = None


def _normalize(expr: str, leaf_map: dict[str, str] | None = None) -> str:
    """规范化表达式串用于去重。``leaf_map``（默认 None → A 股）必须与 `evaluate_expressions`
    产出 `seen_expressions` 时用的同一套映射一致——否则 crypto 表达式在这里 parse 失败退回
    原串、与规范化的 `seen_expressions` 失配，同一 trial 被评估两次致 N over-count。"""
    try:
        return to_expr_string(parse_expr(expr, leaf_map))
    except ValueError:
        return expr


def _task_text(task: dict) -> str:
    """把分解出的因子任务渲染成供 Coder 翻译的方向文本（名称 + 描述 + 构造理由）。"""
    parts = [task.get("name", ""), task.get("description", "")]
    if task.get("rationale"):
        parts.append(f"构造理由: {task['rationale']}")
    return "；".join(p for p in parts if p)


def _to_date(s: str):
    """'YYYYMMDD' → datetime.date。"""
    import datetime as _dt
    return _dt.datetime.strptime(s, "%Y%m%d").date()


def _prepare_segments(daily: pl.DataFrame, *, eval_start: str | None, holdout_ratio: float):
    """先裁到 [eval_start, end] 再切 holdout——预热段只作求值前缀，不进任何评估段。

    `split_holdout` 按整帧交易日切，若帧含预热段，mining_df 起点就是帧起点，
    预热段随 `DataBundle` 的 train 段进 IC 序列。`eval_start=None` 时退化为旧行为。
    """
    sample = daily if eval_start is None else daily.filter(pl.col("trade_date") >= _to_date(eval_start))
    return split_holdout(sample, holdout_ratio=holdout_ratio)


def _narrow_holdout_price_frame(holdout_df: pl.DataFrame) -> pl.DataFrame:
    """P5：长驻 holdout 只留键 + 价列（holdout_ic / residual hold_fwd 所需）。

    因子值走 warmup/session_prepped 扩窗求值；护栏不再依赖 holdout 全宽叶子。
    保留 open_adj/open 以便 ``exec_price_col`` 可实现口径（缺则 compute_fwd_returns 会炸）。
    """
    cols = ["trade_date", "ts_code"]
    for c in ("close_adj", "close", "open_adj", "open"):
        if c in holdout_df.columns:
            cols.append(c)
    return holdout_df.select(cols)


def _evaluate_and_record(state, exprs, hypothesis, *, daily, bundle, mem_seen,
                         eval_start=None, eval_end=None, profile=None, leaf_map=None,
                         prepped=None):
    """评估一批表达式（跳过 mem_seen 去重），写 AttemptRecord，返回本批新评估的结果列表。

    灵魂约束：此函数不碰 ledger，N 诚实记账由外层 node_guardrails 统一负责（每轮恰好一次）。

    ``eval_start``/``eval_end``：会话级 train 段边界（date，或 None）。**None-gating**：
    为 None 时原样转发 None 给 `evaluate_expressions`（等价裸调用，零回归）；非 None 时
    调用方须传 ``daily`` 为含预热前缀的完整帧——裁剪与预热门在 `evaluate_expressions`
    内部完成。调用方负责按会话级 `eval_start` 是否为 None 选择正确的 `daily`
    （mining_df 还是 warmup_daily），本函数只透传，不做二次判断。

    ``prepped``：session 级已 prep 帧（可选）；透传给 ``evaluate_expressions`` 避免重复 prep。
    """
    # **批内也要去重**：heal_rounds=0 时 heal_expressions 的去重不生效，多个 task 很容易
    # 翻译出同一表达式。重复评估会让 node_guardrails 把同一个 trial 记两次 → N over-count
    # （方向偏严，但记账不诚实），并向 index 写重复行。
    fresh: list[str] = []
    batch_seen: set[str] = set()
    for e in exprs:
        norm = _normalize(e, leaf_map)
        if norm in mem_seen or norm in state.seen_expressions or norm in batch_seen:
            continue
        batch_seen.add(norm)
        fresh.append(e)
    results = (
        evaluate_expressions(fresh, daily, bundle, eval_start=eval_start, eval_end=eval_end,
                             profile=profile, seen_fingerprints=state.seen_fingerprints,
                             prepped=prepped)
        if fresh else []
    )
    for r in results:
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"],
            turnover=r.get("turnover"), n_train=r.get("n_train"),
            nonzero_coverage=r.get("nonzero_coverage"),
            is_sparse=bool(r.get("is_sparse") or False),
            subset_ic_train=r.get("subset_ic_train"),
            subset_n_days_train=r.get("subset_n_days_train"),
            subset_mask_leaves=r.get("subset_mask_leaves"),
        ))
        state.seen_expressions.add(r["expression"])
    return results


def _run_one_round(
    state, llm_fn, *, index, ledger, rounds_log, mining_df, holdout_df, bundle,
    pending, seed, top_k, heal_rounds, structured, health, data_window, warmup_daily,
    eval_start=None, leaf_budgets=None, hypotheses_per_round=1, profile=None, ctx=None,
    lib_pool=None, library_covered=None, library_crowded=None,
    library_exprs=None, objective: str = "residual",
    llm_workers: int = 1, residual_projector=None,
    run_id: str | None = None, campaign_id: str | None = None,
    prepped=None,
    exec_lag: int = 0,
    exec_price_col: str | None = None,
    sleeve_gate: bool = True,
) -> dict | None:
    """跑一轮 Librarian→Hypothesis/Coder→Evaluator→Critic→Librarian。

    ``profile`` / ``ctx``：市场上下文（默认 None / A 股 `AgentContext()`，零回归）。``ctx``
    的 market/leaf_names/leaf_map 透传给 Hypothesis/Coder prompt、自愈、去重与护栏；
    ``profile`` 透传给评估/护栏的派生列与叶子映射。

    `state` / `ledger` / `rounds_log` / `index` 为可变对象，就地更新；返回下一轮的 Critic 反馈。
    抽成独立函数是为了让主循环能整轮 `try/except LLMClientError` 而不必把 120 行内联进 try 块。

    ``eval_start``：会话级 train 段起点（date，或 None，由 `run_team_agent` 解析一次后逐轮传入）。
    **None-gating**：为 None 时 train 段求值走裸 `evaluate_expressions(exprs, mining_df, bundle)`
    （旧调用者零回归）；非 None 时改在 ``warmup_daily``（完整帧）上求值、裁剪到
    ``[eval_start, mining_df 终点]``——`mining_df` 此时已被 `_prepare_segments` 提前裁到
    `eval_start`，不能直接把 `mining_df["trade_date"].min()` 当判据（`eval_start=None`
    时它就是帧起点，会让预热门把可用预热样本数误判成 0，见 task-1.4 CORRECTION）。
    """
    if ctx is None:
        ctx = AgentContext()
    _step("  ① Librarian 检索历史经验（known valid/invalid + leaf_guidance + library）")
    # 每轮重算 leaf_stats：本 session 刚写入的失败也会进入后续轮次的挖穿/未探索。
    # leaf_names=ctx.leaf_names（leaf_health 摘除后的存活叶），死叶不进任一侧。
    # library_covered 在 session 开始预构建，逐轮复用（库文件本 session 不改）。
    # library_exprs：库内全部表达式，供挖穿判定兜底（≠ library_covered，后者按族截断
    # 只为喂 LLM；拿截断列表判挖穿会漏掉绝大多数库因子，等于没兜底）。
    rec = recall(
        index, k=5, data_window=data_window, leaf_names=list(ctx.leaf_names),
        library_covered=library_covered, library_exprs=library_exprs,
    )
    tasks: list[dict] = []

    # ②/③ Hypothesis +（任务分解）+ Coder（依据上一轮 Critic 反馈，跨轮）
    # 产出 hyp_batches: [(假设, 表达式集)]。**revise 批次与新假设同轮并行**：
    # 修订不再独占整轮——GPT 类引擎的候选常被 Critic 连环判 revise_expr，
    # 纯修订轮会把吞吐塌缩（实测 19→3→2）；修订价值保留，但不得挤占新假设配额。
    #
    # LLM 墙钟并行（llm_workers>1）：彼此独立的调用经 `_llm_map` 并发；
    # workers=1 走纯串行、不进 executor（有状态 scripted llm 零回归）。
    # 拓扑：revise∥propose → 跨假设「分解→write」链并发（链内 decompose 先于 write，
    # 同假设多 task 的 write 也可并发）→ heal 批并发 → 评估串行 → 预热 revise 并发。
    # Critic/Evaluator/护栏不动。
    hyp_batches: list[tuple[str, list[str]]] = []
    # revise_expr 的 reason 同样作为 feedback 供新假设避坑（原语义仅 revise_hypothesis）
    fb = pending["reason"] if pending and pending["kind"] in (
        "revise_hypothesis", "revise_expr") else ""

    def _do_revise() -> list[str]:
        assert pending is not None
        return revise_expressions(
            pending["hypothesis"], pending["exprs"], pending["reason"], llm_fn,
            leaf_budgets=leaf_budgets, market=ctx.market, leaf_names=ctx.leaf_names)

    def _do_propose() -> list[str]:
        if structured:
            shyps = propose_structured(
                llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
                feedback=fb, n=hypotheses_per_round, market=ctx.market,
                leaf_guidance=rec.leaf_guidance, library_covered=rec.library_covered,
                lift_rejected=rec.lift_rejected, library_crowded=library_crowded,
            )
            return [format_structured(h) for h in shyps]
        return propose_hypotheses(
            llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
            feedback=fb, n=hypotheses_per_round, market=ctx.market,
            leaf_guidance=rec.leaf_guidance, library_covered=rec.library_covered,
            lift_rejected=rec.lift_rejected, library_crowded=library_crowded,
        )

    # 修订批 ∥ propose（互相独立）；提交序：revise（若有）→ propose
    gen_fns: list[Callable[[], list[str]]] = []
    gen_tags: list[str] = []
    if pending and pending["kind"] == "revise_expr":
        _step("  ②→③ Coder 依 Critic 反馈修订表达式（与新假设并行）")
        gen_fns.append(_do_revise)
        gen_tags.append("revise")
    _step("  ② Hypothesis 提假设"
          + (f"（×{hypotheses_per_round}）" if hypotheses_per_round > 1 else "")
          + ("（结构化：机制/预期符号/证伪）" if structured else ""))
    gen_fns.append(_do_propose)
    gen_tags.append("propose")
    gen_outs = _llm_map(gen_fns, llm_workers)
    hyps: list[str] = []
    for tag, out in zip(gen_tags, gen_outs, strict=True):
        if tag == "revise":
            if out and pending is not None:
                hyp_batches.append((str(pending["hypothesis"]), list(out)))
        else:
            hyps = list(out)
    if not hyps and not hyp_batches:
        _step("  · Hypothesis 未产出假设，跳过本轮")
        state.iteration += 1
        return None

    # 逐假设独立走「任务分解 → Coder 翻译」。跨假设可并发；链内 decompose 先于其 write；
    # 同假设多 task 的 write 之间也可并发。结果按假设提交序装配。
    def _hyp_chain(h: str) -> tuple[str, list[dict], list[str]]:
        h_tasks = decompose_tasks(h, llm_fn) if structured else []
        if h_tasks:
            def _mk_write(task: dict) -> Callable[[], list[str]]:
                def _run() -> list[str]:
                    return write_expressions(
                        _task_text(task), llm_fn, avoid=rec.known_invalid,
                        leaf_budgets=leaf_budgets, market=ctx.market,
                        leaf_names=ctx.leaf_names)
                return _run
            expr_lists = _llm_map([_mk_write(t) for t in h_tasks], llm_workers)
            h_exprs = [e for xs in expr_lists for e in xs]
        else:
            # 未启用分解、或 LLM 分解失败（空 tasks）→ 降级为整条假设直译，不空转
            h_exprs = write_expressions(
                h, llm_fn, avoid=rec.known_invalid,
                leaf_budgets=leaf_budgets, market=ctx.market, leaf_names=ctx.leaf_names)
        return h, h_tasks, h_exprs

    def _mk_chain(hyp: str) -> Callable[[], tuple[str, list[dict], list[str]]]:
        return lambda: _hyp_chain(hyp)

    chain_outs = _llm_map([_mk_chain(h) for h in hyps], llm_workers)
    for h, h_tasks, h_exprs in chain_outs:
        tasks.extend(h_tasks)
        hyp_batches.append((h, h_exprs))
    if hyps:
        _step(f"  ③ Coder 翻译表达式（{len(hyps)} 假设"
              + (f" / {len(tasks)} 子任务" if tasks else "") + "）")
    n_unknown_op_dropped = 0
    if heal_rounds > 0:
        from factorzen.agents.self_heal import heal_expressions

        def _heal_one(item: tuple[str, list[str]]) -> tuple[str, list[str], int]:
            h, ex = item
            # 每批独立 stats，避免 llm_workers>1 时共享 dict 竞态丢计数
            local_stats: dict[str, int] = {"n_unknown_op_dropped": 0}
            healed = heal_expressions(
                ex, h, llm_fn, max_rounds=heal_rounds, health_check=health,
                leaf_map=ctx.leaf_map, market=ctx.market, leaf_names=ctx.leaf_names,
                stats=local_stats,
            )
            return h, healed, int(local_stats.get("n_unknown_op_dropped", 0))

        def _mk_heal(batch: tuple[str, list[str]]) -> Callable[[], tuple[str, list[str], int]]:
            return lambda: _heal_one(batch)

        heal_outs = _llm_map([_mk_heal(b) for b in hyp_batches], llm_workers)
        hyp_batches = [(h, ex) for h, ex, _n in heal_outs]
        n_unknown_op_dropped = sum(_n for _h, _ex, _n in heal_outs)

    # ③½ exhausted 叶子硬过滤（coder/heal 产出 → 进评估前；None/空 exhausted → 直通）
    # 配额 dict 本轮新建，跨 hyp_batches 共享（混族组合配额内放行）。
    from factorzen.agents.scout_support import filter_exhausted_expressions

    exh_set = set(rec.exhausted_leaves or []) if rec.exhausted_leaves else None
    quota_used: dict[str, int] = {}
    n_exhausted_filtered = 0
    if exh_set:
        filtered_batches: list[tuple[str, list[str]]] = []
        for h, h_exprs in hyp_batches:
            kept_e, n_d = filter_exhausted_expressions(
                h_exprs,
                exhausted=exh_set,
                leaf_map=ctx.leaf_map,
                quota_used=quota_used,
            )
            n_exhausted_filtered += n_d
            filtered_batches.append((h, kept_e))
        hyp_batches = filtered_batches
        if n_exhausted_filtered:
            _step(f"  ③½ exhausted 硬过滤 ▸ 丢弃 {n_exhausted_filtered} 条纯重挖/超配额")

    # ④ Evaluator：逐假设评估（跨 session + session 内去重）+ 预热错误回灌（只一轮，B3）
    # _evaluate_and_record 不碰 ledger；node_guardrails 本轮恰好一次（N 诚实）。
    # None-gating（非 None 才切到 warmup_daily + 段边界，None 时裸调用 mining_df，
    # 零回归）：gate 在会话级 eval_start 本身，不能用 mining_df.min() 判断——
    # eval_start=None 时 mining_df 就是帧起点，误用会让预热门把整段判成 0 可用预热。
    if eval_start is not None:
        ev_daily, ev_end = warmup_daily, mining_df["trade_date"].max()
    else:
        ev_daily, ev_end = mining_df, None
    results: list[dict] = []
    n_window_clamped = 0
    warm_budget = 6   # 每轮预热回灌上限 6 条（跨假设共享），控 LLM 成本
    for h, h_exprs in hyp_batches:
        # W5b：评估前按 leaf_budgets 钳制过大窗口字面量（显式；budgets 空/None → 不动）
        if leaf_budgets:
            clamped: list[str] = []
            for e in h_exprs:
                ce, did = clamp_window_literals(e, leaf_budgets, ctx.leaf_map)
                if did:
                    n_window_clamped += 1
                clamped.append(ce)
            h_exprs = clamped
        _step(f"  ④ Evaluator 评估 {len(h_exprs)} 个表达式")
        h_results = _evaluate_and_record(
            state, h_exprs, h, daily=ev_daily, bundle=bundle, mem_seen=rec.seen,
            eval_start=eval_start, eval_end=ev_end, profile=profile, leaf_map=ctx.leaf_map,
            prepped=prepped,
        )
        results += h_results
        # B3 预热错误回灌：把「预热不足」诊断（连同 leaf_budgets）回灌 Coder 修正，修正版
        # 并入本轮 results。**只回灌一轮**——修正版仍预热不足就认栽（error 落盘，下轮
        # negative recall 自然规避）。仅在 eval_start 非 None（预热门生效）时触发。
        # 回灌的 attempts 进 state.attempts → DeflationBasis.from_ir_pool / ledger 自动涵盖，
        # 不额外手动记账（多轮累积计数陷阱）。
        if eval_start is not None and warm_budget > 0:
            warm_errs = [r for r in h_results
                         if r["error"] and "预热不足" in r["error"]][:warm_budget]
            warm_budget -= len(warm_errs)
            # revise_from_error 逐条独立可并发；结果按提交序展平（与串行 extend 序一致）
            def _mk_refeed(row: dict, hyp: str) -> Callable[[], list[str]]:
                def _run() -> list[str]:
                    return revise_from_error(
                        hyp, row["expression"], row["error"], llm_fn,
                        leaf_budgets=leaf_budgets, market=ctx.market,
                        leaf_names=ctx.leaf_names)
                return _run

            refed_lists = _llm_map([_mk_refeed(r, h) for r in warm_errs], llm_workers)
            refed = [e for xs in refed_lists for e in xs]
            if refed:
                if leaf_budgets:
                    clamped_ref: list[str] = []
                    for e in refed:
                        ce, did = clamp_window_literals(e, leaf_budgets, ctx.leaf_map)
                        if did:
                            n_window_clamped += 1
                        clamped_ref.append(ce)
                    refed = clamped_ref
                # 预热回灌同样过 exhausted 硬闸（共享本轮配额）
                if exh_set:
                    refed, n_ref_drop = filter_exhausted_expressions(
                        refed, exhausted=exh_set, leaf_map=ctx.leaf_map,
                        quota_used=quota_used,
                    )
                    n_exhausted_filtered += n_ref_drop
                if refed:
                    _step(f"  ④+ 预热错误回灌 revise（{len(warm_errs)} 条 → {len(refed)} 修正）")
                    results += _evaluate_and_record(
                        state, refed, h, daily=ev_daily, bundle=bundle, mem_seen=rec.seen,
                        eval_start=eval_start, eval_end=ev_end, profile=profile,
                        leaf_map=ctx.leaf_map, prepped=prepped,
                    )
    # 代表假设/表达式：供 Critic stub 与 revise pending（多假设时取最后一个批次，同现状语义；
    # 纯修订轮——新假设为空但修订批次在——回退到修订批次的假设，不许空串）
    hypothesis = hyps[-1] if hyps else (hyp_batches[-1][0] if hyp_batches else "")
    exprs = hyp_batches[-1][1] if hyp_batches else []
    n_before = len(state.candidates)                       # Important 1: 护栏前快照
    _step("  ⑤ 防过拟合护栏（DSR / holdout / CI / 去相关 / 库级正交）")
    node_guardrails(
        state, daily=mining_df, holdout_df=holdout_df,
        bundle=bundle, ledger=ledger, top_k=top_k,
        warmup_daily=warmup_daily,   # holdout 扩窗预热用完整帧
        eval_start=eval_start,       # 池级 PBO 的 None-gating：None 时裸求值，零回归
        profile=profile,             # crypto 派生列 + 叶子映射；None 零回归
        lib_pool=lib_pool,           # 库级正交 + 残差面板（全窗物化；None/空 → 零回归）
        objective=objective,
        residual_projector=residual_projector,  # session 级 QR，全量残差 train IC 快路径
        prepped=prepped,             # P5：session 同源 prep，跳过护栏再 prep 全帧
        exec_lag=exec_lag,
        exec_price_col=exec_price_col,
        sleeve_gate=sleeve_gate,
    )
    _print_rejections("mine-team", state)
    new_cands = state.candidates[n_before:]                # Important 1/Minor 2: 本轮新增候选

    # ⑤ Critic：按 hypothesis 分组裁决（每假设一次 critique，控 LLM 成本）。
    # W5c：无本轮新增候选 → **不调** critique LLM，确定性 revise_hypothesis
    # （陈旧 stub 指标裁决与本轮无关，浪费一次 LLM）。
    # 多假设时不得整轮连坐：verdict 只回填同组 attempts，drop 只移同组候选。
    # next_pending / rounds_log["verdict"|"reason"] 取最后一组（= 代表假设为最后批次，零回归）。
    _step("  ⑥ Critic 裁决")
    group_verdicts: list[dict] = []
    critic_skipped = False
    n_fingerprint_dup = sum(1 for r in results if r.get("error") == "duplicate_fingerprint")
    if not new_cands:
        critic_skipped = True
        cand = {
            "expression": results[-1]["expression"] if results else (exprs[0] if exprs else ""),
            "hypothesis": hypothesis,
            "ic_train": results[-1]["ic_train"] if results else None,
            "ir_train": results[-1]["ir_train"] if results else None,
            "turnover": results[-1].get("turnover") if results else None,
        }
        # 从 attempts 补正交字段（W6；有则填，无则缺省——空轮已跳 LLM，仅保留形态一致）
        if results:
            _rexpr = cand["expression"]
            for a in reversed(state.attempts):
                if a.iteration == state.iteration and a.expression == _rexpr:
                    if getattr(a, "residual_ic_train", None) is not None:
                        cand["residual_ic_train"] = a.residual_ic_train
                    if getattr(a, "residual_holdout_ic", None) is not None:
                        cand["residual_holdout_ic"] = a.residual_holdout_ic
                    break
        verdict = CriticVerdict(
            "revise_hypothesis",
            "本轮无新候选(去重/过滤/评估失败),建议换方向",
        )
        round_expr = cand.get("expression", "")
        for a in state.attempts:
            if a.iteration == state.iteration and a.expression == round_expr:
                a.critic_verdict = verdict.verdict
        group_verdicts.append({
            "hypothesis": cand.get("hypothesis") or hypothesis,
            "verdict": verdict.verdict,
            "reason": verdict.reason,
        })
    else:
        # 按 hypothesis 分组，保持首次出现序（cand 必有 hypothesis，见 nodes.py cand_row）
        groups: dict[str, list[dict]] = {}
        for c in new_cands:
            groups.setdefault(c["hypothesis"], []).append(c)

        cand = new_cands[-1]   # 占位；循环末写为最后一组代表
        verdict = None         # type: ignore[assignment]
        drop_exprs: set[str] = set()
        for hyp_key, group_cands in groups.items():
            rep = group_cands[-1]   # 该组最后一个候选为代表
            # W6：候选 dict 已由 node_guardrails 写入 residual_*/max_corr_library（有则保留）
            # 若缺则从同 expression 的 AttemptRecord 回填
            if "residual_ic_train" not in rep or "max_corr_library" not in rep:
                for a in reversed(state.attempts):
                    if a.iteration == state.iteration and a.expression == rep.get("expression"):
                        if "residual_ic_train" not in rep and getattr(a, "residual_ic_train", None) is not None:
                            rep = {**rep, "residual_ic_train": a.residual_ic_train,
                                   "residual_holdout_ic": getattr(a, "residual_holdout_ic", None)}
                        break
            v = critique(rep, llm_fn, lift_rejected=rec.lift_rejected)
            group_verdicts.append({
                "hypothesis": hyp_key,
                "verdict": v.verdict,
                "reason": v.reason,
            })
            group_exprs = {c["expression"] for c in group_cands}
            for a in state.attempts:
                if a.iteration == state.iteration and a.expression in group_exprs:
                    a.critic_verdict = v.verdict
            if v.verdict == "drop":
                drop_exprs |= group_exprs
            cand, verdict = rep, v

        # drop：按 expression 匹配且 index>=n_before 过滤重建（禁止 del [n_before:] 整段连坐）
        # 否决回路：drop 不得进 known_valid；passed_guardrails 是不可变事实，不由 verdict 改写
        # （见 ExperimentIndex._VETOED_VERDICTS / known_valid 读 critic_verdict）。
        if drop_exprs:
            state.candidates = [
                c for i, c in enumerate(state.candidates)
                if not (i >= n_before and c["expression"] in drop_exprs)
            ]
            new_cands = [c for c in new_cands if c["expression"] not in drop_exprs]

    assert verdict is not None  # stub 或 groups 非空时必赋值

    # leaf_guidance 摘要：可复现审计（挖穿/未探索列表；None 时不落假值）
    _lg = rec.leaf_guidance
    _lg_summary = (
        {
            "exhausted": list(_lg.get("exhausted") or []),
            "unexplored": list(_lg.get("unexplored") or []),
        }
        if _lg is not None else None
    )
    rounds_log.append({
        "round": state.iteration,
        # 多假设时记全部（"；" 连接）；单假设时即该假设字符串（零回归）。
        # rounds_log.hypothesis 只进 manifest 溯源，无下游按单字符串解析（per-attempt 归属
        # 由 AttemptRecord.hypothesis 各自承载），故连接安全。
        "hypothesis": "；".join(hyps),
        "tasks": tasks,                       # 步2 产物，实验溯源用（非 structured 轮为 []）
        "expressions": [r["expression"] for r in results],
        "verdict": verdict.verdict,           # 最后一组（消费方零回归）
        "reason": verdict.reason,
        "verdicts": group_verdicts,           # 全组按组序，审计用
        "leaf_guidance": _lg_summary,
        "n_exhausted_filtered": n_exhausted_filtered,
        "n_fingerprint_dup": n_fingerprint_dup,
        "n_unknown_op_dropped": n_unknown_op_dropped,
        "n_window_clamped": n_window_clamped,
        "critic_skipped": critic_skipped,
    })

    # 最后一组 verdict → 下一轮 feedback（跨轮；不在本轮重跑护栏，避免 N 三角和）
    # drop 的 candidates/new_cands 清理已在分组循环完成；stub 无新增候选可删。
    if verdict.verdict == "drop":
        next_pending = None
    elif verdict.verdict == "revise_expr":
        # 定位代表候选所属假设 + 该假设的表达式集（多假设时归位到正确批次；
        # 单假设时 cand_h == hypothesis、cand_exprs == exprs，零回归）。
        cand_h = cand.get("hypothesis") or hypothesis
        cand_exprs = next((ex for h, ex in reversed(hyp_batches) if h == cand_h), exprs)
        next_pending = {
            "kind": "revise_expr",
            "hypothesis": cand_h,
            "exprs": cand_exprs,
            "reason": verdict.reason,
        }
    elif verdict.verdict == "revise_hypothesis":
        next_pending = {"kind": "revise_hypothesis", "reason": verdict.reason}
    else:
        next_pending = None

    # ⑥ Librarian：本轮 attempts 写 experiment_index（含 holdout_ic 回填）
    round_attempts = [a for a in state.attempts if a.iteration == state.iteration]
    record(
        index,
        round_attempts,
        run_id=run_id if run_id is not None else f"team_{seed}",
        candidates=new_cands,
        data_window=data_window,
        campaign_id=campaign_id,
    )
    state.iteration += 1
    return next_pending


def run_team_agent(
    daily,
    llm_fn: LLMFn,
    *,
    n_rounds: int,
    seed: int,
    index_path: str,
    top_k: int = 5,
    holdout_ratio: float = 0.2,
    patience: int | None = None,
    heal_rounds: int = 2,
    structured: bool = False,
    on_round_end: Callable[[TeamResult], None] | None = None,
    llm_failure_patience: int = 3,
    data_window: dict | None = None,
    eval_start: str | None = None,
    hypotheses_per_round: int = 1,
    profile=None,
    update_library: bool = True,
    library_root: str | None = None,
    horizon: int = 1,
    library_orthogonal: bool = True,
    objective: str = "residual",
    llm_workers: int = 1,
    auto_lift: bool = True,
    lift_se_mult: float = 1.0,
    lift_workers: int | None = None,  # None→run_lift_tests 按可用内存自适应
    # 测试注入：session 末 lift 钩子（materialize / active 面板 / ret）
    lift_materialize_candidate=None,
    lift_active_factor_dfs: dict | None = None,
    lift_ret_df=None,
    # 跨 session DSR N 累计：从 experiment_index 重建同 campaign 历史 trial 池。
    # CLI 旗标由主控后补；测试可关以验证零回归。
    campaign_prior_enabled: bool = True,
    # 测试注入：固定 session run_id（None → team_{seed}_{uuid8}，每次调用唯一）
    run_id: str | None = None,
    # 日内 Feature Scout：每轮 LLM 提案 K 个 bar 表达式 → 物化 → 筛 → 注入 session；
    # 仅被准入因子引用的 ix_* 在 session 末永久化。默认 False → 零开销零回归。
    intraday_scout: bool = False,
    scout_k: int = 4,
    scout_max_leaves: int = 12,
    scout_freq: str = "5min",
    scout_base_dir: str | Path | None = None,  # 测试隔离 registry/缓存；生产 None
    pool_cache_dir: str | None = None,
    # 成交口径：0 = t 日收盘成交（默认，向后兼容但**不可实现**——算信号需 t 日收盘价）；
    # 1 + "open_adj" = t+1 开盘成交（可实现）。贯穿护栏 / holdout / lift 裁决。
    exec_lag: int = 0,
    exec_price_col: str | None = None,
    # 稀疏因子 sleeve 旁路（子集 IC → lift_queue）；False=关闭，稠密路径不变。
    sleeve_gate: bool = True,
) -> TeamResult:
    """跨轮 feedback 流水线：每轮 Librarian→Hypothesis/Coder→Evaluator→Critic→Librarian。

    ``profile``：市场 profile（默认 None → A 股，零回归）。经 `AgentContext.from_profile`
    得叶子集/映射/市场名，逐层透传给 prompt/评估/护栏/预热预算。

    N 诚实：node_guardrails 每轮恰好调用一次（记本轮 N），不在同轮内多次调用（避免三角和）。
    holdout 隔离：mining_df 供角色/记忆，holdout_df 只进 node_guardrails。
    跨轮 feedback：Critic verdict=revise_expr → 下轮 Coder.revise；revise_hypothesis → 下轮 Hypothesis（带 feedback）。

    ``on_round_end``：每个**成功**轮次结束时回调，供调用方增量落盘——否则进程在第 N 轮崩溃
    会让前 N-1 轮的候选全部丢失（manifest 只在返回后才写）。

    ``llm_failure_patience``：连续多少轮 LLM 不可用即提前终止。单轮的 ``LLMClientError``
    （client 层重试已耗尽，或 422 这类不可重试错误）只跳过该轮；计数器在成功轮重置，
    否则零散抖动会被累计成「持续不可用」。只吞 ``LLMClientError``，其余异常照常冒泡。

    ``eval_start``：``"YYYYMMDD"``，训练段的干净起点（预热段的边界）。``daily`` 先按它裁
    （`_prepare_segments`）再 split holdout，`mining_df`/`holdout_df`/`bundle` 全部建在
    干净样本上；完整的 ``daily`` 只作为求值时的预热前缀。``None``（默认）时退化为旧行为
    （`split_holdout` 直接切整帧），对现有调用方零回归。

    ``llm_workers``：轮内彼此独立的 LLM 调用并发度。``1``（API 缺省）纯串行、不进
    ``ThreadPoolExecutor``——既有有状态 scripted ``llm_fn`` 零回归。``>1`` 时 futures
    按提交序装配，同 seed 产物与完成序无关。CLI ``fz mine team`` 缺省 4。

    ``pool_cache_dir``：可选库池 parquet 缓存目录（CLI ``pool-prebuild`` / ``--pool-subproc``）；
    传给 ``build_library_pool(cache_dir=...)``；命中则跳过进程内物化。默认 None 零回归。
    """
    # 峰值重排(v16 侦破):池构建前只算 holdout 边界日期(无帧),mining/holdout/bundle
    # 的物化挪到库池之后——池尾期(值列累积+分配器滞留)净省 ~3G(死点只差 1.5-2G)。
    # 边界与 _prepare_segments→split_holdout 单一口径,池后切分处有响亮断言防漂移。
    _dates_split = daily["trade_date"]
    if eval_start is not None:
        _dates_split = _dates_split.filter(_dates_split >= _to_date(eval_start))
    holdout_start = holdout_boundary(
        sorted(_dates_split.unique().to_list()), holdout_ratio)
    del _dates_split
    # 市场上下文（profile=None → A 股默认）：叶子集/映射/市场名，供 budgets 与逐轮透传。
    ctx = AgentContext.from_profile(profile)
    # session 级单次 prep：leaf_health / leaf_budgets / lib_pool / health / 每轮 evaluate /
    # lift_ctx 全部复用。scout 注入新 ix_* 列后必须重建（见循环内失效逻辑）。
    from factorzen.discovery.evaluation import _preprocess_daily
    session_prepped = _preprocess_daily(daily, profile)
    # 开局摘死叶：必须在与求值同一套 prep 帧上量覆盖（close→close_adj 别名 + 派生列），
    # 否则 ret_1d/vwap 等会被误判为「列不存在→覆盖 0」整批摘除。
    from factorzen.discovery.leaf_health import (
        apply_leaf_exclusion,
        filter_leaves_by_holdout_coverage,
        log_excluded_leaves,
    )
    _kept, excluded_leaves = filter_leaves_by_holdout_coverage(
        session_prepped, list(ctx.leaf_names), holdout_start,
        leaf_map=ctx.leaf_map,
    )
    log_excluded_leaves(excluded_leaves, prefix="mine-team")
    ctx.leaf_names, ctx.leaf_map = apply_leaf_exclusion(
        list(ctx.leaf_names), ctx.leaf_map, excluded_leaves,
    )
    _eval_start_date = _to_date(eval_start) if eval_start is not None else None
    # 叶子历史预算（只算一次，逐轮复用）：在含预热前缀的完整帧上算各叶子 eval_start 前的
    # 可用预热，只保留短于预热前缀（AGENT_WARMUP_LOOKBACK）的叶子回灌 LLM——引导它别对
    # 短历史叶（north_ratio 等）写超预热的长窗口而被预热门直接拒。须用 evaluate_expressions
    # 内部同一套 _preprocess_daily 帧算，才能与预热门判 have 逐值一致（见 leaf_warmup_budgets）。
    leaf_budgets: dict[str, int] | None = None
    if _eval_start_date is not None:
        from factorzen.discovery.expression import leaf_warmup_budgets
        _all_budgets = leaf_warmup_budgets(
            session_prepped, _eval_start_date, ctx.leaf_names,
            leaf_map=ctx.leaf_map)
        leaf_budgets = {k: v for k, v in _all_budgets.items() if v < AGENT_WARMUP_LOOKBACK}
    # ── campaign trial family：跨 session DSR N 累计 ──────────────────────────
    # ledger 仍每 session 从 0 起（本 session 诚实计数，manifest.n_trials 语义不变）。
    # session 末从 experiment_index 重建同 campaign 历史 IR 池（见 campaign_prior），
    # 与本 session 池做表达式级 union 后交给 node_finalize_guardrails —— N 与
    # sharpe_variance 同源（R8）。开关 campaign_prior_enabled（默认 True）。
    #
    # 另有一条更根本、且 N 累积管不到的问题：**holdout 跨 session 复用**。每个 session 都拿
    # 同一段 holdout 验收候选，跑 10 个 session 它就被看了 10 遍，不再是 OOS 而是第二个训练集。
    # 那是 OOS 污染，修法是预算/轮换而非累积 N。单列待评估。
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    index = ExperimentIndex(index_path)
    # health 诊断器挪至池后切分处创建(峰值重排;用 session_prepped 同源)
    rounds_log: list[dict] = []
    # 上一轮 Critic 反馈：{"kind", "hypothesis", "exprs", "reason"}
    pending: dict | None = None
    no_improve = 0
    last_cand_count = 0
    llm_failures = 0

    # 库级正交 + 残差面板：session 开始物化一次。
    # 残差目标需要 train∪holdout → 在完整 prepped 帧上物化（不再只裁 holdout）。
    # 空库/关开关 → lib_pool={}、library_covered=None，objective 退化 raw，行为与旧一致。
    lib_pool: Any = {}
    _lib_hash_at_pool: str | None = None  # lift 基线复用判据(池构建时的库文件 hash)
    library_covered: list[str] | None = None
    library_crowded: list[tuple[str, int]] | None = None
    library_exprs: list[str] | None = None   # 挖穿豁免证据源（在任记录表达式）
    market = getattr(profile, "name", None) or (
        (data_window or {}).get("market")) or "ashare"
    lib_root = library_root or str(Path(index_path).parent / "factor_library")
    if library_orthogonal:
        try:
            from factorzen.discovery.factor_library import (
                build_library_pool,
                library_covered_by_family,
            )
            # eval_start:求值后裁掉 504 天预热前缀(库因子自身滚动窗在完整帧上算,
            # 裁剪只去掉无消费的前缀行;train/holdout 残差与 lift 全在 eval 窗内)。
            # 全 A 9.57M 行时池行数 -24%(≈-1.6G),None 时不裁零回归。
            # cache_dir:CLI pool-prebuild 子进程产物;命中/未命中日志在 build_library_pool 内。
            # 池前序列(prep→剪叶→build_library_pool)与 _cmd_pool_prebuild 同源——改一侧必查另一侧。
            lib_pool = build_library_pool(
                market, session_prepped, ctx.leaf_map, root=lib_root,
                eval_start=_eval_start_date,
                cache_dir=pool_cache_dir,
                # 库含 python 记录时物化必需；与 _cmd_pool_prebuild 同口径
                universe=(data_window or {}).get("universe"),
            )
            # lift 基线复用判据(session 末):库文件内容 hash。记录级键集比较
            # 不可用——active 记录中恒有少数物化 skip(87 记录→84 物化),
            # 键集恒不等;而 lift 自建走同函数同库,skip 相同 → 文件未变即等价。
            from factorzen.discovery.factor_library import library_file_hash
            _lib_hash_at_pool = library_file_hash(market, lib_root)
            covered, crowded = library_covered_by_family(
                market, per_family=2, max_total=12, root=lib_root,
            )
            library_covered = covered or None
            library_crowded = crowded or None
            # 挖穿判定的成功证据源：库内**在任**记录（active/probation）的表达式。
            # ≠ library_covered（后者按族截断 per_family=2/max_total=12 只为喂 LLM，
            # 拿它判挖穿等于没兜底）。限定在任是有意的——库的生命周期即豁免的过期
            # 机制：那条因子被降级出 active/probation，该叶的豁免自动消失。
            from factorzen.discovery.factor_library import load_library
            library_exprs = [
                r.expression for r in load_library(market, root=lib_root)
                if r.expression and r.status in ("active", "probation")
            ] or None
            state.library_pool_size = len(lib_pool)
            if lib_pool:
                _step(f"库级正交 ▸ 物化 {len(lib_pool)} 个 active 库因子（root={lib_root}）")
        except Exception as exc:
            _LOG.warning("库池物化失败，本 session 跳过库级正交: %s: %s",
                         type(exc).__name__, exc)
            lib_pool, library_covered, library_crowded = {}, None, None
    state.objective = objective  # type: ignore[attr-defined]

    # ── 池后切分(峰值错峰 v20):切分在 X+Q(各 ~4G)建成**之前**、池值列之后——
    # 此刻基线无投影矩阵;切分完立即释放 raw daily,projector 期无 raw(-3.5G)。
    # 顺序:pool → 切分+release raw → projector,两大块错峰,峰值 ≈-3.3G。
    mining_df, holdout_df, _holdout_start2 = _prepare_segments(
        daily, eval_start=eval_start, holdout_ratio=holdout_ratio)
    assert _holdout_start2 == holdout_start, (
        f"holdout 边界漂移: 预池路径 {holdout_start} vs 切分 {_holdout_start2}"
    )
    # 成交口径贯穿护栏：quick_fitness / residual_ic / pool_pbo / ic_overfit
    # 全部基于 bundle.fwd_returns，故这里传错等于整轮护栏评的是另一种收益。
    bundle = DataBundle.build(
        mining_df, exec_lag=exec_lag, exec_price_col=exec_price_col)
    _step(f"数据切分 ▸ 训练 {mining_df['trade_date'].n_unique()} 天 / "
          f"holdout {holdout_df['trade_date'].n_unique()} 天")
    # P5：holdout 长驻窄投影（键+价）；因子求值走 warmup/session_prepped。
    holdout_df = _narrow_holdout_price_frame(holdout_df)
    # P5：scout off 时 raw daily 无其它消费（evaluate/护栏/lift 均用 session_prepped）；
    # 释放全宽 raw。scout on 保留供注入后重建。
    if not intraday_scout:
        daily = session_prepped
    # 求值层诊断器只建一次;与求值同源 session_prepped
    health = make_health_check(
        mining_df, profile=profile, leaf_map=ctx.leaf_map, prepped=session_prepped,
    ) if heal_rounds > 0 else None

    # residual 模式 + 库非空：session 开始建一次 ResidualProjector，整 session 复用
    # （多候选残差 train IC 近免费）。接线点在护栏前全量写 residual_ic_train / 选槽。
    residual_projector = None
    if objective == "residual" and lib_pool:
        try:
            from factorzen.discovery.residual import (
                ResidualProjector,
                build_library_panel,
            )
            _panel = build_library_panel(lib_pool)
            if _panel is not None and _panel.k > 0:
                residual_projector = ResidualProjector.from_panel(_panel)
                _step(f"残差投影 ▸ ResidualProjector 就绪（k={_panel.k}）")
        except Exception as exc:
            _LOG.warning("ResidualProjector 构建失败（本 session 残差走 lstsq）: %s: %s",
                         type(exc).__name__, exc)
            residual_projector = None

    # session 级唯一 run_id（同 seed 复用不再互斥排除历史）+ 完整统计问题 campaign_id
    session_run_id = run_id if run_id is not None else f"team_{seed}_{uuid.uuid4().hex[:8]}"
    from factorzen.discovery.campaign import campaign_key as _campaign_key
    from factorzen.discovery.guardrails import DEFAULT_GATE as _DEFAULT_GATE

    _dw0 = data_window or {}
    session_campaign_id = _campaign_key(
        market=market,
        universe=_dw0.get("universe"),
        start=_dw0.get("start"),
        end=_dw0.get("end"),
        holdout_ratio=holdout_ratio,
        objective=objective,
        horizon=horizon,
        gate=_DEFAULT_GATE,
    )

    # 日内 Feature Scout：仅 flag-on 建状态（flag-off 零开销）
    scout_state = None
    scout_promoted: list[str] = []
    if intraday_scout:
        from factorzen.agents.scout_support import ScoutState

        scout_state = ScoutState()
        _step(f"日内 Scout 启用 ▸ k={scout_k} max_leaves={scout_max_leaves} freq={scout_freq}")

    for round_i in range(n_rounds):
        # 自适应早停：连续 patience 轮无新 passed 候选则停（patience=None → 跑满，零回归）
        if patience is not None and round_i > 0:
            no_improve = 0 if len(state.candidates) > last_cand_count else no_improve + 1
            if no_improve >= patience:
                _step(f"连续 {patience} 轮无新候选 → 提前早停（已跑 {round_i} 轮）")
                break
        last_cand_count = len(state.candidates)
        _step(f"── 第 {round_i + 1}/{n_rounds} 轮 " + "─" * 40)
        # 轮初 scout：注入后重绑 mining/holdout/daily 再进 _run_one_round
        if scout_state is not None:
            from factorzen.agents.scout_support import run_scout_round
            from factorzen.discovery.intraday_expr import _frame_date_bounds

            _s0, _s1 = _frame_date_bounds(daily)
            scout_start = _s0 or (_dw0.get("start") or "")
            scout_end = _s1 or (_dw0.get("end") or "")
            if leaf_budgets is None and _eval_start_date is not None:
                leaf_budgets = {}
            try:
                _frames = run_scout_round(
                    llm_fn=llm_fn,
                    state=scout_state,
                    k=scout_k,
                    max_leaves=scout_max_leaves,
                    start=scout_start,
                    end=scout_end,
                    freq=scout_freq,
                    frames={"mining": mining_df, "holdout": holdout_df, "daily": daily},
                    ctx=ctx,
                    holdout_start=holdout_start,
                    eval_start=_eval_start_date,
                    leaf_budgets=leaf_budgets,
                    profile=profile,
                )
                mining_df = _frames["mining"]
                holdout_df = _narrow_holdout_price_frame(_frames["holdout"])
                daily = _frames["daily"]
                # scout 注入新 ix_* 列后，旧 session_prepped 缺列 → 必须重建
                if any(c not in session_prepped.columns for c in daily.columns):
                    session_prepped = _preprocess_daily(daily, profile)
                    if heal_rounds > 0:
                        health = make_health_check(
                            mining_df, profile=profile, leaf_map=ctx.leaf_map,
                            prepped=session_prepped,
                        )
                if scout_state.injected:
                    _step(f"  ⓪ Scout 注入叶: {scout_state.injected}")
            except Exception as exc:
                _LOG.warning("scout 轮次失败（跳过本轮注入）: %s: %s",
                             type(exc).__name__, exc)
        try:
            pending = _run_one_round(
                state, llm_fn, index=index, ledger=ledger, rounds_log=rounds_log,
                mining_df=mining_df, holdout_df=holdout_df, bundle=bundle,
                pending=pending, seed=seed, top_k=top_k,
                heal_rounds=heal_rounds, structured=structured, health=health,
                data_window=data_window, warmup_daily=daily,
                eval_start=_eval_start_date, leaf_budgets=leaf_budgets,
                hypotheses_per_round=hypotheses_per_round, profile=profile, ctx=ctx,
                lib_pool=lib_pool, library_covered=library_covered,
                library_crowded=library_crowded, library_exprs=library_exprs,
                objective=objective, llm_workers=llm_workers,
                residual_projector=residual_projector,
                run_id=session_run_id, campaign_id=session_campaign_id,
                prepped=session_prepped,
                exec_lag=exec_lag, exec_price_col=exec_price_col,
                sleeve_gate=sleeve_gate,
            )
        except LLMClientError as exc:
            llm_failures += 1
            state.iteration += 1   # 角色流水线未跑完，此处补推进以保持轮次语义一致
            pending = None
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
        if on_round_end is not None:
            on_round_end(TeamResult(state=state, candidates=state.candidates,
                                    n_trials=ledger.n_trials, rounds_log=rounds_log))

    # 收尾复核：早轮候选此前按「截至当轮」的 N 定 p，门槛偏松。用最终 basis 统一重判。
    # 若启用 campaign prior：先从 index 重建同族历史 trial 池（按 campaign_id 精确过滤，
    # 排除本 session_run_id 防双计），finalize 用 prior∪session 的 union N 做 deflation。
    _step("收尾复核：以最终 N 统一重判候选 DSR")
    prior = None
    campaign_id: str | None = None
    prior_n_trials = 0
    prior_n_sessions = 0
    if campaign_prior_enabled:
        try:
            from factorzen.discovery.campaign import campaign_prior

            dw = data_window or {}
            # 与库/index 分族同一套 market 解析：profile.name → data_window → ashare
            mkt = getattr(profile, "name", None) or dw.get("market") or "ashare"
            campaign_id = session_campaign_id
            # 本 session 行已逐轮 record 写入，须按唯一 session_run_id 排除，防双计
            prior = campaign_prior(
                index_path,
                market=mkt,
                universe=dw.get("universe"),
                start=dw.get("start"),
                end=dw.get("end"),
                exclude_run_ids={session_run_id},
                campaign_id=session_campaign_id,
            )
            if prior is not None:
                prior_n_trials = prior.n_trials
                prior_n_sessions = prior.n_sessions
                if prior.n_trials:
                    _step(f"  campaign prior ▸ N_hist={prior.n_trials} "
                          f"sessions={prior.n_sessions} id={campaign_id}")
        except Exception as exc:
            _LOG.warning(
                "campaign prior 构造失败，本 session 退化为 session 内 N: %s: %s",
                type(exc).__name__, exc,
            )
            prior = None
            # campaign_id 若已算成则仍落盘，便于审计；prior 计数字段保持 0

    before = {c["expression"] for c in state.candidates}
    basis = node_finalize_guardrails(
        state, daily=mining_df, bundle=bundle, profile=profile, prior=prior,
    )
    demoted = before - {c["expression"] for c in state.candidates}
    if demoted:
        # Librarian 逐轮写 index 时 `passed=True` 已经落盘。补写更正记录——
        # `ExperimentIndex._last_wins` 保证同表达式后写覆盖，否则被否掉的因子
        # 仍会以「已验证有效」喂给后续 session。
        record(
            index,
            [a for a in state.attempts if a.expression in demoted],
            run_id=session_run_id,
            data_window=data_window,
            campaign_id=session_campaign_id,
        )

    # ── 自动维护因子库（M5/M6 收尾 upsert）──────────────────────────────────────
    # 与 M1(run_session) 双路径登记簿配对：收尾把最终 passed 候选 upsert 进库（gate 复用
    # acceptance_reasons(gate="library")）。库根默认从 index_path 推导（测试的 tmp index 天然
    # 隔离）；市场从 profile.name/data_window 取。整块 try/except 兜底，不拖垮挖掘产出。
    if update_library:
        _library_upsert_team(
            state.candidates, seed=seed, mining_df=mining_df, ctx=ctx, profile=profile,
            data_window=data_window, eval_start=eval_start, index_path=index_path,
            library_root=library_root, top_k=top_k, horizon=horizon,
            run_id=session_run_id)

    # ── lift 基线复用 session 库池(v25 探针死点:lift 内部 build_library_pool
    # 在父进程重新逐因子求值 87 库因子,全 A 在 ⑤ 后余量上直接 OOM)────────────
    # 语义守卫:upsert(上方)在 lift 前,基线须含本 session 新升 active 的因子。
    # 仅当「upsert 后库 active 键集 == session lib_pool 键集」(0 新增,探针/收敛
    # 期常态)才传 lib_pool——此时两者仅差 eval_start 预热前缀裁剪,而 lift 评分
    # /build_panel 全在 admission 窗(⊆ 裁剪后窗)、baseline_hash 只依赖键集,
    # 逐值等价。键集有变 → 保持 None 让 lift 自建(语义正确优先,内存回退现状)。
    if lift_active_factor_dfs is None and lib_pool:
        try:
            from factorzen.discovery.factor_library import library_file_hash
            _lib_hash_now = library_file_hash(market, lib_root)
            if _lib_hash_now == _lib_hash_at_pool:
                lift_active_factor_dfs = lib_pool
                _step("lift 基线 ▸ 复用 session 库池(库文件未变,免重物化)")
            else:
                _step("lift 基线 ▸ 库文件已变(本 session upsert),重新物化")
        except Exception as exc:
            _LOG.warning("lift 基线复用检查失败,回退重物化: %s: %s",
                         type(exc).__name__, exc)

    # ── session 末自动 lift 钩子（写 manifest 前；失败不杀死挖掘 session）────
    lift_meta = _session_end_auto_lift(
        state,
        daily=daily,
        holdout_df=holdout_df,
        profile=profile,
        ctx=ctx,
        market=market,
        library_root=lib_root,
        seed=seed,
        auto_lift=auto_lift,
        lift_se_mult=lift_se_mult,
        lift_workers=lift_workers,
        data_window=data_window,
        prepped=session_prepped,
        materialize_candidate=lift_materialize_candidate,
        active_factor_dfs=lift_active_factor_dfs,
        ret_df=lift_ret_df,
        run_id=session_run_id,
        horizon=horizon,
        index=index,  # lift 拒绝写回 experiment_index（None-gating 在钩子内）
        # 成交口径须与护栏/holdout 同源，否则准入用一个口径、lift 裁决用另一个
        exec_lag=exec_lag,
        exec_price_col=exec_price_col,
    )

    # ── session 末：被准入/probation 因子引用的 ix_* 永久化 ─────────────────
    scout_block = None
    if scout_state is not None:
        from factorzen.agents.scout_support import (
            promote_admitted_exprs,
            scout_manifest_block,
        )
        from factorzen.discovery.intraday_expr import _frame_date_bounds

        admitted_exprs = [c["expression"] for c in state.candidates if c.get("expression")]
        for row in (lift_meta.get("lift_results") or []):
            if isinstance(row, dict) and row.get("passed") and row.get("expression"):
                admitted_exprs.append(str(row["expression"]))
        _fs, _fe = _frame_date_bounds(daily)
        full_start = (_dw0.get("start") or _fs or "")
        full_end = (_dw0.get("end") or _fe or "")
        # 优先 builtin 面板 manifest coverage（B-W1 同口径读盘）
        try:
            from factorzen.daily.data.intraday import _read_manifest_fields

            _bh, cov_s, cov_e = _read_manifest_fields("v1", scout_freq)
            if cov_s:
                full_start = str(cov_s).replace("-", "")[:8]
            if cov_e:
                full_end = str(cov_e).replace("-", "")[:8]
            del _bh
        except Exception:
            pass
        try:
            scout_promoted = promote_admitted_exprs(
                session_dir=None,
                admitted_exprs=admitted_exprs,
                state=scout_state,
                session=session_run_id,
                full_start=full_start,
                full_end=full_end,
                freq=scout_freq,
                base_dir=Path(scout_base_dir) if scout_base_dir is not None else None,
                leaf_map=ctx.leaf_map,
            )
            if scout_promoted:
                _step(f"Scout promote ▸ {scout_promoted}")
        except Exception as exc:
            _LOG.warning("scout promote 失败: %s: %s", type(exc).__name__, exc)
            scout_promoted = []
        scout_block = scout_manifest_block(scout_state, promoted=scout_promoted)

    return TeamResult(
        state=state,
        candidates=state.candidates,
        n_trials=ledger.n_trials,
        rounds_log=rounds_log,
        sharpe_variance=basis.sharpe_variance,
        excluded_leaves=excluded_leaves,
        n_lift_queue=lift_meta.get("n_lift_queue", 0),
        lift_group=lift_meta.get("lift_group"),
        lift_results=lift_meta.get("lift_results") or [],
        lift_admissions=lift_meta.get("lift_admissions") or {
            "added_active": 0, "added_probation": 0,
        },
        n_lift_evaluated=lift_meta.get("n_lift_evaluated", 0),
        lift_dropped_coverage=lift_meta.get("lift_dropped_coverage") or [],
        lift_error=lift_meta.get("lift_error"),
        campaign_id=campaign_id,
        prior_n_trials=prior_n_trials,
        prior_n_sessions=prior_n_sessions,
        n_trials_family=basis.n_trials,
        intraday_scout=scout_block,
    )


def _collect_lift_queue(state: AgentState) -> list[dict]:
    """从 attempts 收集 lift 队列行（expression 去重，保留首次）。"""
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    queue: list[dict] = []
    seen: set[str] = set()
    for a in state.attempts:
        if a.reject_category != REJECT_CATEGORY_LIFT_QUEUE:
            continue
        expr = a.expression
        if not expr or expr in seen:
            continue
        seen.add(expr)
        row = {
            "expression": expr,
            "ic_train": a.ic_train,
            "ir_train": a.ir_train,
            "residual_ic_train": getattr(a, "residual_ic_train", None),
            "residual_holdout_ic": getattr(a, "residual_holdout_ic", None),
            "n_holdout_days": getattr(a, "n_holdout_days", None),
            "n_residual_holdout_days": getattr(a, "n_residual_holdout_days", None),
            "reject_category": a.reject_category,
            "reject_reason": a.reject_reason,
            "hypothesis": a.hypothesis,
        }
        # sleeve 旁路审计字段（lift 层不读；缺字段容忍）
        if getattr(a, "sleeve_candidate", False):
            row["sleeve_candidate"] = True
            for _k in (
                "subset_ic_train", "subset_ic_holdout",
                "subset_n_days_train", "subset_n_days_holdout",
                "nonzero_coverage", "subset_mask_leaves",
            ):
                _v = getattr(a, _k, None)
                if _v is not None:
                    row[_k] = _v
        queue.append(row)
    return queue


def _empty_lift_meta(*, n_lift_queue: int = 0) -> dict:
    return {
        "n_lift_queue": n_lift_queue,
        "lift_group": None,
        "lift_results": [],
        "lift_admissions": {"added_active": 0, "added_probation": 0},
        "n_lift_evaluated": 0,
        "lift_dropped_coverage": [],
        "lift_error": None,
    }


def _lift_admission_str(v) -> str | None:
    """holdout 边界 → admission 窗字符串（对齐 polars Date→Utf8 的 YYYY-MM-DD）。"""
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip().replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    if len(s) >= 10 and s[4] == "-":
        return s[:10]
    return s


def _session_end_auto_lift(
    state: AgentState,
    *,
    daily,
    holdout_df,
    profile,
    ctx,
    market: str,
    library_root: str,
    seed: int,
    horizon: int,
    auto_lift: bool = True,
    lift_se_mult: float = 1.0,
    lift_workers: int | None = None,  # None→run_lift_tests 按可用内存自适应
    data_window: dict | None = None,
    prepped=None,
    materialize_candidate=None,
    active_factor_dfs: dict | None = None,
    ret_df=None,
    run_id: str | None = None,
    index=None,
    exec_lag: int = 0,
    exec_price_col: str | None = None,
) -> dict:
    """session 末：lift 队列 → 覆盖把关 → 组门 → 逐候选 → upsert。

    残差增量口径（``residual_ic_v1``）：组门与逐候选共用同一库快照投影，
    不再复用基线 IC 序列。``lift_workers`` 透传到逐候选并行
    （None=按可用内存自适应；``<=1`` 串行）。
    ``horizon``：与 ``run_team_agent`` 的 mining horizon 一致，强制显式传入
    （禁止再吃 ``DEFAULT_HORIZON`` 隐式默认，避免 single 评估与 lift 入库漂移）。

    整块 try/except：lift 失败绝不杀死已完成的挖掘 session。

    ``index``：可选 ExperimentIndex；非 None 时把 lift 拒绝（组门 / below_bar）
    追加写回 experiment_index（``reject_category=lift_rejected``）。None → 零写入（零回归）。
    """
    queue = _collect_lift_queue(state)
    if not queue:
        return _empty_lift_meta(n_lift_queue=0)
    if not auto_lift:
        return _empty_lift_meta(n_lift_queue=len(queue))

    meta = _empty_lift_meta(n_lift_queue=len(queue))
    _step(f"lift 钩子 ▸ 队列 {len(queue)} 个候选（expression 去重）")
    try:
        from factorzen.discovery.guardrails import (
            DEFAULT_HOLDOUT_MIN_DAYS,
            DEFAULT_LIFT_THRESHOLD,
        )
        from factorzen.discovery.lift_test import (
            make_lift_context,
            run_group_lift,
            run_lift_tests,
        )

        leaf_map = ctx.leaf_map if ctx is not None else None
        holdout_start = holdout_df["trade_date"].min()
        holdout_end = holdout_df["trade_date"].max()
        # admission 窗：与 polars Date→Utf8 口径对齐（YYYY-MM-DD），供评分日 IC 裁剪
        adm_start = _lift_admission_str(holdout_start)
        adm_end = _lift_admission_str(holdout_end)

        # 统一评估上下文：优先复用 session 同源 prepped；否则内部 prep 一次。
        # horizon 跟随 mining session（run_team_agent 入参），禁止硬编码 DEFAULT_HORIZON
        lift_ctx = make_lift_context(
            market, daily,
            profile=profile,
            leaf_map=leaf_map,
            horizon=horizon,
            admission_start=adm_start,
            admission_end=adm_end,
            library_root=library_root,
            prepped=prepped,
            # python 型候选/基线物化口径（三 lift 消费方同口径，改一查三）
            python_universe=(data_window or {}).get("universe"),
            python_market=market,
            # 成交口径须与护栏/holdout 一致，否则准入用一个口径、lift 裁决用另一个
            exec_lag=exec_lag,
            exec_price_col=exec_price_col,
        )
        meta["admission_start"] = adm_start
        meta["admission_end"] = adm_end
        meta["horizon"] = lift_ctx.horizon

        # 物化路径：显式注入优先；否则用 ctx.prepped 的 materializer（消除 prep 不对称）
        mat = materialize_candidate
        if mat is None:
            from factorzen.discovery.lift_test import _materializer_from_prepped
            mat = _materializer_from_prepped(
                lift_ctx.prepped, leaf_map,
                python_universe=lift_ctx.python_universe,
                python_market=lift_ctx.python_market,
            )

        # 覆盖门：W1b 旁路已前置 holdout 残差覆盖；此处双保险（物化后评分窗日数）。
        # filter_candidates_by_coverage 与 CLI lift-test 共用，语义零变化。
        from factorzen.discovery.lift_test import (
            filter_candidates_by_coverage,
            group_gate_ok,
            partition_lift_queue_by_sleeve,
        )
        kept, dropped = filter_candidates_by_coverage(
            queue,
            materialize_candidate=mat,
            holdout_start=holdout_start,
            min_days=DEFAULT_HOLDOUT_MIN_DAYS,
        )
        meta["lift_dropped_coverage"] = dropped
        if dropped:
            _step(f"lift 钩子 ▸ 覆盖剔除 {len(dropped)} 个（OOS < {DEFAULT_HOLDOUT_MIN_DAYS} 天）")
        if not kept:
            _step("lift 钩子 ▸ 覆盖后队列为空，跳过组测")
            return meta

        # sleeve 不与稠密混 residual 组门（overlay 个体；CLI 同源）
        dense_kept, sleeve_kept = partition_lift_queue_by_sleeve(kept)
        results: list = []
        n_eval = 0

        if dense_kept:
            # 组门：稠密批残差等权组合一次——失败则不跑稠密逐候选
            group = run_group_lift(
                dense_kept,
                market=market,
                daily=daily,
                leaf_map=leaf_map,
                library_root=library_root,
                seed=seed,
                threshold=DEFAULT_LIFT_THRESHOLD,
                active_factor_dfs=active_factor_dfs,
                ret_df=ret_df,
                materialize_candidate=mat,
                ctx=lift_ctx,
            )
            # 防御性剥离：组结果本无 base_daily；若旧 mock 注入帧则不进 JSON manifest
            meta["lift_group"] = {
                k: v for k, v in group.items() if k != "base_daily"
            }
            n_eval += 1  # 组门计 1 次（多重检验 N 记账）

            group_ok, bar = group_gate_ok(
                group,
                threshold=float(DEFAULT_LIFT_THRESHOLD),
                lift_se_mult=float(lift_se_mult),
            )
            g_lift = group.get("lift")
            g_se = group.get("lift_se")
            _step(
                f"lift 钩子 ▸ 组 lift={g_lift!r} se={g_se!r} bar={bar:.4f} "
                f"→ {'过' if group_ok else '拒'}（dense={len(dense_kept)}）"
            )
            if not group_ok:
                # 组门不过：仅稠密 reject；sleeve 仍走 overlay
                _append_lift_rejects_to_index(
                    index,
                    dense_kept,
                    lift=g_lift,
                    lift_se=g_se,
                    lift_reason="group_gate_fail",
                    data_window=data_window,
                    admission_start=adm_start,
                    admission_end=adm_end,
                    baseline_rank_ic=group.get("baseline"),
                    source="session_auto_lift",
                )
            else:
                dense_results = run_lift_tests(
                    dense_kept,
                    market=market,
                    daily=daily,
                    leaf_map=leaf_map,
                    library_root=library_root,
                    top_m=None,  # 全测，no silent caps
                    seed=seed,
                    threshold=DEFAULT_LIFT_THRESHOLD,
                    active_factor_dfs=active_factor_dfs,
                    ret_df=ret_df,
                    materialize_candidate=mat,
                    ctx=lift_ctx,
                    lift_workers=lift_workers,
                )
                results.extend(dense_results)
                n_eval += len(dense_results)
        else:
            meta["lift_group"] = {
                "skipped": "no_dense_after_sleeve_split",
                "n_sleeve": len(sleeve_kept),
            }

        if sleeve_kept:
            _step(
                f"lift 钩子 ▸ sleeve overlay 个体 {len(sleeve_kept)} 条"
                f"（跳过 residual 组门）"
            )
            sleeve_results = run_lift_tests(
                sleeve_kept,
                market=market,
                daily=daily,
                leaf_map=leaf_map,
                library_root=library_root,
                top_m=None,
                seed=seed,
                threshold=DEFAULT_LIFT_THRESHOLD,
                active_factor_dfs=active_factor_dfs,
                ret_df=ret_df,
                materialize_candidate=mat,
                ctx=lift_ctx,
                lift_workers=lift_workers,
            )
            results.extend(sleeve_results)
            n_eval += len(sleeve_results)

        meta["lift_results"] = results
        meta["n_lift_evaluated"] = n_eval
        if not results:
            return meta

        # 延迟导入：任务 D 契约；测试 monkeypatch factor_library.upsert_lift_admissions
        from factorzen.discovery.factor_library import upsert_lift_admissions
        from factorzen.discovery.lift_test import lift_admission

        dw = data_window or {}
        # session 自动路径一律 cap（不传 allow_active → 默认 False）：
        # 校准前 auto-lift 最多写 probation；要写 active 走 CLI --allow-active。
        adm = upsert_lift_admissions(
            results,
            market=market,
            root=library_root,
            meta={
                "eval_start": dw.get("start"),
                "eval_end": dw.get("end"),
                "universe": dw.get("universe"),
                "horizon": lift_ctx.horizon,
                "run_id": (
                    f"{run_id}_lift" if run_id is not None else f"team_lift_{seed}"
                ),
                "git_sha": get_git_sha(),
                "leaf_map": leaf_map,
            },
            threshold=DEFAULT_LIFT_THRESHOLD,
            se_mult=float(lift_se_mult),
            # W1 相关性门：复用 run_lift_tests 同一个物化器（`mat`）。
            # 不传 = 静默漏掉去重，故此处必须接通。
            materialize=mat,
        )
        meta["lift_admissions"] = {
            "added_active": int(adm.get("added_active", 0)),
            "added_probation": int(adm.get("added_probation", 0)),
        }
        _step(
            f"lift 钩子 ▸ 准入 active={meta['lift_admissions']['added_active']} "
            f"probation={meta['lift_admissions']['added_probation']}"
        )
        # 逐候选 admission reject → 写回 index（active/probation 不写）
        cand_by_expr = {c.get("expression"): c for c in kept if c.get("expression")}
        reject_rows: list[dict] = []
        for row in results:
            if lift_admission(
                row,
                threshold=float(DEFAULT_LIFT_THRESHOLD),
                se_mult=float(lift_se_mult),
            ) != "reject":
                continue
            expr = row.get("expression")
            if not expr:
                continue
            src = cand_by_expr.get(expr) or {}
            reject_rows.append({
                "expression": expr,
                "ic_train": src.get("ic_train"),
                "residual_ic_train": src.get("residual_ic_train"),
                "lift": row.get("lift"),
                "lift_se": row.get("lift_se"),
                "baseline": row.get("baseline"),
            })
        if reject_rows:
            _append_lift_rejects_to_index(
                index,
                reject_rows,
                lift=None,  # per-row
                lift_se=None,
                lift_reason="below_bar",
                data_window=data_window,
                admission_start=adm_start,
                admission_end=adm_end,
                baseline_rank_ic=None,
                source="session_auto_lift",
                per_row_lift=True,
            )
        return meta
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        meta["lift_error"] = msg
        _LOG.warning("session 末 lift 钩子失败（不影响挖掘产出）: %s", msg)
        _step(f"lift 钩子 ▸ 失败（已记 lift_error）: {msg}")
        return meta


def _append_lift_rejects_to_index(
    index,
    candidates: list[dict],
    *,
    lift,
    lift_se,
    lift_reason: str,
    data_window: dict | None,
    admission_start,
    admission_end,
    baseline_rank_ic,
    source: str,
    per_row_lift: bool = False,
) -> None:
    """把 lift 拒绝行追加到 experiment_index。``index is None`` → 零行为。"""
    if index is None or not candidates:
        return
    from factorzen.agents.experiment_index import build_lift_reject_record

    records: list[dict] = []
    for c in candidates:
        expr = c.get("expression") if isinstance(c, dict) else None
        if not expr:
            continue
        if per_row_lift:
            row_lift = c.get("lift")
            row_se = c.get("lift_se")
            row_base = c.get("baseline")
        else:
            row_lift = lift
            row_se = lift_se
            row_base = baseline_rank_ic
        records.append(build_lift_reject_record(
            expression=str(expr),
            data_window=data_window,
            lift=row_lift,
            lift_se=row_se,
            lift_reason=lift_reason,
            source=source,
            ic_train=c.get("ic_train") if isinstance(c, dict) else None,
            residual_ic_train=(
                c.get("residual_ic_train") if isinstance(c, dict) else None
            ),
            baseline_rank_ic=row_base,
            admission_start=admission_start,
            admission_end=admission_end,
        ))
    if records:
        index.append(records)


def _library_upsert_team(candidates, *, seed, mining_df, ctx, profile, data_window,
                         eval_start, index_path, library_root, top_k, horizon,
                         run_id: str | None = None) -> None:
    """M5/M6 收尾把最终 passed 候选 upsert 进因子库。全 try/except 兜底，A股零回归底线。"""
    from datetime import date

    try:
        if not candidates:
            return
        from factorzen.discovery import factor_library as _fl
        from factorzen.discovery.evaluation import _preprocess_daily
        market = getattr(profile, "name", None) or (
            (data_window or {}).get("market")) or "ashare"
        root = library_root or str(Path(index_path).parent / "factor_library")
        dw = data_window or {}
        _start = dw.get("start") or eval_start or mining_df["trade_date"].min().strftime("%Y%m%d")
        _end = dw.get("end") or mining_df["trade_date"].max().strftime("%Y%m%d")
        leaf_map = ctx.leaf_map
        prepped = _preprocess_daily(mining_df, profile).sort(["ts_code", "trade_date"])
        # 去相关用紧凑矩阵物化器（内存有界，见 factor_library.make_compact_materializer）。
        compact = _fl.make_compact_materializer(prepped, leaf_map)

        _fl.upsert(
            market, candidates, eval_window=(_start, _end), universe=dw.get("universe"),
            horizon=horizon,
            run_id=run_id if run_id is not None else f"team_{seed}",
            session_dir=None,
            git_sha=get_git_sha(), now=date.today().strftime("%Y-%m-%d"),
            compact_materialize=compact, leaf_map=leaf_map, root=root)
    except Exception as exc:  # 库写入失败不许影响挖掘产出（A股零回归底线）
        _LOG.warning("因子库 upsert 失败（不影响挖掘产出）: %s: %s", type(exc).__name__, exc)


def write_team_manifest(
    result: TeamResult, *, out_dir: str, run_id: str, params: dict, partial: bool = False
) -> Path:
    """落 team manifest。

    ``partial=True`` 表示轮末的增量快照——挖掘尚未跑完，进程若在此后崩溃，留在磁盘上的就是它。
    """
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "seed": result.state.seed,
        "n_trials": result.n_trials,
        # deflation 基准的尺度。与 n_trials 一起，才够复算出候选的 dsr_pvalue
        # （`expected_max_sharpe ∝ sqrt(sharpe_variance)`）。partial 快照写 null——
        # 那时还没有最终 basis。`deflation_two_sided` 说明 effective_trials = 2×n_trials。
        "sharpe_variance": json_safe_float(result.sharpe_variance),
        "deflation_two_sided": True,
        # campaign trial family：跨 session 累计 N（n_trials 仍=本 session）
        "campaign_id": getattr(result, "campaign_id", None),
        "prior_n_trials": getattr(result, "prior_n_trials", 0) or 0,
        "prior_n_sessions": getattr(result, "prior_n_sessions", 0) or 0,
        "n_trials_family": getattr(result, "n_trials_family", 0) or 0,
        "iterations": result.state.iteration,
        "params": params,
        "partial": partial,
        "pbo": json_safe_float(result.state.pbo),
        "roles": ["hypothesis", "coder", "evaluator", "critic", "librarian"],
        "rounds_log": result.rounds_log,
        "attempts": [a.__dict__ for a in result.state.attempts],
        "candidates": result.candidates,
        "excluded_leaves": getattr(result, "excluded_leaves", {}) or {},
        "library_pool_size": getattr(result.state, "library_pool_size", 0),
        "n_library_correlated_rejects": getattr(
            result.state, "n_library_correlated_rejects", 0),
        "n_gray_zone": getattr(result.state, "n_gray_zone", 0),
        # n_gray_zone 语义=lift 队列计数（兼容旧字段）；n_lift_queue 为显式同义
        "n_lift_queue": getattr(result, "n_lift_queue", 0),
        "lift_group": getattr(result, "lift_group", None),
        "lift_results": getattr(result, "lift_results", None) or [],
        "lift_admissions": getattr(result, "lift_admissions", None) or {
            "added_active": 0, "added_probation": 0,
        },
        "n_lift_evaluated": getattr(result, "n_lift_evaluated", 0),
        "lift_dropped_coverage": getattr(result, "lift_dropped_coverage", None) or [],
        "lift_error": getattr(result, "lift_error", None),
        "objective": getattr(result.state, "objective", None),
        "git_sha": get_git_sha(),
    }
    scout_block = getattr(result, "intraday_scout", None)
    if scout_block is not None:
        manifest["intraday_scout"] = scout_block
    path = run_dir / "manifest.json"
    dump_manifest(manifest, path)
    return path
