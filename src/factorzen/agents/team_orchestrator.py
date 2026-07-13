"""多角色团队编排：Librarian→Hypothesis→Coder→Evaluator→Critic 流水线 + 否决回路。"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from factorzen.agents.evaluation import evaluate_expressions, make_health_check
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
from factorzen.agents.roles.critic import critique
from factorzen.agents.roles.hypothesis import (
    format_structured,
    propose_hypotheses,
    propose_structured,
)
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.core.experiment import get_git_sha
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.client import LLMClientError
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import split_holdout
from factorzen.validation.multiple_testing import TrialLedger

_LOG = logging.getLogger(__name__)


def _step(msg: str) -> None:
    """过程提示 → stdout。挖掘由 CLI 触发，用户要看实时进度；不走 logging 免被默认级别吞掉。"""
    print(f"[mine-team] {msg}", flush=True)


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


def _evaluate_and_record(state, exprs, hypothesis, *, daily, bundle, mem_seen,
                         eval_start=None, eval_end=None, profile=None, leaf_map=None):
    """评估一批表达式（跳过 mem_seen 去重），写 AttemptRecord，返回本批新评估的结果列表。

    灵魂约束：此函数不碰 ledger，N 诚实记账由外层 node_guardrails 统一负责（每轮恰好一次）。

    ``eval_start``/``eval_end``：会话级 train 段边界（date，或 None）。**None-gating**：
    为 None 时原样转发 None 给 `evaluate_expressions`（等价裸调用，零回归）；非 None 时
    调用方须传 ``daily`` 为含预热前缀的完整帧——裁剪与预热门在 `evaluate_expressions`
    内部完成。调用方负责按会话级 `eval_start` 是否为 None 选择正确的 `daily`
    （mining_df 还是 warmup_daily），本函数只透传，不做二次判断。
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
                             profile=profile)
        if fresh else []
    )
    for r in results:
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"],
            turnover=r.get("turnover"), n_train=r.get("n_train")))
        state.seen_expressions.add(r["expression"])
    return results


def _run_one_round(
    state, llm_fn, *, index, ledger, rounds_log, mining_df, holdout_df, bundle,
    pending, seed, top_k, heal_rounds, structured, health, data_window, warmup_daily,
    eval_start=None, leaf_budgets=None, hypotheses_per_round=1, profile=None, ctx=None,
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
    _step("  ① Librarian 检索历史经验（known valid/invalid）")
    rec = recall(index, k=5, data_window=data_window)          # ① Librarian（按窗口分族）
    tasks: list[dict] = []

    # ②/③ Hypothesis +（任务分解）+ Coder（依据上一轮 Critic 反馈，跨轮）
    # 产出 hyp_batches: [(假设, 表达式集)]。revise 分支单假设；fresh 分支可多假设（task D，
    # hypotheses_per_round>1 时逐假设独立走分解→翻译，attempts 累积，护栏/Critic 仍每轮一次）。
    if pending and pending["kind"] == "revise_expr":
        _step("  ②→③ Coder 依 Critic 反馈修订表达式")
        hypothesis = pending["hypothesis"]
        hyps = [hypothesis]
        hyp_batches = [(hypothesis, revise_expressions(
            hypothesis, pending["exprs"], pending["reason"], llm_fn,
            leaf_budgets=leaf_budgets, market=ctx.market, leaf_names=ctx.leaf_names))]
    else:
        _step("  ② Hypothesis 提假设"
              + (f"（×{hypotheses_per_round}）" if hypotheses_per_round > 1 else "")
              + ("（结构化：机制/预期符号/证伪）" if structured else ""))
        fb = pending["reason"] if pending and pending["kind"] == "revise_hypothesis" else ""
        if structured:
            # RD-Agent 步1 结构化假设：direction/mechanism/expected_sign/falsification
            shyps = propose_structured(
                llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
                feedback=fb, n=hypotheses_per_round, market=ctx.market,
            )
            hyps = [format_structured(h) for h in shyps]
        else:
            hyps = propose_hypotheses(
                llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
                feedback=fb, n=hypotheses_per_round, market=ctx.market,
            )
        if not hyps:
            _step("  · Hypothesis 未产出假设，跳过本轮")
            state.iteration += 1
            return None
        # 逐假设独立走「任务分解 → Coder 翻译」（步2 任务分解拆两步：每次 LLM 调用只专注
        # 一件事，合并则假设过细或规格过粗）。所有假设的 tasks 汇入 rounds_log 供溯源。
        hyp_batches = []
        for h in hyps:
            h_tasks = decompose_tasks(h, llm_fn) if structured else []
            tasks.extend(h_tasks)
            if h_tasks:
                h_exprs: list[str] = []
                for t in h_tasks:
                    h_exprs.extend(write_expressions(
                        _task_text(t), llm_fn, avoid=rec.known_invalid,
                        leaf_budgets=leaf_budgets, market=ctx.market, leaf_names=ctx.leaf_names))
            else:
                # 未启用分解、或 LLM 分解失败（空 tasks）→ 降级为整条假设直译，不空转
                h_exprs = write_expressions(h, llm_fn, avoid=rec.known_invalid,
                                            leaf_budgets=leaf_budgets, market=ctx.market,
                                            leaf_names=ctx.leaf_names)
            hyp_batches.append((h, h_exprs))
        _step(f"  ③ Coder 翻译表达式（{len(hyps)} 假设"
              + (f" / {len(tasks)} 子任务" if tasks else "") + "）")
    if heal_rounds > 0:
        from factorzen.agents.self_heal import heal_expressions
        hyp_batches = [
            (h, heal_expressions(ex, h, llm_fn, max_rounds=heal_rounds, health_check=health,
                                 leaf_map=ctx.leaf_map, market=ctx.market,
                                 leaf_names=ctx.leaf_names))
            for h, ex in hyp_batches
        ]

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
    warm_budget = 6   # 每轮预热回灌上限 6 条（跨假设共享），控 LLM 成本
    for h, h_exprs in hyp_batches:
        _step(f"  ④ Evaluator 评估 {len(h_exprs)} 个表达式")
        h_results = _evaluate_and_record(
            state, h_exprs, h, daily=ev_daily, bundle=bundle, mem_seen=rec.seen,
            eval_start=eval_start, eval_end=ev_end, profile=profile, leaf_map=ctx.leaf_map,
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
            refed: list[str] = []
            for r in warm_errs:
                refed.extend(revise_from_error(h, r["expression"], r["error"], llm_fn,
                                               leaf_budgets=leaf_budgets, market=ctx.market,
                                               leaf_names=ctx.leaf_names))
            if refed:
                _step(f"  ④+ 预热错误回灌 revise（{len(warm_errs)} 条 → {len(refed)} 修正）")
                results += _evaluate_and_record(
                    state, refed, h, daily=ev_daily, bundle=bundle, mem_seen=rec.seen,
                    eval_start=eval_start, eval_end=ev_end, profile=profile, leaf_map=ctx.leaf_map,
                )
    # 代表假设/表达式：供 Critic stub 与 revise pending（多假设时取最后一个批次，同现状语义）
    hypothesis = hyps[-1] if hyps else ""
    exprs = hyp_batches[-1][1] if hyp_batches else []
    n_before = len(state.candidates)                       # Important 1: 护栏前快照
    _step("  ⑤ 防过拟合护栏（DSR / holdout / CI / 去相关）")
    node_guardrails(
        state, daily=mining_df, holdout_df=holdout_df,
        bundle=bundle, ledger=ledger, top_k=top_k,
        warmup_daily=warmup_daily,   # holdout 扩窗预热用完整帧
        eval_start=eval_start,       # 池级 PBO 的 None-gating：None 时裸求值，零回归
        profile=profile,             # crypto 派生列 + 叶子映射；None 零回归
    )
    _print_rejections("mine-team", state)
    new_cands = state.candidates[n_before:]                # Important 1/Minor 2: 本轮新增候选

    # ⑤ Critic：看本轮候选（guardrails 已跑，含 dsr/holdout）
    # Minor 2: 取本轮新增候选；无则构造 stub（不误杀，不取旧候选）
    cand = new_cands[-1] if new_cands else {
        "expression": results[-1]["expression"] if results else (exprs[0] if exprs else ""),
        "hypothesis": hypothesis,
        "ic_train": results[-1]["ic_train"] if results else None,
        "ir_train": results[-1]["ir_train"] if results else None,
        "turnover": results[-1].get("turnover") if results else None,
    }
    _step("  ⑥ Critic 裁决")
    verdict = critique(cand, llm_fn)

    # 回填 critic_verdict 到本轮**全部**新增候选对应的 attempt，而不只是 Critic 直接点评的
    # 代表候选：裁决针对的是本轮的假设方向，而 new_cands 同源于一个 hypothesis。
    # 这也是 known_valid() 判定所依赖的字段——只回填代表的话，drop 时其余候选 verdict=None，
    # 会漏进「已验证有效」。（早先靠重置 passed_guardrails 实现连坐，见 drop 分支的说明。）
    round_expr = cand.get("expression", "")
    round_exprs = {c["expression"] for c in new_cands} or {round_expr}
    for a in state.attempts:
        if a.iteration == state.iteration and a.expression in round_exprs:
            a.critic_verdict = verdict.verdict

    rounds_log.append({
        "round": state.iteration,
        # 多假设时记全部（"；" 连接）；单假设时即该假设字符串（零回归）。
        # rounds_log.hypothesis 只进 manifest 溯源，无下游按单字符串解析（per-attempt 归属
        # 由 AttemptRecord.hypothesis 各自承载），故连接安全。
        "hypothesis": "；".join(hyps),
        "tasks": tasks,                       # 步2 产物，实验溯源用（非 structured 轮为 []）
        "expressions": [r["expression"] for r in results],
        "verdict": verdict.verdict,
        "reason": verdict.reason,
    })

    # verdict → 下一轮 feedback（跨轮；不在本轮重跑护栏，避免 N 三角和）
    if verdict.verdict == "drop":
        del state.candidates[n_before:]                    # Important 1: 移除本轮新增候选
        # 否决回路（原 commit 1e0bda4）：drop 的候选不得被 known_valid() 当作「已验证有效」
        # 喂给后续轮次/session 的假设生成。
        #
        # 早先的实现是把 AttemptRecord.passed_guardrails 重置为 False——那是**用事实字段
        # 编码复用决策**：该因子确实过了全部定量护栏，标成 passed=False 会让它落进
        # known_invalid 被当作「已验证无效」，同样是污染，只是方向相反。
        # 现在 passed_guardrails 是不可变的事实，否决由 known_valid() 读 verdict 完成
        # （见 ExperimentIndex._VETOED_VERDICTS）。语义不变，契约自洽。
        new_cands = []          # 不再回填 holdout_ic 等"已验证"字段（Librarian 写入用）
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
        run_id=f"team_{seed}",
        candidates=new_cands,
        data_window=data_window,
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
    """
    mining_df, holdout_df, holdout_start = _prepare_segments(
        daily, eval_start=eval_start, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)
    _step(f"数据切分 ▸ 训练 {mining_df['trade_date'].n_unique()} 天 / "
          f"holdout {holdout_df['trade_date'].n_unique()} 天")
    # 市场上下文（profile=None → A 股默认）：叶子集/映射/市场名，供 budgets 与逐轮透传。
    ctx = AgentContext.from_profile(profile)
    # 开局摘死叶：必须在与求值同一套 prep 帧上量覆盖（close→close_adj 别名 + 派生列），
    # 否则 ret_1d/vwap 等会被误判为「列不存在→覆盖 0」整批摘除。
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
        from factorzen.agents.evaluation import _preprocess_daily
        from factorzen.discovery.expression import leaf_warmup_budgets
        from factorzen.pipelines.factor_mine import AGENT_WARMUP_LOOKBACK
        _all_budgets = leaf_warmup_budgets(
            _preprocess_daily(daily, profile), _eval_start_date, ctx.leaf_names,
            leaf_map=ctx.leaf_map)
        leaf_budgets = {k: v for k, v in _all_budgets.items() if v < AGENT_WARMUP_LOOKBACK}
    # ── 记录在案的假设：多重检验的 N **不跨 session 累积** ──────────────────────
    # ledger 每 session 从 0 起。而 Librarian 把历史已试表达式喂给 LLM 让它避开，于是后续
    # session 在同一搜索空间的剩余部分继续搜索——累计试了 120 次，DSR 却按本 session 的 N 判。
    #
    # 这是一个**假设**，不是已验证的结论：「跨 session 的 N 应该是多少」是建模立场
    # （对比 P0——M1 真实的 dsr_pvalue 可反解校验，那才是事实）。当前它 latent：
    # run_team_mine 只有 `fz mine team` 一个调用者，ops daily / research run 都不跑 team 挖掘。
    #
    # 若将来出现无人值守的 team 挖掘循环（多个 session 堆在同一 index_path 上），须改为：
    # 从 index 重建**同一 data_window** 的历史 IR 池，与本 session 池合并后交给
    # DeflationBasis.from_ir_pool —— N 与 sharpe_variance 天然同源（R8）。
    # 前提字段（ir_train / n_train / data_window）已由 Librarian 落盘。
    #
    # 另有一条更根本、且 N 累积管不到的问题：**holdout 跨 session 复用**。每个 session 都拿
    # 同一段 holdout 验收候选，跑 10 个 session 它就被看了 10 遍，不再是 OOS 而是第二个训练集。
    # 那是 OOS 污染，修法是预算/轮换而非累积 N。单列待评估。
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    index = ExperimentIndex(index_path)
    # 求值层诊断器只建一次（预处理较重）；heal_rounds=0 时不建，零开销
    health = make_health_check(mining_df, profile=profile, leaf_map=ctx.leaf_map) \
        if heal_rounds > 0 else None
    rounds_log: list[dict] = []
    # 上一轮 Critic 反馈：{"kind", "hypothesis", "exprs", "reason"}
    pending: dict | None = None
    no_improve = 0
    last_cand_count = 0
    llm_failures = 0

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
            pending = _run_one_round(
                state, llm_fn, index=index, ledger=ledger, rounds_log=rounds_log,
                mining_df=mining_df, holdout_df=holdout_df, bundle=bundle,
                pending=pending, seed=seed, top_k=top_k,
                heal_rounds=heal_rounds, structured=structured, health=health,
                data_window=data_window, warmup_daily=daily,
                eval_start=_eval_start_date, leaf_budgets=leaf_budgets,
                hypotheses_per_round=hypotheses_per_round, profile=profile, ctx=ctx,
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
    _step("收尾复核：以最终 N 统一重判候选 DSR")
    before = {c["expression"] for c in state.candidates}
    basis = node_finalize_guardrails(state, daily=mining_df, bundle=bundle, profile=profile)
    demoted = before - {c["expression"] for c in state.candidates}
    if demoted:
        # Librarian 逐轮写 index 时 `passed=True` 已经落盘。补写更正记录——
        # `ExperimentIndex._last_wins` 保证同表达式后写覆盖，否则被否掉的因子
        # 仍会以「已验证有效」喂给后续 session。
        record(
            index,
            [a for a in state.attempts if a.expression in demoted],
            run_id=f"team_{seed}",
            data_window=data_window,
        )

    # ── 自动维护因子库（M5/M6 收尾 upsert）──────────────────────────────────────
    # 与 M1(run_session) 双路径登记簿配对：收尾把最终 passed 候选 upsert 进库（gate 复用
    # acceptance_reasons(gate="library")）。库根默认从 index_path 推导（测试的 tmp index 天然
    # 隔离）；市场从 profile.name/data_window 取。整块 try/except 兜底，不拖垮挖掘产出。
    if update_library:
        _library_upsert_team(
            state.candidates, seed=seed, mining_df=mining_df, ctx=ctx, profile=profile,
            data_window=data_window, eval_start=eval_start, index_path=index_path,
            library_root=library_root, top_k=top_k, horizon=horizon)

    return TeamResult(
        state=state,
        candidates=state.candidates,
        n_trials=ledger.n_trials,
        rounds_log=rounds_log,
        sharpe_variance=basis.sharpe_variance,
        excluded_leaves=excluded_leaves,
    )


def _library_upsert_team(candidates, *, seed, mining_df, ctx, profile, data_window,
                         eval_start, index_path, library_root, top_k, horizon) -> None:
    """M5/M6 收尾把最终 passed 候选 upsert 进因子库。全 try/except 兜底，A股零回归底线。"""
    from datetime import date

    try:
        if not candidates:
            return
        from factorzen.agents.evaluation import _preprocess_daily
        from factorzen.discovery import factor_library as _fl
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
            horizon=horizon, run_id=f"team_{seed}", session_dir=None,
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
        "iterations": result.state.iteration,
        "params": params,
        "partial": partial,
        "pbo": json_safe_float(result.state.pbo),
        "roles": ["hypothesis", "coder", "evaluator", "critic", "librarian"],
        "rounds_log": result.rounds_log,
        "attempts": [a.__dict__ for a in result.state.attempts],
        "candidates": result.candidates,
        "excluded_leaves": getattr(result, "excluded_leaves", {}) or {},
        "git_sha": get_git_sha(),
    }
    path = run_dir / "manifest.json"
    dump_manifest(manifest, path)
    return path
