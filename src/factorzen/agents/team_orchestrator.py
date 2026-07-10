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
from factorzen.agents.nodes import node_finalize_guardrails, node_guardrails
from factorzen.agents.roles.coder import (
    decompose_tasks,
    revise_expressions,
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


@dataclass
class TeamResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int
    rounds_log: list[dict] = field(default_factory=list)
    # deflation 基准的尺度；缺了它光凭 n_trials 复算不出 dsr_pvalue。
    # 默认 nan：中途的 on_round_end 检查点尚无最终 basis，写 null 比写假值诚实。
    sharpe_variance: float = float("nan")


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
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
                         eval_start=None, eval_end=None):
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
        norm = _normalize(e)
        if norm in mem_seen or norm in state.seen_expressions or norm in batch_seen:
            continue
        batch_seen.add(norm)
        fresh.append(e)
    results = (
        evaluate_expressions(fresh, daily, bundle, eval_start=eval_start, eval_end=eval_end)
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
    eval_start=None,
) -> dict | None:
    """跑一轮 Librarian→Hypothesis/Coder→Evaluator→Critic→Librarian。

    `state` / `ledger` / `rounds_log` / `index` 为可变对象，就地更新；返回下一轮的 Critic 反馈。
    抽成独立函数是为了让主循环能整轮 `try/except LLMClientError` 而不必把 120 行内联进 try 块。

    ``eval_start``：会话级 train 段起点（date，或 None，由 `run_team_agent` 解析一次后逐轮传入）。
    **None-gating**：为 None 时 train 段求值走裸 `evaluate_expressions(exprs, mining_df, bundle)`
    （旧调用者零回归）；非 None 时改在 ``warmup_daily``（完整帧）上求值、裁剪到
    ``[eval_start, mining_df 终点]``——`mining_df` 此时已被 `_prepare_segments` 提前裁到
    `eval_start`，不能直接把 `mining_df["trade_date"].min()` 当判据（`eval_start=None`
    时它就是帧起点，会让预热门把可用预热样本数误判成 0，见 task-1.4 CORRECTION）。
    """
    rec = recall(index, k=5, data_window=data_window)          # ① Librarian（按窗口分族）
    tasks: list[dict] = []

    # ②/③ Hypothesis +（任务分解）+ Coder（依据上一轮 Critic 反馈，跨轮）
    if pending and pending["kind"] == "revise_expr":
        hypothesis = pending["hypothesis"]
        exprs = revise_expressions(hypothesis, pending["exprs"], pending["reason"], llm_fn)
    else:
        fb = pending["reason"] if pending and pending["kind"] == "revise_hypothesis" else ""
        if structured:
            # RD-Agent 步1 结构化假设：direction/mechanism/expected_sign/falsification
            shyps = propose_structured(
                llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
                feedback=fb, n=1,
            )
            hyps = [format_structured(h) for h in shyps]
        else:
            hyps = propose_hypotheses(
                llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
                feedback=fb, n=1,
            )
        if not hyps:
            state.iteration += 1
            return None
        hypothesis = hyps[0]
        if structured:
            # RD-Agent 步2 任务分解：假设 → 因子任务清单，逐任务独立翻译。
            # 拆两步是为了让每次 LLM 调用只专注一件事（合并则假设过细或规格过粗）。
            tasks = decompose_tasks(hypothesis, llm_fn)
        if tasks:
            exprs = []
            for t in tasks:
                exprs.extend(
                    write_expressions(_task_text(t), llm_fn, avoid=rec.known_invalid)
                )
        else:
            # 未启用分解、或 LLM 分解失败（空 tasks）→ 降级为整条假设直译，不空转
            exprs = write_expressions(hypothesis, llm_fn, avoid=rec.known_invalid)
    if heal_rounds > 0:
        from factorzen.agents.self_heal import heal_expressions
        exprs = heal_expressions(exprs, hypothesis, llm_fn,
                                 max_rounds=heal_rounds, health_check=health)

    # ④ Evaluator：评估（跨 session + session 内去重）
    # _evaluate_and_record 不碰 ledger；node_guardrails 本轮恰好一次（N 诚实）
    #
    # None-gating（非 None 才切到 warmup_daily + 段边界，None 时裸调用 mining_df，
    # 零回归）：gate 在会话级 eval_start 本身，不能用 mining_df.min() 判断——
    # eval_start=None 时 mining_df 就是帧起点，误用会让预热门把整段判成 0 可用预热。
    if eval_start is not None:
        ev_daily, ev_end = warmup_daily, mining_df["trade_date"].max()
    else:
        ev_daily, ev_end = mining_df, None
    results = _evaluate_and_record(
        state, exprs, hypothesis,
        daily=ev_daily, bundle=bundle, mem_seen=rec.seen,
        eval_start=eval_start, eval_end=ev_end,
    )
    n_before = len(state.candidates)                       # Important 1: 护栏前快照
    node_guardrails(
        state, daily=mining_df, holdout_df=holdout_df,
        bundle=bundle, ledger=ledger, top_k=top_k,
        warmup_daily=warmup_daily,   # holdout 扩窗预热用完整帧
        eval_start=eval_start,       # 池级 PBO 的 None-gating：None 时裸求值，零回归
    )
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
        "hypothesis": hypothesis,
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
        next_pending = {
            "kind": "revise_expr",
            "hypothesis": hypothesis,
            "exprs": exprs,
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
) -> TeamResult:
    """跨轮 feedback 流水线：每轮 Librarian→Hypothesis/Coder→Evaluator→Critic→Librarian。

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
    mining_df, holdout_df, _ = _prepare_segments(
        daily, eval_start=eval_start, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)
    _eval_start_date = _to_date(eval_start) if eval_start is not None else None
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
    health = make_health_check(mining_df) if heal_rounds > 0 else None
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
                break
        last_cand_count = len(state.candidates)
        try:
            pending = _run_one_round(
                state, llm_fn, index=index, ledger=ledger, rounds_log=rounds_log,
                mining_df=mining_df, holdout_df=holdout_df, bundle=bundle,
                pending=pending, seed=seed, top_k=top_k,
                heal_rounds=heal_rounds, structured=structured, health=health,
                data_window=data_window, warmup_daily=daily,
                eval_start=_eval_start_date,
            )
        except LLMClientError as exc:
            llm_failures += 1
            state.iteration += 1   # 角色流水线未跑完，此处补推进以保持轮次语义一致
            pending = None
            _LOG.warning("第 %d 轮 LLM 不可用（连续第 %d 次），跳过本轮: %s",
                         round_i, llm_failures, exc)
            if llm_failures >= llm_failure_patience:
                _LOG.error("连续 %d 轮 LLM 不可用，提前终止挖掘（已产出 %d 个候选）",
                           llm_failures, len(state.candidates))
                break
            continue
        llm_failures = 0
        if on_round_end is not None:
            on_round_end(TeamResult(state=state, candidates=state.candidates,
                                    n_trials=ledger.n_trials, rounds_log=rounds_log))

    # 收尾复核：早轮候选此前按「截至当轮」的 N 定 p，门槛偏松。用最终 basis 统一重判。
    before = {c["expression"] for c in state.candidates}
    basis = node_finalize_guardrails(state, daily=mining_df, bundle=bundle)
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

    return TeamResult(
        state=state,
        candidates=state.candidates,
        n_trials=ledger.n_trials,
        rounds_log=rounds_log,
        sharpe_variance=basis.sharpe_variance,
    )


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
        "git_sha": get_git_sha(),
    }
    path = run_dir / "manifest.json"
    dump_manifest(manifest, path)
    return path
