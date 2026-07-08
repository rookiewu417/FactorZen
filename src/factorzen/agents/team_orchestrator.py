"""多角色团队编排：Librarian→Hypothesis→Coder→Evaluator→Critic 流水线 + 否决回路。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.nodes import node_guardrails
from factorzen.agents.roles.coder import revise_expressions, write_expressions
from factorzen.agents.roles.critic import critique
from factorzen.agents.roles.hypothesis import propose_hypotheses
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.core.experiment import get_git_sha
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.scoring import DataBundle
from factorzen.llm.generation import LLMFn
from factorzen.validation.holdout import split_holdout
from factorzen.validation.multiple_testing import TrialLedger


@dataclass
class TeamResult:
    state: AgentState
    candidates: list[dict]
    n_trials: int
    rounds_log: list[dict] = field(default_factory=list)


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


def _evaluate_and_record(state, exprs, hypothesis, *, daily, bundle, mem_seen):
    """评估一批表达式（跳过 mem_seen 去重），写 AttemptRecord，返回本批新评估的结果列表。

    灵魂约束：此函数不碰 ledger，N 诚实记账由外层 node_guardrails 统一负责（每轮恰好一次）。
    """
    fresh = [e for e in exprs if _normalize(e) not in mem_seen
             and _normalize(e) not in state.seen_expressions]
    results = evaluate_expressions(fresh, daily, bundle) if fresh else []
    for r in results:
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"],
            turnover=r.get("turnover")))
        state.seen_expressions.add(r["expression"])
    return results


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
) -> TeamResult:
    """跨轮 feedback 流水线：每轮 Librarian→Hypothesis/Coder→Evaluator→Critic→Librarian。

    N 诚实：node_guardrails 每轮恰好调用一次（记本轮 N），不在同轮内多次调用（避免三角和）。
    holdout 隔离：mining_df 供角色/记忆，holdout_df 只进 node_guardrails。
    跨轮 feedback：Critic verdict=revise_expr → 下轮 Coder.revise；revise_hypothesis → 下轮 Hypothesis（带 feedback）。
    """
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)
    ledger = TrialLedger()
    state = AgentState(seed=seed)
    index = ExperimentIndex(index_path)
    rounds_log: list[dict] = []
    # 上一轮 Critic 反馈：{"kind", "hypothesis", "exprs", "reason"}
    pending: dict | None = None
    no_improve = 0
    last_cand_count = 0

    for round_i in range(n_rounds):
        # 自适应早停：连续 patience 轮无新 passed 候选则停（patience=None → 跑满，零回归）
        if patience is not None and round_i > 0:
            no_improve = 0 if len(state.candidates) > last_cand_count else no_improve + 1
            if no_improve >= patience:
                break
        last_cand_count = len(state.candidates)
        rec = recall(index, k=5)                                   # ① Librarian

        # ②/③ Hypothesis + Coder（依据上一轮 Critic 反馈，跨轮）
        if pending and pending["kind"] == "revise_expr":
            hypothesis = pending["hypothesis"]
            exprs = revise_expressions(hypothesis, pending["exprs"], pending["reason"], llm_fn)
        else:
            fb = pending["reason"] if pending and pending["kind"] == "revise_hypothesis" else ""
            hyps = propose_hypotheses(
                llm_fn, known_invalid=rec.known_invalid, known_valid=rec.known_valid,
                feedback=fb, n=1,
            )
            if not hyps:
                state.iteration += 1
                pending = None
                continue
            hypothesis = hyps[0]
            exprs = write_expressions(hypothesis, llm_fn, avoid=rec.known_invalid)

        # ④ Evaluator：评估（跨 session + session 内去重）
        # _evaluate_and_record 不碰 ledger；node_guardrails 本轮恰好一次（N 诚实）
        results = _evaluate_and_record(
            state, exprs, hypothesis,
            daily=mining_df, bundle=bundle, mem_seen=rec.seen,
        )
        n_before = len(state.candidates)                       # Important 1: 护栏前快照
        node_guardrails(
            state, daily=mining_df, holdout_df=holdout_df,
            bundle=bundle, ledger=ledger, top_k=top_k,
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

        # Important 2: 回填 critic_verdict 到本轮代表 attempt
        round_expr = cand.get("expression", "")
        for a in state.attempts:
            if a.iteration == state.iteration and a.expression == round_expr:
                a.critic_verdict = verdict.verdict
                break

        rounds_log.append({
            "round": state.iteration,
            "hypothesis": hypothesis,
            "expressions": [r["expression"] for r in results],
            "verdict": verdict.verdict,
            "reason": verdict.reason,
        })

        # verdict → 下一轮 feedback（跨轮；不在本轮重跑护栏，避免 N 三角和）
        if verdict.verdict == "drop":
            del state.candidates[n_before:]                    # Important 1: 移除本轮新增候选
            # Bug fix（否决回路名存实亡）：node_guardrails 把通过定量护栏的 AttemptRecord
            # .passed_guardrails 置 True 后，全仓库没有其它地方重置回 False。候选被 Critic
            # drop 时必须同步重置，否则 Librarian 落盘会写出 passed=True + verdict=drop 的
            # 自相矛盾记录，被 ExperimentIndex.known_valid() 当作"已验证有效"喂给后续轮次/
            # session 的假设生成，否决回路被绕过。
            # 按 new_cands（本轮新增候选快照）整体重置，而非只重置 Critic 直接点评的代表候选
            # cand——同一轮内若有多个候选因 drop 被一并删除，状态也要一并清理，不留连坐残留。
            dropped_exprs = {c["expression"] for c in new_cands}
            for a in state.attempts:
                if a.iteration == state.iteration and a.expression in dropped_exprs:
                    a.passed_guardrails = False
            new_cands = []          # 不再回填 holdout_ic 等"已验证"字段（Librarian 写入用）
            pending = None
        elif verdict.verdict == "revise_expr":
            pending = {
                "kind": "revise_expr",
                "hypothesis": hypothesis,
                "exprs": exprs,
                "reason": verdict.reason,
            }
        elif verdict.verdict == "revise_hypothesis":
            pending = {"kind": "revise_hypothesis", "reason": verdict.reason}
        else:
            pending = None

        # ⑥ Librarian：本轮 attempts 写 experiment_index（含 holdout_ic 回填）
        round_attempts = [a for a in state.attempts if a.iteration == state.iteration]
        record(
            index,
            round_attempts,
            run_id=f"team_{seed}",
            candidates=new_cands,
        )
        state.iteration += 1

    return TeamResult(
        state=state,
        candidates=state.candidates,
        n_trials=ledger.n_trials,
        rounds_log=rounds_log,
    )


def write_team_manifest(
    result: TeamResult, *, out_dir: str, run_id: str, params: dict
) -> Path:
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "seed": result.state.seed,
        "n_trials": result.n_trials,
        "iterations": result.state.iteration,
        "params": params,
        "roles": ["hypothesis", "coder", "evaluator", "critic", "librarian"],
        "rounds_log": result.rounds_log,
        "attempts": [a.__dict__ for a in result.state.attempts],
        "candidates": result.candidates,
        "git_sha": get_git_sha(),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
