"""Agent 闭环的函数式节点：node(State) -> State。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.agents.memory import negative_recall
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import (
    LLMFn,
    build_agent_messages,
    generate_factor_proposal,
    semantic_check,
)


@dataclass
class _PendingExpr:
    hypothesis: str
    expression: str


@dataclass
class AgentContext:
    op_names: list[str] = field(default_factory=lambda: list(OPERATORS.keys()))
    leaf_names: list[str] = field(default_factory=lambda: list(LEAF_FEATURES.keys()))


def node_generate(state: AgentState, llm_fn: LLMFn, *, daily, bundle,
                  n_hypotheses: int = 1, feedback: str = "") -> AgentState:
    """生成假设+表达式 → 语义对齐自检 → 暂存待评估（compile/eval 在 node_evaluate）。"""
    ctx = AgentContext()
    msgs = build_agent_messages(ctx.op_names, ctx.leaf_names, feedback, state.negative_examples)
    proposals = generate_factor_proposal(msgs, llm_fn, n_hypotheses=n_hypotheses)
    pending: list[_PendingExpr] = []
    for p in proposals:
        for expr in p.expressions:
            try:
                norm = to_expr_string(parse_expr(expr))
            except ValueError:
                norm = expr
            if norm in state.seen_expressions:
                continue
            ok, _reason = semantic_check(p.hypothesis, expr, llm_fn)
            if ok:
                pending.append(_PendingExpr(p.hypothesis, norm))
    state.__dict__.setdefault("_pending", [])
    state._pending = pending  # type: ignore[attr-defined]
    return state


def node_evaluate(state: AgentState, *, daily, bundle) -> AgentState:
    """对暂存表达式批量评估，写 AttemptRecord + 更新 seen。"""
    pending = getattr(state, "_pending", [])
    exprs = [p.expression for p in pending]
    results = evaluate_expressions(exprs, daily, bundle) if exprs else []
    for p, r in zip(pending, results, strict=True):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis=p.hypothesis, expression=r["expression"],
            compile_ok=r["compile_ok"], ic_train=r["ic_train"], passed_guardrails=False,
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"],
            turnover=r.get("turnover")))
        state.seen_expressions.add(r["expression"])
    state._pending = []  # type: ignore[attr-defined]
    return state


def node_guardrails(
    state: AgentState,
    *,
    daily,
    holdout_df,
    bundle,
    ledger,
    top_k: int = 5,
    dsr_alpha: float = 0.05,
) -> AgentState:
    """对过编译的候选记账 N、跑 holdout_ic/DSR，过关者进 candidates。

    passed 判定委托 discovery.guardrails.guardrail_passed（DSR p 值口径），与 M1 统一，
    消除双路径漂移（旧 dsr>0.5 松约 10 倍，收紧到 pval<dsr_alpha）。池级 PBO 记入 state.pbo。
    """
    from datetime import datetime as _dt

    import polars as pl

    from factorzen.agents.evaluation import _node_to_factor_df
    from factorzen.discovery.guardrails import guardrail_passed, pool_pbo
    from factorzen.discovery.scoring import max_correlation
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    from factorzen.validation.holdout import holdout_ic

    passed = [a for a in state.attempts
              if a.iteration == state.iteration and a.compile_ok and a.ic_train is not None]
    ledger.record(len(passed))
    passed.sort(key=lambda a: abs(a.ic_train or 0.0), reverse=True)

    existing_exprs: set[str] = {c["expression"] for c in state.candidates}

    pool: dict = {}
    for i, c in enumerate(state.candidates):
        try:
            pool[f"prev_{i}"] = _node_to_factor_df(parse_expr(c["expression"]), holdout_df)
        except Exception:
            continue

    cut = _dt.strptime(bundle.train_end, "%Y%m%d").date()
    n_obs = max(daily.filter(pl.col("trade_date") <= cut)["trade_date"].n_unique(), 20)

    for a in passed[:top_k]:
        if a.expression in existing_exprs:
            continue
        try:
            node = parse_expr(a.expression)
            fdf_hold = _node_to_factor_df(node, holdout_df)
            ic_h, ir_h, (ci_lo, ci_hi) = holdout_ic(fdf_hold, holdout_df)
            sharpe = abs(a.ir_train) if a.ir_train is not None else abs(a.ic_train or 0.0)
            dsr, pval = deflated_sharpe(sharpe, ledger.n_trials, n_obs=n_obs)
            ic_tr = a.ic_train or 0.0
            if guardrail_passed(
                ic_train=ic_tr, holdout_ic=ic_h, dsr_pvalue=pval,
                ci_low=ci_lo, ci_high=ci_hi, dsr_alpha=dsr_alpha,
            ):
                corr = max_correlation(fdf_hold, pool)
                if corr > 0.7:
                    continue
                a.passed_guardrails = True
                pool[a.expression] = fdf_hold
                existing_exprs.add(a.expression)
                state.candidates.append({
                    "expression": a.expression,
                    "hypothesis": a.hypothesis,
                    "ic_train": a.ic_train,
                    "ir_train": a.ir_train,
                    "turnover": a.turnover,
                    "holdout_ic": ic_h,
                    "holdout_ir": ir_h,
                    "dsr": dsr,
                    "dsr_pvalue": pval,
                })
        except Exception:
            continue

    try:
        cand_fdfs = [
            _node_to_factor_df(parse_expr(c["expression"]), daily) for c in state.candidates
        ]
        state.pbo = pool_pbo(cand_fdfs, bundle.fwd_returns)
    except Exception:
        state.pbo = float("nan")
    return state


def node_critic(state: AgentState, llm_fn: LLMFn) -> AgentState:
    """LLM 以风控审计员身份批判每个候选：keep/drop/mutate。"""
    for a in state.attempts:
        if a.critic_verdict is not None:
            continue
        msgs = [
            {"role": "system", "content": (
                "你是风控审计员，判断因子是否过拟合/经济直觉是否成立。"
                "注意：换手率高意味着交易成本侵蚀，train_IC 高但换手率高的因子未必可实现"
                "超额收益（成本双杀）；结合 ICIR（信息比率，越高越稳定）综合判断。"
                '只输出 JSON: {"verdict":"keep"|"drop"|"mutate","reason":"..."}')},
            {"role": "user", "content": (
                f"假设:{a.hypothesis} 表达式:{a.expression} "
                f"train_IC:{a.ic_train} ICIR:{a.ir_train} "
                f"换手率(单边,成本代理):{a.turnover} 过护栏:{a.passed_guardrails}")},
        ]
        try:
            obj = json.loads(llm_fn(msgs))
            a.critic_verdict = str(obj.get("verdict", "keep"))
        except Exception:
            a.critic_verdict = "keep"
    return state


def node_reflect(state: AgentState, *, ic_threshold: float = 0.01) -> AgentState:
    """更新 Negative RAG 负例库 + 推进迭代计数。"""
    seen = [(a.expression, a.ic_train) for a in state.attempts if a.ic_train is not None]
    state.negative_examples = negative_recall(seen, k=5, ic_threshold=ic_threshold)
    state.iteration += 1
    return state
