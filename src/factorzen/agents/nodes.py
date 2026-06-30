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
            # 归一化后去重（与 node_evaluate 中 seen_expressions 一致）
            try:
                norm = to_expr_string(parse_expr(expr))
            except ValueError:
                norm = expr  # 非法保持原始（evaluate 记 compile_ok=False）
            if norm in state.seen_expressions:
                continue
            ok, _reason = semantic_check(p.hypothesis, expr, llm_fn)
            if ok:
                pending.append(_PendingExpr(p.hypothesis, norm))  # 暂存归一化形式
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
            critic_verdict=None, error=r["error"], ir_train=r["ir_train"]))
        state.seen_expressions.add(r["expression"])
    state._pending = []  # type: ignore[attr-defined]
    return state


def node_guardrails(
    state: AgentState,
    *,
    daily,
    holdout_df,
    ledger,
    top_k: int = 5,
    dsr_threshold: float = 0.5,
) -> AgentState:
    """对过编译的候选记账 N、跑 holdout_ic/DSR，过关者进 candidates。

    灵魂约束：
    - ledger.record(N)：诚实记账本轮所有编译成功且有 IC 的表达式数（多重检验）。
    - holdout_df 只在此节点接触，生成/反思全程不见（隔离）。
    - family-aware 去冗余：新候选与已入选 candidates 截面相关 > 0.7 则跳过。
    """
    import math

    from factorzen.agents.evaluation import _node_to_factor_df
    from factorzen.discovery.scoring import max_correlation
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    from factorzen.validation.holdout import holdout_ic

    passed = [a for a in state.attempts if a.compile_ok and a.ic_train is not None]
    ledger.record(len(passed))  # N 累加本轮所有评估过的表达式（诚实多重检验）
    passed.sort(key=lambda a: abs(a.ic_train), reverse=True)

    # Minor 2：跨 iteration 去重——已入选的表达式不重复入选
    existing_exprs: set[str] = {c["expression"] for c in state.candidates}

    # family-aware pool: {expression: holdout_factor_df} for already-accepted candidates
    pool: dict = {}

    for a in passed[:top_k]:
        # Minor 2：跨 iteration 去重
        if a.expression in existing_exprs:
            continue
        try:
            node = parse_expr(a.expression)
            fdf_hold = _node_to_factor_df(node, holdout_df)
            ic_h, ir_h, _ci = holdout_ic(fdf_hold, holdout_df)
            # Critical: n_obs = 交易日数（IC 序列长度），与 M2 mining_session 口径一致
            n_obs = max(daily["trade_date"].n_unique(), 20)
            # Important: 优先用 ir_train 作 Sharpe 代理（更稳定）；回退到 abs(ic_train)
            sharpe = a.ir_train if a.ir_train is not None else abs(a.ic_train or 0.0)
            dsr, pval = deflated_sharpe(
                sharpe,
                ledger.n_trials,
                n_obs=n_obs,
            )
            # Minor 1: holdout_ic 返回 float（可能 NaN），用 math.isnan 而非 is not None
            if not math.isnan(ic_h) and dsr > dsr_threshold:
                # family-aware 去冗余：新候选与已入选 candidates 截面相关 > 0.7 则跳过
                corr = max_correlation(fdf_hold, pool)
                if corr > 0.7:
                    continue
                a.passed_guardrails = True
                pool[a.expression] = fdf_hold
                existing_exprs.add(a.expression)
                state.candidates.append(
                    {
                        "expression": a.expression,
                        "hypothesis": a.hypothesis,
                        "ic_train": a.ic_train,
                        "holdout_ic": ic_h,
                        "holdout_ir": ir_h,
                        "dsr": dsr,
                        "dsr_pvalue": pval,
                    }
                )
        except Exception:
            continue
    return state


def node_critic(state: AgentState, llm_fn: LLMFn) -> AgentState:
    """LLM 以风控审计员身份批判每个候选：keep/drop/mutate。"""
    for a in state.attempts:
        if a.critic_verdict is not None:
            continue
        msgs = [
            {
                "role": "system",
                "content": (
                    "你是风控审计员，判断因子是否过拟合/经济直觉是否成立，"
                    '只输出 JSON: {"verdict":"keep"|"drop"|"mutate","reason":"..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"假设:{a.hypothesis} 表达式:{a.expression} "
                    f"train_IC:{a.ic_train} 过护栏:{a.passed_guardrails}"
                ),
            },
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
