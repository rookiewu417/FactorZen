"""Agent 闭环的函数式节点：node(State) -> State。"""
from __future__ import annotations

from dataclasses import dataclass, field

from factorzen.agents.evaluation import evaluate_expressions
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
            critic_verdict=None, error=r["error"]))
        state.seen_expressions.add(r["expression"])
    state._pending = []  # type: ignore[attr-defined]
    return state
