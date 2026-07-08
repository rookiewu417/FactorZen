# src/factorzen/agents/self_heal.py
"""表达式层自愈循环（CoSTEER 轻量落地）——对无法解析的表达式回灌报错让 LLM 修正。

对齐 RD-Agent 最大能力缺口③：CoSTEER「写码→跑→读报错→改→重跑」的 N 轮内循环。
本项目生成的是 DSL 表达式而非可执行 Python，故在**解析层**落地该精神（无 exec/Conda 沙箱、
无 A股回归/安全风险）：parse 失败 → 把 ValueError 回灌 Coder 修正，最多 max_rounds 轮。

N 记账不受影响：heal 只产出「可解析的唯一表达式」，交给下游正常 evaluate+guardrails 记 N
（每个唯一表达式恰好一次，重试不重复计数）。
"""
from __future__ import annotations

from factorzen.agents.roles.coder import revise_from_error
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.llm.generation import LLMFn


def heal_expressions(
    exprs: list[str],
    hypothesis: str,
    llm_fn: LLMFn,
    *,
    max_rounds: int = 3,
) -> list[str]:
    """返回 exprs 中可解析的（归一化）表达式；对不可解析者把报错回灌 LLM 修正，最多 max_rounds 轮。

    可解析的表达式**不触发** LLM 调用（零额外成本）。修正产物再次校验，仍失败则继续下一轮，
    直到 max_rounds 耗尽后丢弃。去重保证同一表达式只保留一次。
    """
    healed: list[str] = []
    seen: set[str] = set()
    pending: list[str] = list(exprs)
    for round_no in range(max_rounds + 1):
        failures: list[tuple[str, str]] = []
        for e in pending:
            try:
                norm = to_expr_string(parse_expr(e))
            except ValueError as exc:
                failures.append((e, str(exc)))
                continue
            if norm not in seen:
                seen.add(norm)
                healed.append(norm)
        if not failures or round_no >= max_rounds:
            break
        pending = []
        for bad_expr, err in failures:
            pending.extend(revise_from_error(hypothesis, bad_expr, err, llm_fn))
    return healed
