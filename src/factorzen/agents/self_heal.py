# src/factorzen/agents/self_heal.py
"""表达式层自愈循环（CoSTEER 轻量落地）——把诊断信息回灌 LLM 修正表达式。

对齐 RD-Agent 最大能力缺口③：CoSTEER「写码→跑→读报错→改→重跑」的 N 轮内循环。
本项目生成的是 DSL 表达式而非可执行 Python，故在**解析层 + 求值层**落地该精神
（无 exec/Conda 沙箱、无 A股回归/安全风险）：

- 解析失败（parse_expr 抛 ValueError）→ 报错回灌；
- 求值失败 / 因子值几乎全 null（health_check 给出诊断）→ 诊断回灌。

后者是 CoSTEER 的另一半：其评估器在沙箱里真正执行代码，把 Traceback **和 NaN 比例**
交给模型修正。只查 parse 会放行 `div(close, sub(close, close))` 这类「parse 通过、
求值全 null」的静默失明表达式（PR #61 的嵌套 .over() bug 同型）。

N 记账不受影响：heal 只产出「可解析且健康的唯一表达式」，交给下游正常 evaluate+guardrails
记 N（每个唯一表达式恰好一次，重试不重复计数）。
"""
from __future__ import annotations

from collections.abc import Callable

from factorzen.agents.roles.coder import revise_from_error
from factorzen.discovery.expression import parse_expr, to_expr_string
from factorzen.llm.generation import LLMFn

HealthCheck = Callable[[str], str | None]


def heal_expressions(
    exprs: list[str],
    hypothesis: str,
    llm_fn: LLMFn,
    *,
    max_rounds: int = 3,
    health_check: HealthCheck | None = None,
    leaf_map: dict[str, str] | None = None,
    market: str = "ashare",
    leaf_names: list[str] | None = None,
) -> list[str]:
    """返回 exprs 中可解析（且通过 health_check）的归一化表达式；病态者回灌 LLM 修正。

    health_check(expr) 返回诊断字符串表示不健康、None 表示健康；为 None 时只查解析
    （零回归：既有调用方行为不变）。健康表达式**不触发** LLM 调用（零额外成本）。
    修正产物再次校验，仍失败则继续下一轮，直到 max_rounds 耗尽后丢弃。
    去重保证同一表达式只保留一次，也避免对同一病态表达式反复求医。

    ``leaf_map`` / ``market`` / ``leaf_names``：市场上下文（默认 None/ashare → A 股，零回归）。
    crypto 必须传 ``leaf_map``，否则合法 crypto 叶子被 `parse_expr` 判为解析失败，健康的
    crypto 表达式被误当病态送修（浪费 LLM 调用 + 可能被改坏）；``market``/``leaf_names``
    透传给 `revise_from_error` 使修正 prompt 用对市场的约束与叶子清单。
    """
    healed: list[str] = []
    seen: set[str] = set()
    tried: set[str] = set()
    pending: list[str] = list(exprs)
    for round_no in range(max_rounds + 1):
        failures: list[tuple[str, str]] = []
        for e in pending:
            try:
                norm = to_expr_string(parse_expr(e, leaf_map))
            except ValueError as exc:
                if e not in tried:
                    tried.add(e)
                    failures.append((e, str(exc)))
                continue
            if norm in seen or norm in tried:
                continue
            diagnosis = health_check(norm) if health_check is not None else None
            if diagnosis is not None:
                tried.add(norm)
                failures.append((norm, diagnosis))
                continue
            seen.add(norm)
            healed.append(norm)
        if not failures or round_no >= max_rounds:
            break
        pending = []
        for bad_expr, err in failures:
            pending.extend(revise_from_error(hypothesis, bad_expr, err, llm_fn,
                                             market=market, leaf_names=leaf_names))
    return healed
