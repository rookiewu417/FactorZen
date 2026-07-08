"""Coder 角色：方向 → 表达式；按 Critic 反馈修正表达式。"""
from __future__ import annotations

from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import LLMFn, _extract_json


def _syntax_prompt() -> str:
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS
    return (
        "可用算子: " + ", ".join(OPERATORS.keys()) + "\n"
        "可用特征(叶子): " + ", ".join(LEAF_FEATURES.keys()) + "\n"
        "时序算子最后一个参数是整型窗口，如 ts_mean(close, 20)。\n"
        '只输出 JSON: {"expressions": ["...", "..."]}。'
        "\n" + ASHARE_CAVEATS
    )


def write_expressions(
    hypothesis: str, llm_fn: LLMFn, *, avoid: list[str] | None = None
) -> list[str]:
    user = f"把这个方向翻译成 2-4 个因子表达式: {hypothesis}"
    if avoid:
        user += "\n避免以下已试过/低效的表达式:\n" + "\n".join(f"- {e}" for e in avoid)
    obj = _extract_json(
        llm_fn(
            [
                {"role": "system", "content": _syntax_prompt()},
                {"role": "user", "content": user},
            ]
        )
    )
    if not obj:
        return []
    exprs = obj.get("expressions")
    return [str(e) for e in exprs] if isinstance(exprs, list) else []


def revise_expressions(
    hypothesis: str,
    prev_exprs: list[str],
    critic_reason: str,
    llm_fn: LLMFn,
) -> list[str]:
    user = (
        f"方向: {hypothesis}\n上一版表达式: {', '.join(prev_exprs)}\n"
        f"风控反馈: {critic_reason}\n请按反馈改写出 1-3 个更稳健的表达式。"
    )
    obj = _extract_json(
        llm_fn(
            [
                {"role": "system", "content": _syntax_prompt()},
                {"role": "user", "content": user},
            ]
        )
    )
    if not obj:
        return []
    exprs = obj.get("expressions")
    return [str(e) for e in exprs] if isinstance(exprs, list) else []
