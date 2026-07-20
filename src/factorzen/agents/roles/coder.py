"""Coder 角色：方向 → 表达式；按 Critic 反馈修正表达式。"""
from __future__ import annotations

from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
from factorzen.llm.generation import (
    LLMFn,
    extract_json_items,
    format_leaf_budget_hint,
)


def _syntax_prompt(leaf_budgets: dict[str, int] | None = None, *,
                   market: str = "ashare", leaf_names: list[str] | None = None) -> str:
    """Coder 语法 prompt。``leaf_budgets``：短历史叶子的可用预热预算，非空时追加提示
    （与 `build_agent_messages` 共用 `format_leaf_budget_hint`，双路径不漂移）；
    None/空 → 与改前逐字节相同（零回归）。

    ``market`` / ``leaf_names``：市场约束与叶子清单按市场注入（默认 ashare + A 股叶子，
    逐字节零回归）。crypto 等传各自的 leaf_names + market 触发对应 caveats。"""
    from factorzen.llm.prompt_fragments import (
        format_threshold_streak_ops_note,
        market_caveats,
    )
    leaves = list(LEAF_FEATURES.keys()) if leaf_names is None else leaf_names
    op_list = list(OPERATORS.keys())
    base = (
        "可用算子: " + ", ".join(op_list) + "\n"
        "可用特征(叶子): " + ", ".join(leaves) + "\n"
        "时序算子最后一个参数是整型窗口，如 ts_mean(close, 20)。\n"
        + format_threshold_streak_ops_note(op_list)
        + '只输出 JSON: {"expressions": ["...", "..."]}。'
        "\n" + market_caveats(market)
    )
    hint = format_leaf_budget_hint(leaf_budgets)
    return base + "\n" + hint if hint else base


def write_expressions(
    hypothesis: str, llm_fn: LLMFn, *, avoid: list[str] | None = None,
    leaf_budgets: dict[str, int] | None = None,
    market: str = "ashare", leaf_names: list[str] | None = None,
) -> list[str]:
    user = f"把这个方向翻译成 2-4 个因子表达式: {hypothesis}"
    if avoid:
        user += "\n避免以下已试过/低效的表达式:\n" + "\n".join(f"- {e}" for e in avoid)
    # extract_json_items 兼容包装对象与裸顶层数组两种真实形状（crypto smoke 实测后者常见）。
    exprs = extract_json_items(
        llm_fn(
            [
                {"role": "system",
                 "content": _syntax_prompt(leaf_budgets, market=market, leaf_names=leaf_names)},
                {"role": "user", "content": user},
            ]
        ),
        "expressions",
    )
    return [str(e) for e in exprs] if exprs else []


def revise_expressions(
    hypothesis: str,
    prev_exprs: list[str],
    critic_reason: str,
    llm_fn: LLMFn,
    *,
    leaf_budgets: dict[str, int] | None = None,
    market: str = "ashare", leaf_names: list[str] | None = None,
) -> list[str]:
    user = (
        f"方向: {hypothesis}\n上一版表达式: {', '.join(prev_exprs)}\n"
        f"风控反馈: {critic_reason}\n请按反馈改写出 1-3 个更稳健的表达式。"
    )
    # extract_json_items 兼容包装对象与裸顶层数组两种真实形状（crypto smoke 实测后者常见）。
    exprs = extract_json_items(
        llm_fn(
            [
                {"role": "system",
                 "content": _syntax_prompt(leaf_budgets, market=market, leaf_names=leaf_names)},
                {"role": "user", "content": user},
            ]
        ),
        "expressions",
    )
    return [str(e) for e in exprs] if exprs else []


def revise_from_error(
    hypothesis: str, failed_expr: str, error: str, llm_fn: LLMFn,
    *, leaf_budgets: dict[str, int] | None = None,
    market: str = "ashare", leaf_names: list[str] | None = None,
) -> list[str]:
    """CoSTEER 轻量版：把诊断信息回灌 LLM 修正（DSL 层，无 exec 沙箱）。

    诊断来源有两类：解析报错（语法/未知算子叶子），以及求值期诊断（抛异常、因子值几乎全
    null/NaN、预热不足）。故措辞不限定为「无法解析」。``leaf_budgets`` 非空时把短历史叶子
    的可用预热一并回灌，避免 LLM 又对同一短叶写超预热的长窗口（预热不足回灌专用）。

    ``market`` / ``leaf_names``：按市场注入约束与叶子清单（默认 ashare，零回归）。
    """
    user = (
        f"方向: {hypothesis}\n以下因子表达式存在问题: {failed_expr}\n"
        f"诊断信息: {error}\n"
        f"请修正为既可解析、又能产出有效因子值的表达式（严格遵守语法与可用算子/叶子清单）。"
    )
    # extract_json_items 兼容包装对象与裸顶层数组两种真实形状（crypto smoke 实测后者常见）。
    exprs = extract_json_items(
        llm_fn(
            [
                {"role": "system",
                 "content": _syntax_prompt(leaf_budgets, market=market, leaf_names=leaf_names)},
                {"role": "user", "content": user},
            ]
        ),
        "expressions",
    )
    return [str(e) for e in exprs] if exprs else []


def decompose_tasks(hypothesis: str, llm_fn: LLMFn) -> list[dict]:
    """RD-Agent 步2 任务分解：把假设拆成带 rationale 的因子任务清单（name/description/rationale）。"""
    sys = (
        "把选股假设分解为 1-3 个可实现的因子任务。每个任务含 name(因子名)、"
        "description(一句话描述)、rationale(为何这样构造)。只输出 JSON: "
        '{"tasks":[{"name":"...","description":"...","rationale":"..."}]}。'
    )
    tasks = extract_json_items(
        llm_fn([
            {"role": "system", "content": sys},
            {"role": "user", "content": f"假设: {hypothesis}"},
        ]),
        "tasks",
    )
    if not tasks:
        return []
    return [
        {"name": str(t.get("name", "")), "description": str(t.get("description", "")),
         "rationale": str(t.get("rationale", ""))}
        for t in tasks if isinstance(t, dict)
    ]
