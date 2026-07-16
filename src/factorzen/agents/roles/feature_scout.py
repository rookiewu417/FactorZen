"""Feature Scout 角色：LLM 提案日内 bar 级表达式（ix_* 叶子候选）。

只负责 prompt + 解析；校验 / 物化 / 筛选在 ``agents.scout_support`` 编排层统一完成，
保证单 Agent 与团队两条路径共用同一验证路径，防双路径漂移。
"""
from __future__ import annotations

from factorzen.discovery.intraday_expr import AGG_FUNCS, BAR_LEAVES, ELEMENTWISE_OPS
from factorzen.llm.generation import LLMFn, extract_json_items


def _scout_system_prompt() -> str:
    """中文 system prompt：叶子表 + 白名单算子 + 聚合清单 + 输出格式与三点约束。"""
    leaves = ", ".join(sorted(BAR_LEAVES.keys()))
    ops = ", ".join(sorted(ELEMENTWISE_OPS))
    aggs = ", ".join(sorted(AGG_FUNCS.keys()))
    return (
        "你是 A 股日内特征 Scout。根据 bar 级叶子与逐元素算子，提出可日聚合的特征表达式。\n"
        f"bar 级叶子（BAR_LEAVES）: {leaves}\n"
        f"允许的逐元素算子（ELEMENTWISE_OPS 白名单）: {ops}\n"
        f"日聚合函数（AGG_FUNCS）: {aggs}\n"
        "规则：\n"
        "1. 每条提案必须有经济直觉（hypothesis 非空、可检验）。\n"
        "2. 避免与 known_features / avoid 中已有特征同方向重复。\n"
        "3. 禁止 ts_* / rank / 截面算子及任何白名单外算子（解析层会直接拒绝）。\n"
        "4. bar_expr 必须用函数式前缀写法（如 sub(high, low)、mul(vol, bar_ret)），"
        "禁止中缀运算符 high-low / close/open。\n"
        "只输出 JSON 数组，元素形如 "
        '{"bar_expr":"sub(high, low)", "agg":"mean", "hypothesis":"..."}；'
        "不要 markdown 围栏，不要其它字段包装。"
    )


def propose_intraday_features(
    llm_fn: LLMFn,
    *,
    k: int,
    avoid: list[str],
    known_features: str,
    market_notes: str = "",
) -> list[dict]:
    """调用 LLM 提案至多 ``k`` 条日内 bar 表达式；返回原始 dict 列表（不做 make_spec）。

    解析容错与 Coder 同款（``extract_json_items``）；LLM 输出畸形或调用失败 → ``[]``（不抛）。
    """
    if k <= 0:
        return []
    user_parts = [
        f"请提出恰好 {k} 条互不重复的日内特征提案。",
    ]
    if known_features:
        user_parts.append("已有/已知特征摘要（避免同向）：\n" + known_features)
    if avoid:
        user_parts.append("避免重复以下已试表达式：\n" + "\n".join(f"- {e}" for e in avoid))
    if market_notes:
        user_parts.append("市场备注：\n" + market_notes)
    user_parts.append(
        '输出 JSON 数组：[{"bar_expr":"...","agg":"...","hypothesis":"..."}, ...]'
    )
    try:
        raw = llm_fn(
            [
                {"role": "system", "content": _scout_system_prompt()},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ]
        )
    except Exception:
        return []

    # 裸顶层数组走 extract_json_items 的 list 回退；包装 {"features":[...]} 也可。
    items = extract_json_items(raw if isinstance(raw, str) else str(raw), "features")
    if not items:
        return []
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bar_expr = it.get("bar_expr")
        agg = it.get("agg")
        if bar_expr is None or agg is None:
            continue
        out.append(
            {
                "bar_expr": str(bar_expr).strip(),
                "agg": str(agg).strip(),
                "hypothesis": str(it.get("hypothesis") or "").strip(),
            }
        )
        if len(out) >= k:
            break
    return out
