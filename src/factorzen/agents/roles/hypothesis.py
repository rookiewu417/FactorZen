"""Hypothesis 角色：提经济直觉方向，注入长期记忆（避开已知无效，借鉴已知有效）。"""
from __future__ import annotations

from factorzen.llm.generation import LLMFn, extract_json_items

# 可用信号族——引导 LLM 跳出量价套路，多提与量价正交的基本面/资金面方向（构造因子库需要 breadth）。
_SIGNAL_FAMILIES = (
    "可用信号族：量价（价格/成交量/振幅）、估值（pb/pe/ps）、"
    "**基本面**（roe/roa/毛利率/净利率/负债率/营收增速/净利增速/资产增速，已按公告日 PIT 对齐）、"
    "**资金流/北向**（主力净流入/北向持股占比）。"
    "量价与估值最拥挤、剩余 alpha 少——优先提**基本面/资金面/多族组合**等与量价正交的方向。"
)

# crypto 信号族：无财报/估值，主打资金费率/持仓量/订单流等衍生品特有维度 + 量价。
_CRYPTO_SIGNAL_FAMILIES = (
    "可用信号族：量价（价格/成交量/vwap/收益）、"
    "**资金费率**（funding_rate：拥挤度/情绪，多头付正）、"
    "**持仓量**（open_interest：趋势确认/背离）、"
    "**订单流**（taker_buy_ratio：主动买卖失衡）。"
    "纯量价动量/反转最拥挤——优先提**资金费率/持仓量/订单流**等衍生品特有、与裸量价正交的方向。"
)

# 期货信号族：无财报/估值，主打持仓量/期限结构/商品动量 + 量价（主力连续后复权）。
_FUTURES_SIGNAL_FAMILIES = (
    "可用信号族：量价（价格/成交量/vwap/收益，主力连续后复权跨展期连续）、"
    "**持仓量**（oi/oi_chg：趋势确认/背离，商品特有）、"
    "**动量/反转**（品种截面动量、超跌反弹）、"
    "**量价背离**（价升量缩/持仓背离）。"
    "纯裸量价动量最拥挤——优先提**持仓量/量价背离/期限结构**等商品特有、经济直觉清晰的方向；"
    "品种截面窄(~40-70)，信号须在截面上稳健、勿被单品种主导。"
)

_SIGNAL_FAMILIES_BY_MARKET: dict[str, str] = {
    "ashare": _SIGNAL_FAMILIES,
    "crypto": _CRYPTO_SIGNAL_FAMILIES,
    "futures": _FUTURES_SIGNAL_FAMILIES,
}


def signal_families(market: str = "ashare") -> str:
    """按市场取可用信号族文案。``market="ashare"`` 逐字节返回旧常量（零回归）；
    未登记市场返回通用量价族提示（不抛，不广告不存在的叶子）。"""
    return _SIGNAL_FAMILIES_BY_MARKET.get(
        market, "可用信号族：量价（价格/成交量/收益）。优先经济直觉清晰、换手可控的方向。"
    )


def propose_hypotheses(
    llm_fn: LLMFn,
    *,
    known_invalid: list[str],
    known_valid: list[str],
    feedback: str = "",
    n: int = 1,
    market: str = "ashare",
) -> list[str]:
    """提 n 个经济直觉方向（自然语言）。解析失败 → 空列表。

    ``market``：信号族与市场约束按市场注入（默认 ashare，逐字节零回归）。"""
    sys = (
        "你是量化研究员，提出有经济直觉的选股方向（自然语言，不写公式）。"
        '只输出 JSON: {"hypotheses": ["方向1", "方向2"]}。'
    )
    from factorzen.llm.prompt_fragments import market_caveats
    sys = sys + "\n" + signal_families(market) + "\n" + market_caveats(market)
    user = f"提出 {n} 个新方向。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if known_invalid:
        user += "\n以下表达式已验证无效，避开这些思路:\n" + "\n".join(
            f"- {e}" for e in known_invalid
        )
    if known_valid:
        user += "\n以下表达式已验证有效，可借鉴其思路方向（但不要照抄）:\n" + "\n".join(
            f"- {e}" for e in known_valid
        )
    # extract_json_items 兼容包装对象与裸顶层数组两种真实形状（crypto smoke 实测后者常见）。
    hyps = extract_json_items(
        llm_fn([{"role": "system", "content": sys}, {"role": "user", "content": user}]),
        "hypotheses",
    )
    return [str(h) for h in hyps] if hyps else []


def propose_structured(
    llm_fn: LLMFn,
    *,
    known_invalid: list[str],
    known_valid: list[str],
    feedback: str = "",
    n: int = 1,
    market: str = "ashare",
) -> list[dict]:
    """结构化假设（RD-Agent 步1）：每个含 direction/mechanism/expected_sign/falsification。

    ``market``：信号族与市场约束按市场注入（默认 ashare，逐字节零回归）。"""
    from factorzen.llm.prompt_fragments import market_caveats
    sys = (
        "你是量化研究员，提出结构化选股假设。每个假设含四要素："
        "direction(方向,自然语言)、mechanism(经济机制)、expected_sign(预期IC符号,+1或-1)、"
        "falsification(可证伪判据)。只输出 JSON: "
        '{"hypotheses":[{"direction":"...","mechanism":"...","expected_sign":1,"falsification":"..."}]}。'
    )
    sys = sys + "\n" + signal_families(market) + "\n" + market_caveats(market)
    user = f"提出 {n} 个结构化假设。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if known_invalid:
        user += "\n避开已验证无效:\n" + "\n".join(f"- {e}" for e in known_invalid)
    if known_valid:
        user += "\n可借鉴已验证有效:\n" + "\n".join(f"- {e}" for e in known_valid)
    # extract_json_items 兼容包装对象与裸顶层数组两种真实形状（crypto smoke 实测后者常见）。
    hyps = extract_json_items(
        llm_fn([{"role": "system", "content": sys}, {"role": "user", "content": user}]),
        "hypotheses",
    )
    if not hyps:
        return []
    out: list[dict] = []
    for h in hyps:
        if isinstance(h, dict) and h.get("direction"):
            out.append({
                "direction": str(h.get("direction", "")),
                "mechanism": str(h.get("mechanism", "")),
                "expected_sign": h.get("expected_sign"),
                "falsification": str(h.get("falsification", "")),
            })
    return out


def format_structured(h: dict) -> str:
    """把结构化假设渲染成供 Coder 翻译的自然语言方向文本。"""
    parts = [h.get("direction", "")]
    if h.get("mechanism"):
        parts.append(f"机制: {h['mechanism']}")
    if h.get("expected_sign") is not None:
        parts.append(f"预期IC符号: {h['expected_sign']}")
    if h.get("falsification"):
        parts.append(f"证伪判据: {h['falsification']}")
    return "；".join(p for p in parts if p)
