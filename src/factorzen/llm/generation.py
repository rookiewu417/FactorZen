"""LLM 因子生成层：假设 + 表达式提议 + 语义对齐自检 + prompt 模板。"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from factorzen.llm.prompt_fragments import (
    format_leaf_guidance,
    format_library_covered,
)

LLMFn = Callable[[list[dict[str, str]]], str]


@dataclass
class FactorProposal:
    hypothesis: str
    expressions: list[str]
    rationale: str


def _extract_json(raw: str) -> dict | None:
    """容错解析：直接 json.loads；失败找首个 {...} 子串；再失败返回 None。"""
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _extract_json_list(raw: str) -> list | None:
    """容错解析**顶层 JSON 数组**：`[...]` 或围栏包裹的数组；非数组/失败返回 None。

    为什么需要它：模型（实测 DeepSeek）即使被指示输出 `{"hypotheses": [...]}`，也常直接
    返回裸数组 `[{...}, {...}]`。`_extract_json` 只认 dict，其 `{...}` 子串回退会截出
    「首元素开括号..末元素闭括号」的**非法两对象片段**，解析恒失败 → 整轮假设被静默丢弃
    （crypto smoke 实测 4/6 轮空转）。调用方在 `_extract_json` 拿不到目标键时回退到本函数。
    """
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else None
    except Exception:
        pass
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, list) else None
        except Exception:
            return None
    return None


def extract_json_items(raw: str, key: str) -> list | None:
    """从 LLM 响应中取「键为 *key* 的列表」，兼容两种真实出现的形状（单点维护，防漂移）：

    1. 包装对象 ``{"<key>": [...]}``（prompt 要求的形状，优先）；
    2. 裸顶层数组 ``[...]``（模型常见的「偷懒」形状，回退接受）。

    解析失败或形状不符返回 None（调用方保持各自的降级语义）。
    """
    obj = _extract_json(raw)
    if obj is not None:
        items = obj.get(key)
        return items if isinstance(items, list) else None
    return _extract_json_list(raw)


def generate_factor_proposal(
    messages: list[dict[str, str]],
    llm_fn: LLMFn,
    *,
    n_hypotheses: int = 1,
) -> list[FactorProposal]:
    """调用 LLM 生成 1+ 个 (假设, 表达式集)。解析失败的丢弃（降级不抛）。"""
    proposals: list[FactorProposal] = []
    for _ in range(max(1, n_hypotheses)):
        obj = _extract_json(llm_fn(messages))
        if not obj:
            continue
        exprs = obj.get("expressions")
        if not isinstance(exprs, list) or not exprs:
            continue
        proposals.append(
            FactorProposal(
                hypothesis=str(obj.get("hypothesis", "")),
                expressions=[str(e) for e in exprs],
                rationale=str(obj.get("rationale", "")),
            )
        )
    return proposals


def semantic_check(
    hypothesis: str, expression: str, llm_fn: LLMFn
) -> tuple[bool, str]:
    """LLM 自查表达式是否实现假设。返回 (一致?, 理由)。解析失败 → (True, '') 放行（避免误杀）。"""
    msgs = [
        {
            "role": "system",
            "content": (
                "你判断量化因子表达式是否与给定假设**方向一致**——只要表达式捕捉的信号方向"
                "与假设相符，或实现了假设的某个**核心侧面**，即算一致（consistent=true）；"
                "无需完整覆盖复合假设的每个条件。仅当表达式与假设**明显无关或方向相反**时"
                "才判 false（宁可放行也不误杀合理因子）。只输出 JSON: "
                '{"consistent": true/false, "reason": "..."}'
            ),
        },
        {"role": "user", "content": f"假设: {hypothesis}\n表达式: {expression}"},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or "consistent" not in obj:
        return True, ""  # 解析失败放行，不误杀
    return bool(obj["consistent"]), str(obj.get("reason", ""))


def format_leaf_budget_hint(leaf_budgets: dict[str, int] | None) -> str:
    """把「短历史叶子的可用预热预算」渲染成一句 prompt 提示（两条生成路径共用，防漂移）。

    ``leaf_budgets``：``{叶子名: 可用预热 bar 数}``，由调用方用 `leaf_warmup_budgets` 算出、
    并只保留历史较短（< 预热前缀）的叶子后传入。空/None → 返回空串（零回归：无提示文案）。

    单/团队两条生成路径（`build_agent_messages` 与 coder `_syntax_prompt`）都调本函数，
    保证同一 budgets 产出逐字节相同的提示——双路径登记簿要求改一侧必改另一侧，共用即免漂移。
    """
    if not leaf_budgets:
        return ""
    caps = "、".join(f"{leaf} ≤ {bars} 根"
                     for leaf, bars in sorted(leaf_budgets.items()))
    return (
        "以下叶子历史较短，表达式中含该叶子的**路径累计窗口**（嵌套时序算子的窗口之和）"
        "不得超过其可用预热，否则会被直接拒绝评估、浪费一次尝试：" + caps + "。"
    )


# 叶子族语义指引（市场特有：A 股财报/资金流，crypto 衍生品特有信号）。单点维护防漂移。
# ashare：中性列举可用族，具体优先/避开由动态 leaf_guidance（Librarian）引导；未登记市场 → 空串
# （不广告不存在的叶子——能力层↔接线层漂移教训）。
_LEAF_GUIDANCE: dict[str, str] = {
    "ashare": (
        "其中 roe/roa/grossprofit_margin/netprofit_margin/debt_to_assets(质量) 与 "
        "or_yoy/netprofit_yoy/assets_yoy(成长) 是**财报基本面**(已按公告日 PIT 对齐，无未来函数)；"
        "holder_num/holder_num_chg 是**股东户数**叶子（按 ann_date PIT；"
        "holder_num_chg 为相邻两期环比，源侧算好随公告生效）；"
        "net_mf_amount(主力资金净流入) 等是**资金流**叶子；"
        "margin_ratio/margin_buy_ratio/margin_balance/short_balance 是**两融/杠杆情绪**叶子"
        "（T 日两融 T+1 早间披露，attach 已内置 lag(1)；rzye/rzmre 单位元，"
        "margin_ratio=rzye/(circ_mv×1e4)，margin_buy_ratio=rzmre/(amount×1e3)）；"
        "top_list_net_buy/top_list_flag 是**龙虎榜**叶子"
        "（t 日榜单 t 日盘后披露，attach 已内置 lag(1)；已知日未上榜=0，未拉取=null；"
        "net_amount 万元、amount 千元，比前统一到元）——"
        "与量价正交的族可作多族组合，避开拥挤方向，别只盯量价波动。\n"
    ),
    "crypto": (
        "其中 funding_rate(资金费率,多头付正=拥挤度/情绪)、open_interest(未平仓量,趋势确认/背离)、"
        "taker_buy_ratio(主动买占比,订单流失衡) 是**衍生品特有信号**——"
        "与裸量价正交，优先用它们构造资金费率/持仓量/订单流方向，别只盯价格动量。\n"
    ),
    "futures": (
        "其中 oi(持仓量)、oi_chg(持仓变化率,展期日已置 null) 是**商品特有信号**(趋势确认/背离)；"
        "close/open/high/low/vwap 为主力连续**后复权**价(ts_* 跨展期连续)、vol/amount/oi 为量列"
        "(换主力天然跳变)——优先用持仓量/量价背离/期限结构等与裸量价正交的方向，别只盯价格动量。\n"
    ),
    "us": (
        "其中 close/open/high/low/vwap 为**后复权**价(拆股/分红已调整,ret_1d 跨拆股连续,"
        "vwap=后复权典型价 (high+low+close)/3)、amount=美元成交额(拆股不变量)、vol=原始股数(未复权)；"
        "**仅价量族叶子(无市值/基本面/资金流)**——用量价背离/波动/振幅/反转等经济直觉清晰的方向,"
        "大盘股截面稳健优先,勿臆造不存在的估值/财务字段。\n"
    ),
}


def build_agent_messages(
    op_names: list[str],
    leaf_names: list[str],
    feedback: str = "",
    negatives: list[str] | None = None,
    leaf_budgets: dict[str, int] | None = None,
    market: str = "ashare",
    leaf_guidance: dict[str, list[str]] | None = None,
    library_covered: list[str] | None = None,
) -> list[dict[str, str]]:
    """构造生成 prompt：算子/特征清单 + 上轮反馈 + Negative RAG 负例 + 短历史叶子预热预算。

    ``leaf_budgets``：``{短历史叶子名: 可用预热 bar 数}``（默认 None → 不追加预算提示）。
    非空时追加一句预热预算提示，引导 LLM 别对短历史叶子写超预热的长窗口。

    ``market``：叶子族指引与市场约束按市场注入（默认 ashare）；未登记市场
    只列算子/叶子 + 通用约束，不广告不存在的叶子族。

    ``leaf_guidance``：Librarian 叶子级挖穿/未探索（与 team Hypothesis 共用
    ``format_leaf_guidance``）；None → 不注入。

    ``library_covered``：库内 active 高 IC 表达式（与 team Hypothesis 共用
    ``format_library_covered``）；None → 不注入。
    """
    neg = negatives or []
    system = (
        "你是量化研究员，提出有经济直觉的假设并翻译成因子表达式。\n"
        "假设必须是**单一机制、可用一个截面因子直接实现**的方向性命题"
        "（如「高换手率的股票未来收益更低」）；不要写多条件、带时序先后的复合叙事"
        "（如「缩量整固后再放量突破」）——单个表达式实现不了它，会被语义自检整批否掉。\n"
        f"可用算子: {', '.join(op_names)}\n"
        f"可用特征(叶子): {', '.join(leaf_names)}\n"
        + _LEAF_GUIDANCE.get(market, "")
        + "时序算子最后一个参数是整型窗口，如 ts_mean(close, 20)。\n"
        "表达式只能用上面列出的算子写成**函数式**，禁止中缀运算符 + - * /"
        "（用 add/sub/mul/div 代替，如 div(close, open) 而非 close / open）。\n"
        '只输出 JSON: {"hypothesis": "...", "expressions": ["...", "..."], "rationale": "..."}'
    )
    from factorzen.llm.prompt_fragments import market_caveats
    system = system + "\n" + market_caveats(market)
    # 仅当本轮可用叶子含 i_* 时注入日内语义表（零回归：不含 i_* 时 system 逐字节不变）
    from factorzen.core.feature_schema import INTRADAY_FEATURES
    if set(leaf_names) & INTRADAY_FEATURES:
        from factorzen.llm.prompt_fragments import ASHARE_INTRADAY_LEAF_NOTES
        system = system + "\n" + ASHARE_INTRADAY_LEAF_NOTES
    hint = format_leaf_budget_hint(leaf_budgets)
    if hint:
        system = system + "\n" + hint
    user = "提出一个新假设并给出 2-4 个候选表达式。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if neg:
        user += "\n避免以下已探索过/低效的模式:\n" + "\n".join(f"- {n}" for n in neg)
    lg = format_leaf_guidance(leaf_guidance)
    if lg:
        user += "\n" + lg
    lc = format_library_covered(library_covered)
    if lc:
        user += "\n" + lc
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
