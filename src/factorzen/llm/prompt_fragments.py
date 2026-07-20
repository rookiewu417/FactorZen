# src/factorzen/llm/prompt_fragments.py
"""LLM 挖掘 prompt 的共享片段（单点维护，避免多处漂移）。

对齐 RD-Agent 研报优化方向①：Prompt 注入 A 股特殊机制说明 + 常见前视偏差陷阱 +
传统风险因子提示，提升 LLM 假设生成的领域深度与可实现性（劣势④：对 A 股特殊机制理解有限）。
"""
from __future__ import annotations

ASHARE_CAVEATS = (
    "A股约束(务必遵守):\n"
    "- 涨跌停(普通±10%/ST±5%)当日难成交,信号避免依赖涨跌停日价格;\n"
    "- 停牌股无行情(已按 vol==0 掩码置 null),勿用停牌日价量;\n"
    "- T+1: 当日买入次日才可卖,超短反转类因子实盘不可实现;\n"
    "- PIT 无未来函数: 只用 ≤t 收盘可得信息(财务按公告日、执行参考 pre_close),"
    "严禁用当日收盘/未来数据造成前视偏差;\n"
    "- 换手率高的因子受交易成本侵蚀(成本双杀),优先经济直觉清晰、换手可控的方向;\n"
    "- 留意市值/流动性等传统风险因子暴露,避免无意中做成风格暴露。"
)

# crypto USDT-M 永续约束。要点：24/7 无交易日历(365 年化)、T+0 可做空无涨跌停、
# funding/OI/taker 语义、尾部厚+单币波动大避免单币叙事、PIT 铁律不变、换手成本更敏感
# (naive 日频截面因子在 crypto 上「零 alpha + 每日换手成本双杀」的教训)。
CRYPTO_CAVEATS = (
    "crypto USDT-M 永续约束(务必遵守):\n"
    "- 24/7 连续交易,无交易日历/无休市(按 365 自然日年化),无价格涨跌幅限制、无停牌;\n"
    "- T+0 双向: 可做多可做空,当日开平仓无隔日限制,反转/多空对冲因子实盘可实现;\n"
    "- funding_rate(资金费率): 8h 结算,多头付正、空头收;可作**拥挤度/情绪**信号"
    "(极端正 funding = 多头拥挤,常预示回调),PIT 用 ≤t 已结算值,勿用未来费率;\n"
    "- open_interest(未平仓量): 持仓规模,升+价升=趋势确认、背离=反转预警;\n"
    "- taker_buy_ratio(主动买占比): 订单流失衡,>0.5=主动买盘占优(短期动量代理);\n"
    "- 尾部厚、单币种波动极大: 避免依赖单一币种叙事,信号要在**截面**上稳健,"
    "警惕小样本/短窗被个别暴涨币主导;\n"
    "- PIT 无未来函数: 只用 ≤t 收盘可得信息,funding/OI 按已披露时点对齐,严禁前视;\n"
    "- 换手成本(taker 费+滑点)比 A 股更重: 高换手因子极易被成本双杀,"
    "优先经济直觉清晰、换手可控的方向。"
)

# 国内商品期货约束。要点：T+0 双向、有涨跌停（MVP ±7% 近似）、主力连续后复权口径、
# 品种截面窄(~40-70)IC 噪声大、商品逻辑族(动量/期限结构/库存-持仓量)、量列跨展期天然跳变。
FUTURES_CAVEATS = (
    "国内商品期货约束(务必遵守):\n"
    "- T+0 双向: 可做多可做空、当日可平,反转/多空对冲因子实盘可实现;\n"
    "- 有涨跌停(各品种/时段幅度不一,MVP 按 ±7% 近似),无停牌次日封板;\n"
    "- **主力连续后复权**: ts_code 为品种连续码(如 CU.SHF),close/open/high/low 已按主力展期"
    "乘法后复权,ret_1d/ts_* 跨展期连续无跳变——**可放心用长窗时序算子**;\n"
    "- **量列不复权**: vol/amount/oi 换主力合约时天然跳变(换更活跃合约),oi_chg 已在展期日置 null,"
    "但用 vol/oi 裸值的长窗时序仍会在展期附近含合约切换效应,优先用比值/截面秩弱化;\n"
    "- open_interest(oi,持仓量): 持仓规模,升+价升=趋势确认、背离=反转预警,是商品特有维度;\n"
    "- **品种截面窄**(约 40-70 个): IC 噪声天然大,信号要在截面上稳健,警惕小样本/单品种(如原油"
    "地缘扰动)主导,避免过度依赖单一品种叙事;\n"
    "- 商品逻辑族: 动量/反转、期限结构(近远月)、库存-持仓量、量价背离——优先经济直觉清晰的方向;\n"
    "- PIT 无未来函数: 主力切换以当日盘后 mapping 为准,只用 ≤t 收盘可得信息,严禁前视;\n"
    "- 换手成本(手续费+滑点)敏感: 高换手因子易被成本侵蚀,优先换手可控的方向。"
)

# 美股约束。要点：无涨跌停 + T+0 可做空、盈利公告跳空、拆股分红已后复权、幸存者偏差(静态成分池)、
# 截面为大型股池(S&P500)、价量族 MVP(无市值/基本面叶子)、PIT 铁律不变。
US_CAVEATS = (
    "美股约束(务必遵守):\n"
    "- 无涨跌停(无涨跌幅限制)、无停牌次日封板; T+0 可当日买卖、可借券做空,反转/多空对冲因子实盘可实现;\n"
    "- **盈利公告跳空**: 财报日常有大幅跳空(隔夜 gap),纯短窗动量/反转因子在财报密集期噪声大、"
    "易被单日跳空主导,信号要在**截面**上稳健;\n"
    "- **拆股/分红已后复权**: close/open/high/low 已按 adjclose 比率复权(ret_1d 跨拆股连续、"
    "无假跳变),**可放心用长窗时序算子**; 量列(vol)未复权,用裸 vol 长窗时序在拆股附近含股数跳变,"
    "优先用 amount(美元成交额,拆股不变量)或比值/截面秩弱化;\n"
    "- **幸存者偏差**(已知 MVP 限制): 标的池为**当前 S&P500 静态成分**、非 PIT 历史成分——"
    "回看窗口天然偏向存活的大盘股,勿把此池的历史表现当无偏结论;\n"
    "- 截面为**大型股池**(S&P500 约 500 只): 流动性好、信息效率高、剩余 alpha 少,"
    "纯量价套路最拥挤,优先经济直觉清晰、换手可控的方向;\n"
    "- 可用叶子仅**价量族**(OHLCV + vwap/log_vol/ret_1d),**无市值/基本面/资金流叶子**"
    "(留二期)——勿臆造不存在的估值/财务字段;\n"
    "- PIT 无未来函数: 只用 ≤t 收盘可得信息,严禁用当日收盘/未来数据造成前视偏差;\n"
    "- 换手成本(佣金零但有滑点+做空借券费)敏感: 高换手因子易被成本侵蚀,优先换手可控的方向。"
)

# 单点维护的市场→约束映射（新增市场在此登记一行，防多处漂移）。
MARKET_CAVEATS: dict[str, str] = {
    "ashare": ASHARE_CAVEATS, "crypto": CRYPTO_CAVEATS, "futures": FUTURES_CAVEATS,
    "us": US_CAVEATS,
}

# 日内特征叶子语义（仅当 leaf_names ∩ INTRADAY_FEATURES 非空时注入；不改既有常量字符串）。
ASHARE_INTRADAY_LEAF_NOTES: str = (
    "日内特征叶子 i_*（已聚合为日频标量，t 日收盘可得、PIT 安全）:\n"
    "- i_rv: 日内已实现波动率；i_rskew/i_rkurt: 日内收益偏度/峰度；\n"
    "- i_downvol_ratio: 下行波动占比；i_updown_vol: 上下行波动比对数；\n"
    "- i_ret_open30/i_ret_close30/i_ret_mid: 开盘/收盘约30分钟与中间时段收益；\n"
    "- i_vwap_dev: 收盘相对全日 VWAP 偏离；i_pv_corr: 价量 Pearson 相关；\n"
    "- i_smart_money: 高冲击桶 VWAP 相对全日 VWAP；\n"
    "- i_vol_open30_share/i_vol_close30_share: 开/收盘约30分钟成交量占比；\n"
    "- i_vol_entropy: 成交量时间分布归一化熵；i_amihud: Amihud 非流动性；\n"
    "- i_path_eff: 价格路径效率；i_max_ret_share: 最大单桶绝对收益占比。\n"
    "涨跌停邻域 i_limit_up_*（A股特有；上面都是连续路径统计，这三个是**硬约束下的"
    "离散状态机**，与日线因子最可能正交）:\n"
    "- i_limit_up_seal_share: 封板时长占比 ∈[0,1]；未封=0（0 有信息，不是缺失）；\n"
    "- i_limit_up_open_count: 开板次数（封住后又打开）；未封=0；\n"
    "- i_limit_up_first_touch: 首次触板相对时刻 ∈(0,1]，越小越早；全日未触=1.0。\n"
    "  注: 单日多数股票取 0/1.0 是正常的，信号常在**截面相对强度 + 短窗记忆**"
    "（如 ts_mean(i_limit_up_seal_share, 5/10/20) 表达「近期爱封板」）。\n"
    "提示: (1) 叶子已是日频聚合，可与日线算子直接组合；"
    "(2) t 日值在 t 日收盘后可得，PIT 安全、无盘中前视；"
    "(3) 与日线量价叶子相关性高，优先找正交方向（结构/路径/聪明钱/时段分解）。"
)

# 未知市场的通用兜底：只保留跨市场恒成立的 PIT 铁律 + 成本提示，不抛异常
# （新市场未登记 caveats 时仍给出核心风控约束，而非空白 prompt）。
_GENERIC_CAVEATS = (
    "通用约束(务必遵守):\n"
    "- PIT 无未来函数: 只用 ≤t 可得信息构造信号,严禁用当日/未来数据造成前视偏差;\n"
    "- 换手率高的因子受交易成本侵蚀(成本双杀),优先换手可控、经济直觉清晰的方向。"
)


# 阈值/游程算子语义(仅当 op_names 含对应算子时注入,避免旧 golden 漂移)
_THRESHOLD_STREAK_OPS = frozenset({"ts_count_gt", "ts_streak_gt", "ts_count_cross_up"})
_THRESHOLD_STREAK_NOTE = (
    "阈值/游程: ts_count_gt(x,y,w)=过去w日x>y占比[0,1]; "
    "ts_streak_gt(x,y,w)=连续x>y天数截断w; "
    "ts_count_cross_up(x,y,w)=过去w日x上穿y次数; y可为常数如0.0。\n"
)


def format_threshold_streak_ops_note(op_names: list[str] | set[str] | tuple[str, ...]) -> str:
    """新算子语义说明。op_names 不含阈值/游程算子 → \"\"(A 股旧 golden 零回归)。"""
    if not _THRESHOLD_STREAK_OPS.intersection(op_names):
        return ""
    return _THRESHOLD_STREAK_NOTE


def format_library_covered(library_covered: list[str] | None) -> str:
    """Render the shared library-coverage hint for all LLM generation paths."""
    if not library_covered:
        return ""
    return "库内已有(追求与其正交,换方向): " + "；".join(library_covered)


def format_library_crowded(crowded: list[tuple[str, int]] | None) -> str:
    """库内拥挤叶子提示。None/[] → \"\"（零回归）。"""
    if not crowded:
        return ""
    body = "、".join(f"{name}({n})" for name, n in crowded)
    return (
        f"库内拥挤叶子(active 数):{body}"
        "——这些方向已充分开采,除非机制明显不同否则避开"
    )


def format_lift_rejected(items: list[dict] | None) -> str:
    """组合层 lift 拒绝方向提示。None/[] → \"\"（零回归）。"""
    if not items:
        return ""
    _REASON_ZH = {
        "below_bar": "组合增量不足",
        "group_gate_fail": "组门整体无增量",
    }
    lines = [
        "以下方向已在组合层证明对当前因子库无增量(lift 拒绝),"
        "避开这些思路及其同源变体(换窗口/换包装的衰减变体同样无增量):"
    ]
    for it in items:
        expr = it.get("expression") or ""
        lift = it.get("lift")
        reason = it.get("lift_reason") or ""
        reason_zh = _REASON_ZH.get(str(reason), str(reason) if reason else "未知")
        if lift is None:
            lift_s = "null"
        else:
            try:
                lift_s = f"{float(lift):g}"
            except (TypeError, ValueError):
                lift_s = str(lift)
        lines.append(f"- {expr}(lift={lift_s}, {reason_zh})")
    return "\n".join(lines)


def format_leaf_guidance(leaf_guidance: dict[str, list[str]] | None) -> str:
    """Render shared exhausted/unexplored leaf guidance for LLM prompts."""
    if not leaf_guidance:
        return ""
    parts: list[str] = []
    exhausted = leaf_guidance.get("exhausted") or []
    unexplored = leaf_guidance.get("unexplored") or []
    if exhausted:
        parts.append("已挖穿(避开,除非机制全新): " + "；".join(exhausted))
    if unexplored:
        parts.append("未探索(优先考虑): " + "、".join(unexplored))
    return "\n".join(parts)


def market_caveats(market: str) -> str:
    """按市场取约束片段。未知市场返回通用 PIT/成本兜底(不抛)。

    ``market="ashare"`` 逐字节返回 ``ASHARE_CAVEATS``（生成/评估侧默认市场，零回归）；
    ``"crypto"`` 返回 ``CRYPTO_CAVEATS``；其余市场返回 ``_GENERIC_CAVEATS``。
    """
    return MARKET_CAVEATS.get(market, _GENERIC_CAVEATS)
