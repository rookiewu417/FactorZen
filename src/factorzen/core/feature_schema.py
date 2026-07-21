"""Market-data leaf schema shared by ingestion and expression evaluation."""

# 日内特征叶子（与 intraday.features.battery("v1") 名字一致；写死在此，core 不依赖 intraday）。
INTRADAY_FEATURES: set[str] = {
    "i_rv",
    "i_rskew",
    "i_rkurt",
    "i_downvol_ratio",
    "i_updown_vol",
    "i_ret_open30",
    "i_ret_close30",
    "i_ret_mid",
    "i_vwap_dev",
    "i_pv_corr",
    "i_smart_money",
    "i_vol_open30_share",
    "i_vol_close30_share",
    "i_vol_entropy",
    "i_amihud",
    "i_path_eff",
    "i_max_ret_share",
    # 涨跌停邻域（A 股特有）：前 17 个都是**连续路径统计**（RV/矩/路径效率/量时分布），
    # 这三个是**硬约束下的离散状态机**（触板/封板/开板），机制上不在同一流形。
    "i_limit_up_seal_share",
    "i_limit_up_open_count",
    "i_limit_up_first_touch",
}

LEAF_FEATURES: dict[str, str] = {
    "close": "close_adj",
    "open": "open_adj",
    "high": "high_adj",
    "low": "low_adj",
    "vol": "vol",
    "amount": "amount",
    "vwap": "vwap",
    "log_vol": "log_vol",
    "ret_1d": "ret_1d",
    "amplitude": "amplitude",
    "intraday_ret": "intraday_ret",
    "overnight_ret": "overnight_ret",
    "total_mv": "total_mv",
    "circ_mv": "circ_mv",
    "pb": "pb",
    "pe_ttm": "pe_ttm",
    "ps_ttm": "ps_ttm",
    "dv_ttm": "dv_ttm",
    "turnover_rate": "turnover_rate",
    "turnover_rate_f": "turnover_rate_f",
    "volume_ratio": "volume_ratio",
    "float_share": "float_share",
    "roe": "roe",
    "roa": "roa",
    "grossprofit_margin": "grossprofit_margin",
    "netprofit_margin": "netprofit_margin",
    "debt_to_assets": "debt_to_assets",
    "or_yoy": "or_yoy",
    "netprofit_yoy": "netprofit_yoy",
    "assets_yoy": "assets_yoy",
    "net_mf_amount": "net_mf_amount",
    "north_ratio": "north_ratio",
    "margin_ratio": "margin_ratio",
    "margin_buy_ratio": "margin_buy_ratio",
    "margin_balance": "margin_balance",
    "short_balance": "short_balance",
    "holder_num": "holder_num",
    "holder_num_chg": "holder_num_chg",
    "top_list_net_buy": "top_list_net_buy",
    "top_list_flag": "top_list_flag",
    # 日内特征叶子（恒等映射；必须追加在末尾，既有键相对顺序不可变——随机搜索按键序采样）
    "i_rv": "i_rv",
    "i_rskew": "i_rskew",
    "i_rkurt": "i_rkurt",
    "i_downvol_ratio": "i_downvol_ratio",
    "i_updown_vol": "i_updown_vol",
    "i_ret_open30": "i_ret_open30",
    "i_ret_close30": "i_ret_close30",
    "i_ret_mid": "i_ret_mid",
    "i_vwap_dev": "i_vwap_dev",
    "i_pv_corr": "i_pv_corr",
    "i_smart_money": "i_smart_money",
    "i_vol_open30_share": "i_vol_open30_share",
    "i_vol_close30_share": "i_vol_close30_share",
    "i_vol_entropy": "i_vol_entropy",
    "i_amihud": "i_amihud",
    "i_path_eff": "i_path_eff",
    "i_max_ret_share": "i_max_ret_share",
    "i_limit_up_seal_share": "i_limit_up_seal_share",
    "i_limit_up_open_count": "i_limit_up_open_count",
    "i_limit_up_first_touch": "i_limit_up_first_touch",
    # 业绩预告/快报事件叶（fill-0 事件窗；必须追加在末尾，既有键相对顺序不可变）
    "fc_type_score": "fc_type_score",
    "fc_surprise": "fc_surprise",
    "fc_flag": "fc_flag",
    "express_yoy": "express_yoy",
}

BASIC_FEATURES: set[str] = {
    "total_mv",
    "circ_mv",
    "pb",
    "pe_ttm",
    "ps_ttm",
    "dv_ttm",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "float_share",
}

FUNDAMENTAL_FEATURES: set[str] = {
    "roe",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "or_yoy",
    "netprofit_yoy",
    "assets_yoy",
}

HOLDER_FEATURES: set[str] = {"holder_num", "holder_num_chg"}

MARGIN_FEATURES: set[str] = {
    "margin_ratio",
    "margin_buy_ratio",
    "margin_balance",
    "short_balance",
}

TOPLIST_FEATURES: set[str] = {"top_list_net_buy", "top_list_flag"}

# 业绩预告事件叶（ann_date PIT + 20 交易日窗 fill-0）
FORECAST_FEATURES: set[str] = {"fc_type_score", "fc_surprise", "fc_flag"}

# 业绩快报事件叶
EXPRESS_FEATURES: set[str] = {"express_yoy"}

# 事件 fill-0 叶合集：leaf_health 按源覆盖审计，不按值分布稀疏误杀
EVENT_FILL0_FEATURES: set[str] = FORECAST_FEATURES | EXPRESS_FEATURES

# 事件掩码子集评估叶：表达式引用任一即触发掩码通道（叶原值非零并集，与包装无关）。
# 含预告/快报 fill-0 + 龙虎榜事件叶；值稀疏（is_sparse）仍作无事件叶时的回退。
EVENT_MASK_LEAVES: frozenset[str] = frozenset(
    FORECAST_FEATURES | EXPRESS_FEATURES | TOPLIST_FEATURES
)

FLOW_FEATURES: set[str] = {"net_mf_amount", "north_ratio"} | MARGIN_FEATURES | TOPLIST_FEATURES
