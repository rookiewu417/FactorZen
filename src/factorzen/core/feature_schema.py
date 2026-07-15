"""Market-data leaf schema shared by ingestion and expression evaluation."""

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

FLOW_FEATURES: set[str] = {"net_mf_amount", "north_ratio"} | MARGIN_FEATURES | TOPLIST_FEATURES
