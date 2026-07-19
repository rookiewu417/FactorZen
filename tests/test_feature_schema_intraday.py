"""日内特征叶子注册：单源守卫 + 键序零回归。"""
from __future__ import annotations

# 改造前 LEAF_FEATURES 的 40 个键顺序（硬编码守卫：只许末尾扩，不许改旧序）
_PRE_CHANGE_LEAF_KEYS: list[str] = [
    "close",
    "open",
    "high",
    "low",
    "vol",
    "amount",
    "vwap",
    "log_vol",
    "ret_1d",
    "amplitude",
    "intraday_ret",
    "overnight_ret",
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
    "roe",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "or_yoy",
    "netprofit_yoy",
    "assets_yoy",
    "net_mf_amount",
    "north_ratio",
    "margin_ratio",
    "margin_buy_ratio",
    "margin_balance",
    "short_balance",
    "holder_num",
    "holder_num_chg",
    "top_list_net_buy",
    "top_list_flag",
]


def test_intraday_features_subset_of_leaf_features():
    from factorzen.core.feature_schema import INTRADAY_FEATURES, LEAF_FEATURES

    assert set(LEAF_FEATURES.keys()) >= INTRADAY_FEATURES
    assert len(INTRADAY_FEATURES) == 20  # 17 连续路径统计 + 3 涨跌停邻域


def test_intraday_leaves_are_identity_i_prefix():
    from factorzen.core.feature_schema import INTRADAY_FEATURES, LEAF_FEATURES

    for name in INTRADAY_FEATURES:
        assert name.startswith("i_"), name
        assert LEAF_FEATURES[name] == name


def test_leaf_features_key_order_prefix_unchanged():
    """既有 40 键相对顺序绝不动（随机搜索按键序采样）。"""
    from factorzen.core.feature_schema import LEAF_FEATURES

    keys = list(LEAF_FEATURES.keys())
    n = len(_PRE_CHANGE_LEAF_KEYS)
    assert keys[:n] == _PRE_CHANGE_LEAF_KEYS
    # 新叶子全部在旧键之后
    assert all(k.startswith("i_") for k in keys[n:])


def test_intraday_features_match_battery_v1_names():
    """与 battery('v1') 名字集合一致（单源守卫；本测试可 import battery）。"""
    from factorzen.core.feature_schema import INTRADAY_FEATURES
    from factorzen.intraday.features import battery

    battery_names = {s.name for s in battery("v1")}
    assert battery_names == INTRADAY_FEATURES
