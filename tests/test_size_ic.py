"""测试分市值 IC：按市值分组（大盘/中盘/小盘）计算 Rank IC。"""

import polars as pl
from daily.evaluation.advanced import compute_size_ic, SizeICResult


def _make_size_data(n_stocks: int = 60) -> pl.DataFrame:
    """构造包含不同市值的因子收益数据。"""
    market_caps = (
        [1e10] * 20 + [1e8] * 20 + [1e6] * 20    # 大盘/中盘/小盘
    )
    return pl.DataFrame({
        "ts_code": [f"s{i}" for i in range(n_stocks)],
        "trade_date": ["2026-01-05"] * n_stocks,
        "factor_value": [i / n_stocks for i in range(n_stocks)],
        "fwd_ret": [i / n_stocks * 0.05 for i in range(n_stocks)],
        "market_cap": market_caps[:n_stocks],
    })


def test_size_ic_returns_dataframe():
    """compute_size_ic 返回包含分市值 IC 的 DataFrame。"""
    df = _make_size_data()
    result = compute_size_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", cap_col="market_cap"
    )
    assert isinstance(result, pl.DataFrame)
    assert "size_group" in result.columns or "cap_bucket" in result.columns
    assert "ic" in result.columns


def test_size_ic_multiple_buckets():
    """结果包含多个市值分桶。"""
    df = _make_size_data()
    result = compute_size_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", cap_col="market_cap",
        n_buckets=3,
    )
    assert result.height == 3


def test_size_ic_ic_non_nan():
    """每个分桶的 IC 不应为 NaN（桶内有足够股票时）。"""
    df = _make_size_data()
    result = compute_size_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", cap_col="market_cap",
        n_buckets=3,
    )
    import numpy as np
    ics = result["ic"].to_numpy()
    assert not np.any(np.isnan(ics)), "每个桶有足够样本时 IC 不应为 NaN"


def test_size_ic_result_object():
    """compute_size_ic 可返回 SizeICResult 对象。"""
    df = _make_size_data()
    result = compute_size_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", cap_col="market_cap",
        return_object=True,
    )
    assert isinstance(result, SizeICResult)
    assert hasattr(result, "buckets")
    assert hasattr(result, "summary")