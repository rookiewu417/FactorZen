"""测试分行业 IC：按行业分组计算 Rank IC。"""

import polars as pl

from daily.evaluation.advanced import SectorICResult, compute_sector_ic


def _make_sector_data(n_stocks: int = 60) -> pl.DataFrame:
    """构造包含多个行业的因子收益数据。"""
    sectors = ["银行", "医药", "科技"] * 20
    return pl.DataFrame({
        "ts_code": [f"s{i}" for i in range(n_stocks)],
        "trade_date": ["2026-01-05"] * n_stocks,
        "factor_value": [i / n_stocks for i in range(n_stocks)],
        "fwd_ret": [i / n_stocks * 0.05 for i in range(n_stocks)],
        "sector": sectors[:n_stocks],
    })


def test_sector_ic_returns_dataframe():
    """compute_sector_ic 返回包含各行业 IC 的 DataFrame。"""
    df = _make_sector_data()
    result = compute_sector_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", sector_col="sector"
    )
    assert isinstance(result, pl.DataFrame)
    assert "sector" in result.columns
    assert "ic" in result.columns


def test_sector_ic_returns_all_sectors():
    """结果包含所有输入行业。"""
    df = _make_sector_data()
    result = compute_sector_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", sector_col="sector"
    )
    input_sectors = set(df["sector"].unique())
    result_sectors = set(result["sector"].unique())
    assert input_sectors == result_sectors, "所有行业应出现在结果中"


def test_sector_ic_non_nan():
    """行业 IC 不应为 NaN（行业内有足够股票时）。"""
    df = _make_sector_data()
    result = compute_sector_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", sector_col="sector"
    )
    ics = result["ic"].to_numpy()
    import numpy as np
    assert not np.any(np.isnan(ics)), "有足够样本时行业 IC 不应为 NaN"


def test_sector_ic_returns_sector_ic_result():
    """compute_sector_ic 也可以返回 SectorICResult 对象。"""
    df = _make_sector_data()
    result = compute_sector_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", sector_col="sector",
        return_object=True,
    )
    assert isinstance(result, SectorICResult)
    assert hasattr(result, "sector_ic_df")
    assert isinstance(result.sector_ic_df, pl.DataFrame)