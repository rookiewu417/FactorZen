"""测试分行业 IC：按行业分组计算 Rank IC。"""

import polars as pl

from factorzen.daily.evaluation.advanced import SectorICResult, compute_sector_ic


def _make_sector_data(n_stocks: int = 60) -> pl.DataFrame:
    """构造包含多个行业的因子收益数据。"""
    sectors = ["银行", "医药", "科技"] * 20
    return pl.DataFrame(
        {
            "ts_code": [f"s{i}" for i in range(n_stocks)],
            "trade_date": ["2026-01-05"] * n_stocks,
            "factor_value": [i / n_stocks for i in range(n_stocks)],
            "fwd_ret": [i / n_stocks * 0.05 for i in range(n_stocks)],
            "sector": sectors[:n_stocks],
        }
    )


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
        df,
        factor_col="factor_value",
        ret_col="fwd_ret",
        sector_col="sector",
        return_object=True,
    )
    assert isinstance(result, SectorICResult)
    assert hasattr(result, "sector_ic_df")
    assert isinstance(result.sector_ic_df, pl.DataFrame)


def test_sector_ic_excludes_degenerate_two_stock_cross_section():
    """单日单行业仅 2 只股票时，秩相关恒为 ±1（退化、零信息），应被排除，
    而非以虚假的 ±1 污染行业 IC 均值。截面充足的正常行业不受影响。
    """
    rows = []
    # Big：单日 30 只，factor 与 ret 完全同序 → IC≈1（有效，应保留）
    for i in range(30):
        rows.append({
            "ts_code": f"big{i}", "trade_date": "2026-01-05",
            "factor_value": float(i), "fwd_ret": float(i) * 0.01, "sector": "Big",
        })
    # Duo：单日仅 2 只 → n=2 时秩相关恒为 -1（退化），应被排除
    rows.append({"ts_code": "duo0", "trade_date": "2026-01-05",
                 "factor_value": 1.0, "fwd_ret": 0.0, "sector": "Duo"})
    rows.append({"ts_code": "duo1", "trade_date": "2026-01-05",
                 "factor_value": 0.0, "fwd_ret": 1.0, "sector": "Duo"})
    df = pl.DataFrame(rows)

    result = compute_sector_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", sector_col="sector"
    )
    sectors = result["sector"].to_list()
    assert "Big" in sectors, "截面充足(30 只)的正常行业应保留"
    assert "Duo" not in sectors, "单日仅 2 只的退化行业(秩相关恒 ±1)应被排除"


def test_sector_ic_drops_undefined_correlations():
    df = pl.DataFrame(
        {
            "ts_code": ["a", "b", "c", "d", "e", "f"],
            "trade_date": ["2026-01-05"] * 6,
            "factor_value": [1.0, 1.0, 1.0, 1.0, 0.1, 0.2],
            "fwd_ret": [0.01, 0.02, 0.03, 0.04, 0.01, 0.02],
            "sector": ["Constant", "Constant", "Constant", "Constant", "Valid", "Valid"],
        }
    )

    result = compute_sector_ic(
        df, factor_col="factor_value", ret_col="fwd_ret", sector_col="sector"
    )

    assert "Constant" not in result["sector"].to_list()
    assert result["ic"].is_nan().sum() == 0
