"""测试截面 Z-score 标准化。"""

import polars as pl

from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore


def _make_test_data(values: list[float], stocks: list[str] | None = None):
    """构造测试用 DataFrame。"""
    n = len(values)
    if stocks is None:
        stocks = [f"stock_{i}" for i in range(n)]
    return pl.DataFrame(
        {
            "stock_code": stocks,
            "trade_date": ["2026-01-05"] * n,
            "factor_value_clip_fill": values,
        }
    )


def test_zero_std():
    """所有股票在同一截面上的值相同 → std=0 → zscore 全为 0.0。"""
    df = _make_test_data([5.0, 5.0, 5.0])
    result = cross_sectional_zscore(df)
    col = "factor_value_clip_fill_z"
    assert result[col].to_list() == [0.0, 0.0, 0.0]


def test_single_stock():
    """截面上只有一只股票 → std=0 → 不崩溃，返回 0.0。"""
    df = _make_test_data([5.0], stocks=["stock_0"])
    result = cross_sectional_zscore(df)
    col = "factor_value_clip_fill_z"
    assert result[col].to_list() == [0.0]


def test_normal_case():
    """多只股票不同值 → 正常计算 Z-score。"""
    df = _make_test_data([1.0, 2.0, 3.0])
    result = cross_sectional_zscore(df)
    col = "factor_value_clip_fill_z"
    # Polars std 默认 ddof=1：std([1,2,3]) = 1.0, mean = 2.0
    # z = (x - 2.0) / 1.0 → [-1.0, 0.0, 1.0]
    expected = [-1.0, 0.0, 1.0]
    assert result[col].to_list() == expected
