"""测试因子排名自相关性：衡量因子排序的跨期稳定性。"""

import polars as pl

from daily.evaluation.advanced import RankAutocorrResult, compute_rank_autocorr


def _make_factor_data() -> pl.DataFrame:
    """构造多期因子值数据。"""
    stocks = [f"s{i}" for i in range(20)]
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    rows = []
    for d in dates:
        for s in stocks:
            rows.append({"trade_date": d, "ts_code": s})
    return pl.DataFrame(rows).with_columns(
        # 因子值有序但每天略有波动
        pl.Series(
            "factor_clean",
            [i / 20 for i in range(20)] * 3,    # 完全稳定
        )
    )


def test_rank_autocorr_returns_result_object():
    """compute_rank_autocorr 返回 RankAutocorrResult。"""
    df = _make_factor_data()
    result = compute_rank_autocorr(df, factor_col="factor_clean")
    assert isinstance(result, RankAutocorrResult)


def test_rank_autocorr_returns_series():
    """RankAutocorrResult 包含自相关系数列表。"""
    df = _make_factor_data()
    result = compute_rank_autocorr(df, factor_col="factor_clean")
    assert hasattr(result, "autocorr_values")
    assert isinstance(result.autocorr_values, list)
    assert all(isinstance(v, float) for v in result.autocorr_values)


def test_rank_autocorr_multiple_lags():
    """compute_rank_autocorr 接受多 lag 参数。"""
    df = _make_factor_data()
    result = compute_rank_autocorr(df, factor_col="factor_clean", lags=[1, 2])
    assert len(result.autocorr_values) == 2


def test_rank_autocorr_stable_factor_high_corr():
    """完全稳定的因子排名 → 自相关接近 1.0。"""
    df = _make_factor_data()
    result = compute_rank_autocorr(df, factor_col="factor_clean", lags=[1])
    assert result.autocorr_values[0] > 0.9, "稳定排名的自相关应接近 1.0"


def test_rank_autocorr_per_lag_accessible():
    """可通过 lag 索引获取各滞后期自相关值。"""
    df = _make_factor_data()
    result = compute_rank_autocorr(df, factor_col="factor_clean", lags=[1, 2, 3])
    assert hasattr(result, "get_lag")
    # get_lag(1) 返回 lag=1 的自相关
    ac1 = result.get_lag(1)
    assert isinstance(ac1, float)