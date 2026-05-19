"""Pearson IC 测试。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def make_factor_ret_df(n_stocks: int = 50, n_dates: int = 20, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        factors = rng.standard_normal(n_stocks)
        rets = factors * 0.01 + rng.standard_normal(n_stocks) * 0.02  # positive IC
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(factors[s]),
                    "ret_1d": float(rets[s]),
                }
            )
    return pl.DataFrame(rows)


def test_pearson_ic_is_positive():
    """已知正信号因子的 Pearson IC 应为正。"""
    from daily.evaluation.ic_analysis import compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="pearson")
    assert result.ic_mean > 0


def test_rank_ic_is_positive():
    from daily.evaluation.ic_analysis import compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="rank")
    assert result.ic_mean > 0


def test_both_ic_returns_dict():
    """method='both' 应返回含 rank 和 pearson 两个 IcStats 的字典。"""
    from daily.evaluation.ic_analysis import IcStats, compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="both")
    assert "rank" in result
    assert "pearson" in result
    assert isinstance(result["rank"], IcStats)
    assert isinstance(result["pearson"], IcStats)


def test_heavy_tail_pearson_less_than_rank():
    """重尾因子（单个极端值）Pearson IC 受影响更大，绝对值应小于 Rank IC。"""
    from daily.evaluation.ic_analysis import compute_ic

    rng = np.random.default_rng(0)
    n_stocks = 100
    n_dates = 20
    start = date(2023, 1, 3)
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        factors = rng.standard_normal(n_stocks)
        factors[0] = 1000.0  # extreme outlier
        rets = np.sign(factors) * 0.01 + rng.standard_normal(n_stocks) * 0.02
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(factors[s]),
                    "ret_1d": float(rets[s]),
                }
            )
    df = pl.DataFrame(rows)
    pearson_res = compute_ic(df, method="pearson")
    rank_res = compute_ic(df, method="rank")
    # Pearson is disturbed by outlier; rank is more robust
    # Rank IC should be >= Pearson IC in absolute value (rank is outlier-robust)
    assert abs(rank_res.ic_mean) >= abs(pearson_res.ic_mean)


def test_ic_stats_fields():
    """IcStats 应含预期字段。"""
    from daily.evaluation.ic_analysis import IcStats, compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="rank")
    assert isinstance(result, IcStats)
    assert hasattr(result, "ic_mean")
    assert hasattr(result, "ic_std")
    assert hasattr(result, "ir")
    assert hasattr(result, "ic_positive_ratio")
    assert hasattr(result, "n_periods")
    assert hasattr(result, "ic_tstat")
    assert hasattr(result, "ic_pvalue")
    assert hasattr(result, "ic_series")
