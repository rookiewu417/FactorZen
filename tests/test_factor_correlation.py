"""因子相关性矩阵测试。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_factor_df(vals_fn, n: int = 50, n_dates: int = 10, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"{s:06d}.SZ",
                    "factor_clean": float(vals_fn(vals, s)),
                }
            )
    return pl.DataFrame(rows)


def test_self_correlation_is_one():
    from daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(42)
    n = 50
    rows = []
    for d in range(10):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])})
    df = pl.DataFrame(rows)
    corr = compute_factor_correlation({"A": df, "B": df})
    # A vs A = 1, B vs B = 1, and A vs B should be 1 (same data)
    assert abs(corr.filter(pl.col("factor") == "A")["A"][0] - 1.0) < 1e-6
    assert abs(corr.filter(pl.col("factor") == "B")["B"][0] - 1.0) < 1e-6


def test_opposite_factor_correlation_is_negative():
    from daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(0)
    n = 50
    rows_pos, rows_neg = [], []
    for d in range(10):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows_pos.append(
                {"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])}
            )
            rows_neg.append(
                {"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(-vals[s])}
            )
    df_pos = pl.DataFrame(rows_pos)
    df_neg = pl.DataFrame(rows_neg)
    corr = compute_factor_correlation({"pos": df_pos, "neg": df_neg})
    corr_val = corr.filter(pl.col("factor") == "pos")["neg"][0]
    assert corr_val < -0.9


def test_factor_correlation_returns_dataframe():
    """返回值应为 pl.DataFrame 含 'factor' 列。"""
    from daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(1)
    n = 30
    rows = []
    for d in range(5):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])})
    df = pl.DataFrame(rows)
    result = compute_factor_correlation({"X": df, "Y": df})
    assert isinstance(result, pl.DataFrame)
    assert "factor" in result.columns
    assert "X" in result.columns
    assert "Y" in result.columns


def test_single_factor_returns_identity():
    """单因子应返回 1x1 矩阵，对角线为 1。"""
    from daily.evaluation.advanced import compute_factor_correlation

    rng = np.random.default_rng(2)
    n = 20
    rows = []
    for d in range(5):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": float(vals[s])})
    df = pl.DataFrame(rows)
    result = compute_factor_correlation({"only": df})
    assert result.filter(pl.col("factor") == "only")["only"][0] == 1.0
