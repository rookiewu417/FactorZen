"""Tests for daily.evaluation.correlation (standalone module, not advanced.py)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def _make_df(n: int = 50, n_dates: int = 10, seed: int = 42, negate: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        dt = (date(2023, 1, 3) + timedelta(days=d)).isoformat()
        vals = rng.standard_normal(n)
        for s in range(n):
            v = float(-vals[s] if negate else vals[s])
            rows.append({"trade_date": dt, "ts_code": f"{s:06d}.SZ", "factor_clean": v})
    return pl.DataFrame(rows)


def test_single_factor_returns_identity():
    from factorzen.daily.evaluation.correlation import CorrelationResult, compute_factor_correlation

    df = _make_df(n=50, n_dates=5)
    result = compute_factor_correlation({"A": df})
    assert isinstance(result, CorrelationResult)
    assert result.factor_names == ["A"]
    assert result.corr_matrix.shape == (1, 1)
    assert result.corr_matrix[0][0] == 1.0


def test_identical_factors_have_correlation_one():
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    df = _make_df(n=50, n_dates=10)
    result = compute_factor_correlation({"X": df, "Y": df})
    assert abs(result.corr_matrix[0][1] - 1.0) < 1e-6
    assert abs(result.corr_matrix[1][0] - 1.0) < 1e-6


def test_anti_correlated_factors():
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    df_pos = _make_df(n=50, n_dates=10, seed=7)
    df_neg = _make_df(n=50, n_dates=10, seed=7, negate=True)
    result = compute_factor_correlation({"pos": df_pos, "neg": df_neg})
    assert result.corr_matrix[0][1] < -0.9


def test_sparse_dates_skipped():
    """Dates with fewer than 30 stocks are skipped; diagonal is still 1."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    # Only 10 stocks — every date will be skipped
    df = _make_df(n=10, n_dates=5)
    result = compute_factor_correlation({"A": df, "B": df})
    assert result.corr_matrix[0][0] == 1.0
    assert result.corr_matrix[1][1] == 1.0


def test_zero_variance_factor_skipped():
    """A constant factor has zero std → that date is skipped; diagonal still 1."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    n = 50
    rows_const = [
        {"trade_date": "2023-01-03", "ts_code": f"{i:06d}.SZ", "factor_clean": 1.0}
        for i in range(n)
    ]
    rng = np.random.default_rng(9)
    rows_rand = [
        {"trade_date": "2023-01-03", "ts_code": f"{i:06d}.SZ", "factor_clean": float(rng.standard_normal())}
        for i in range(n)
    ]
    result = compute_factor_correlation(
        {"const": pl.DataFrame(rows_const), "rand": pl.DataFrame(rows_rand)}
    )
    assert result.corr_matrix[0][0] == 1.0
    assert result.corr_matrix[1][1] == 1.0


def test_non_overlapping_stocks_returns_identity():
    """Inner join on non-overlapping ts_code → empty merged → identity matrix."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    n = 50
    rows_a = [{"trade_date": "2023-01-03", "ts_code": f"A{i:05d}.SZ", "factor_clean": float(i)} for i in range(n)]
    rows_b = [{"trade_date": "2023-01-03", "ts_code": f"B{i:05d}.SZ", "factor_clean": float(i)} for i in range(n)]
    result = compute_factor_correlation({"A": pl.DataFrame(rows_a), "B": pl.DataFrame(rows_b)})
    assert result.corr_matrix[0][0] == 1.0
    assert result.corr_matrix[1][1] == 1.0


def test_diagonal_is_always_one():
    """Diagonal elements must be exactly 1 regardless of off-diagonal values."""
    from factorzen.daily.evaluation.correlation import compute_factor_correlation

    df1 = _make_df(n=60, n_dates=10, seed=1)
    df2 = _make_df(n=60, n_dates=10, seed=2)
    df3 = _make_df(n=60, n_dates=10, seed=3)
    result = compute_factor_correlation({"f1": df1, "f2": df2, "f3": df3})
    assert result.corr_matrix.shape == (3, 3)
    for i in range(3):
        assert result.corr_matrix[i][i] == pytest.approx(1.0)


def test_summary_contains_factor_names_and_values():
    from factorzen.daily.evaluation.correlation import CorrelationResult

    result = CorrelationResult(
        factor_names=["alpha", "beta"],
        corr_matrix=np.array([[1.0, 0.42], [0.42, 1.0]]),
    )
    summary = result.summary()
    assert "alpha" in summary
    assert "beta" in summary
    assert "0.420" in summary


def test_summary_single_factor():
    from factorzen.daily.evaluation.correlation import CorrelationResult

    result = CorrelationResult(factor_names=["only"], corr_matrix=np.eye(1))
    summary = result.summary()
    assert "only" in summary
