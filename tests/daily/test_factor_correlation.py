"""
test_factor_correlation.py：因子相关性矩阵测试。
test_factor_correlation_module.py：Tests for daily.evaluation.correlation (standalone module, not advanced.py).
test_factor_crowding.py：测试因子拥挤度：衡量多个因子间的信号相似度。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.advanced import (
    compute_factor_crowding,
)


# ==== 来自 test_factor_correlation.py ====
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
    from factorzen.daily.evaluation.advanced import compute_factor_correlation

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
    from factorzen.daily.evaluation.advanced import compute_factor_correlation

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



def test_single_factor_returns_identity__factor_correlation():
    """单因子应返回 1x1 矩阵，对角线为 1。"""
    from factorzen.daily.evaluation.advanced import compute_factor_correlation

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

# ==== 来自 test_factor_correlation_module.py ====
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


def test_single_factor_returns_identity__factor_correlation_module():
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


# ==== 来自 test_factor_crowding.py ====
def _make_factor_dict() -> dict[str, pl.DataFrame]:
    """构造多个因子数据的字典。"""
    stocks = [f"s{i}" for i in range(50)]
    base = pl.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 50,
            "ts_code": stocks,
        }
    )
    # 因子 A 和 B 强相关（线性相关），因子 C 独立
    return {
        "momentum": base.with_columns(pl.lit(0.5).alias("factor_clean")),
        "value": base.with_columns(pl.Series("factor_clean", [i / 50 for i in range(50)])),
        "low_vol": base.with_columns(pl.Series("factor_clean", [i / 50 * (-1) for i in range(50)])),
    }




def test_crowding_diagonal_is_one():
    """相关性矩阵对角线为 1.0。"""
    factor_dict = _make_factor_dict()
    result = compute_factor_crowding(factor_dict, factor_col="factor_clean")
    n = len(result.factor_names)
    for i in range(n):
        assert abs(result.corr_matrix[i][i] - 1.0) < 1e-10



