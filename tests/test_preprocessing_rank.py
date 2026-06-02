"""Tests for cross_sectional_rank and quantile_transform."""

import numpy as np
import polars as pl


def make_df(n=100) -> pl.DataFrame:
    """100 stocks × 5 dates."""
    dates = [f"2024-01-{d+1:02d}" for d in range(5)]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        vals = rng.standard_normal(n)
        for i, v in enumerate(vals):
            rows.append({"trade_date": d, "ts_code": f"00{i:04d}.SZ", "factor": v})
    return pl.DataFrame(rows)


def test_rank_uniform_in_01():
    from factorzen.daily.preprocessing.normalizer import cross_sectional_rank

    df = make_df()
    result = cross_sectional_rank(df, "factor", method="uniform")
    vals = result["factor"].drop_nulls()
    assert vals.min() > 0.0
    assert vals.max() < 1.0


def test_rank_normal_approx_standard_normal():
    from scipy.stats import kstest

    from factorzen.daily.preprocessing.normalizer import cross_sectional_rank

    df = make_df(500)
    result = cross_sectional_rank(df, "factor", method="normal")
    vals = result["factor"].drop_nulls().to_numpy()
    _stat, p = kstest(vals, "norm")
    assert p > 0.01  # not rejected at 1%


def test_quantile_transform_constant_column():
    from factorzen.daily.preprocessing.normalizer import quantile_transform

    df = pl.DataFrame(
        {
            "trade_date": ["2024-01-01"] * 10,
            "ts_code": [f"00{i:04d}.SZ" for i in range(10)],
            "factor": [1.0] * 10,
        }
    )
    result = quantile_transform(df, "factor")
    # Should not raise; all values same (constant)
    assert len(result) == 10


def test_quantile_transform_schema_preserved():
    from factorzen.daily.preprocessing.normalizer import quantile_transform

    df = make_df()
    result = quantile_transform(df, "factor")
    assert result.schema == df.schema
    assert len(result) == len(df)
