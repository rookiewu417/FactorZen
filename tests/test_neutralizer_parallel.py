"""Tests for joblib-parallel neutralize_ols."""

import numpy as np
import polars as pl


def make_factor_df(n_dates=20, n_stocks=50) -> pl.DataFrame:
    from datetime import date, timedelta

    rng = np.random.default_rng(42)
    start = date(2023, 1, 3)
    rows = []
    industries = [f"ind_{i}" for i in range(5)]
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor": float(rng.standard_normal()),
                    "log_mktcap": float(rng.uniform(10, 20)),
                    "industry": industries[s % 5],
                }
            )
    return pl.DataFrame(rows)


def test_parallel_matches_serial():
    from daily.preprocessing.neutralizer import neutralize_ols

    df = make_factor_df()
    serial = neutralize_ols(df, "factor", n_jobs=1)
    parallel = neutralize_ols(df, "factor", n_jobs=2)
    # Results should be numerically identical
    np.testing.assert_allclose(
        serial["factor"].to_numpy(),
        parallel["factor"].to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    )


def test_serial_baseline():
    from daily.preprocessing.neutralizer import neutralize_ols

    df = make_factor_df()
    result = neutralize_ols(df, "factor", n_jobs=1)
    assert len(result) == len(df)
    assert "factor" in result.columns
