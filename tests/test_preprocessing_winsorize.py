"""Tests for winsorize_percentile and sigma_clip."""

import numpy as np
import polars as pl


def test_winsorize_clips_extremes():
    from daily.preprocessing.outlier import winsorize_percentile

    df = pl.DataFrame(
        {
            "trade_date": ["2024-01-01"] * 100,
            "ts_code": [f"00{i:04d}.SZ" for i in range(100)],
            "factor": list(range(100)),  # 0..99
        }
    ).with_columns(pl.col("factor").cast(pl.Float64))
    result = winsorize_percentile(df, "factor", lower=0.01, upper=0.99)
    vals = result["factor"]
    # original min=0, max=99; after 1%/99% clip: ~ [1, 98]
    assert vals.max() <= 99.0
    assert vals.min() >= 0.0
    # the extreme values should be clipped
    assert vals.max() < 99.0 or vals.min() > 0.0


def test_sigma_clip_removes_outliers():
    from daily.preprocessing.outlier import sigma_clip

    rng = np.random.default_rng(42)
    vals = [*list(rng.standard_normal(99)), 100.0]  # one huge outlier
    df = pl.DataFrame(
        {
            "trade_date": ["2024-01-01"] * 100,
            "ts_code": [f"00{i:04d}.SZ" for i in range(100)],
            "factor": vals,
        }
    )
    result = sigma_clip(df, "factor", n_sigma=3.0)
    # The 100.0 outlier is clipped to mean + 3*std.  With the outlier included
    # in the distribution the clipped max is ~31 — still well below 100.
    assert result["factor"].max() < 100.0  # outlier was reduced
    assert result["factor"].max() < 50.0  # meaningfully reduced from 100
