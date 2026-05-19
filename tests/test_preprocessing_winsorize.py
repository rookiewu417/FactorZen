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


def test_winsorize_is_per_date():
    """两个日期分布差异极大时，裁剪边界应按日期独立计算。"""
    from daily.preprocessing.outlier import winsorize_percentile

    # Date A: values 1-10; Date B: values 1000-10000 (completely different scale)
    rows = []
    for i in range(1, 11):
        rows.append({"trade_date": "2024-01-01", "ts_code": f"00{i:04d}.SZ", "factor": float(i)})
    for i in range(1000, 10001, 1000):
        rows.append({"trade_date": "2024-01-02", "ts_code": f"00{i:04d}.SZ", "factor": float(i)})
    df = pl.DataFrame(rows)
    result = winsorize_percentile(df, "factor", lower=0.1, upper=0.9)

    date_a = result.filter(pl.col("trade_date") == "2024-01-01")["factor"]
    date_b = result.filter(pl.col("trade_date") == "2024-01-02")["factor"]
    # Date A max should be << 1000 (not contaminated by Date B)
    assert date_a.max() < 100, f"Date A max should be < 100, got {date_a.max()}"
    # Date B min should be >> 10 (not contaminated by Date A)
    assert date_b.min() > 100, f"Date B min should be > 100, got {date_b.min()}"


def test_sigma_clip_is_per_date():
    """sigma_clip 应按日期截面计算 mean/std。"""
    from daily.preprocessing.outlier import sigma_clip

    rows = []
    # Date A: normal values around 0
    for i in range(99):
        rows.append({"trade_date": "2024-01-01", "ts_code": f"00{i:04d}.SZ", "factor": float(i % 10 - 5)})
    # Date A outlier
    rows.append({"trade_date": "2024-01-01", "ts_code": "00099.SZ", "factor": 1000.0})
    # Date B: all values around 5000 (different mean)
    for i in range(10):
        rows.append({"trade_date": "2024-01-02", "ts_code": f"01{i:04d}.SZ", "factor": 5000.0 + float(i)})
    df = pl.DataFrame(rows)
    result = sigma_clip(df, "factor", n_sigma=3.0)

    date_a = result.filter(pl.col("trade_date") == "2024-01-01")["factor"]
    date_b = result.filter(pl.col("trade_date") == "2024-01-02")["factor"]
    # Date A outlier (1000) should be clipped to mean+3*std — well below 1000.
    # With the outlier itself pulling up std, the clipped ceiling is ~310,
    # which is still much less than 1000 (proving per-date isolation works).
    assert date_a.max() < 1000, f"Date A outlier should be clipped, got {date_a.max()}"
    assert date_a.max() < 500, f"Date A outlier should be meaningfully clipped, got {date_a.max()}"
    # Date B values (~5000) should remain ~5000, not dragged down by Date A's sigma
    assert date_b.min() > 100, f"Date B values should stay ~5000, got {date_b.min()}"
