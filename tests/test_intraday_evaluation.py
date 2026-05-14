"""Tests for intraday/evaluation/ic_analysis.py."""

from datetime import datetime, date

import numpy as np
import polars as pl
import pytest

from intraday.evaluation.ic_analysis import (
    IntradayICResult,
    compute_intraday_rank_ic,
    _assign_segment,
)


def _make_intraday_data(n_stocks: int = 30, n_days: int = 5, seed: int = 0):
    """Generates synthetic minute-bar factor + return DataFrames."""
    rng = np.random.default_rng(seed)
    from datetime import timedelta
    base_times = [
        datetime(2024, 1, 2) + timedelta(days=d, hours=9, minutes=30 + m)
        for d in range(n_days)
        for m in range(0, 90, 5)  # every 5 min, 09:30–11:00 (18 bars/day)
    ]
    minutes_per_day = base_times

    factor_rows = []
    ret_rows = []
    for ts in minutes_per_day:
        for i in range(n_stocks):
            code = f"{i:06d}.SH"
            factor_rows.append({
                "trade_time": ts,
                "ts_code": code,
                "factor_value": float(rng.standard_normal()),
            })
            ret_rows.append({
                "trade_time": ts,
                "ts_code": code,
                "fwd_ret_1bar": float(rng.standard_normal() * 0.002),
            })

    return pl.DataFrame(factor_rows), pl.DataFrame(ret_rows)


@pytest.fixture()
def intraday_data():
    return _make_intraday_data()


def test_returns_result_object(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert isinstance(result, IntradayICResult)


def test_ic_is_finite(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert np.isfinite(result.ic_mean)
    assert np.isfinite(result.ic_std)


def test_daily_ic_has_correct_dates(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert not result.daily_ic.is_empty()
    # 5 trading days -> 5 rows in daily_ic
    assert result.daily_ic.shape[0] == 5


def test_segment_ic_has_three_segments(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    segments = set(result.segment_ic["segment"].to_list())
    # Only open bars (09:30-10:00) and midday in our synthetic data
    assert len(segments) >= 1


def test_n_periods_positive(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert result.n_periods > 0


def test_summary_string(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    text = result.summary()
    assert "Intraday IC" in text
    assert "IC Mean" in text


def test_empty_input_returns_zeros():
    factor_df = pl.DataFrame({
        "trade_time": pl.Series([], dtype=pl.Datetime),
        "ts_code": pl.Series([], dtype=pl.Utf8),
        "factor_value": pl.Series([], dtype=pl.Float64),
    })
    ret_df = pl.DataFrame({
        "trade_time": pl.Series([], dtype=pl.Datetime),
        "ts_code": pl.Series([], dtype=pl.Utf8),
        "fwd_ret_1bar": pl.Series([], dtype=pl.Float64),
    })
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert result.ic_mean == 0.0
    assert result.n_periods == 0


def test_assign_segment_labels():
    df = pl.DataFrame({
        "trade_time": [
            datetime(2024, 1, 2, 9, 30),   # open
            datetime(2024, 1, 2, 11, 0),   # midday
            datetime(2024, 1, 2, 14, 45),  # close
        ]
    })
    result = _assign_segment(df, "trade_time")
    segs = result["segment"].to_list()
    assert segs[0] == "open"
    assert segs[1] == "midday"
    assert segs[2] == "close"
