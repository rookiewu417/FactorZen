"""tests/test_intraday_returns.py"""

import datetime

import polars as pl
import pytest

from factorzen.intraday.evaluation.returns import compute_intraday_fwd_returns


def _make_minute_df() -> pl.DataFrame:
    """3 只股票，每只 10 根 bar。"""
    rows = []
    base_time = datetime.datetime(2026, 5, 16, 9, 30, 0)
    for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        for i in range(10):
            rows.append(
                {
                    "ts_code": ts,
                    "trade_time": base_time + datetime.timedelta(minutes=i),
                    "close": 10.0 + i * 0.1,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_time").cast(pl.Datetime))


def test_fwd_returns_columns():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1, 5])
    assert "fwd_ret_1bar" in df.columns
    assert "fwd_ret_5bar" in df.columns


def test_fwd_ret_1bar_last_row_is_null():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    last_rows = df.filter(pl.col("ts_code") == "000001.SZ").sort("trade_time").tail(1)
    assert last_rows["fwd_ret_1bar"][0] is None


def test_fwd_ret_1bar_value():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    row = df.filter((pl.col("ts_code") == "000001.SZ") & (pl.col("trade_time").dt.minute() == 30))
    expected = (10.1 - 10.0) / 10.0
    assert abs(row["fwd_ret_1bar"][0] - expected) < 1e-9


def test_no_cross_stock_leakage():
    df = compute_intraday_fwd_returns(_make_minute_df(), periods=[1])
    for ts in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        last_val = df.filter(pl.col("ts_code") == ts).sort("trade_time").tail(1)["fwd_ret_1bar"][0]
        assert last_val is None, f"{ts} 最后一行应为 null，实为 {last_val}"


def test_fwd_return_does_not_cross_trading_day_boundary():
    df = pl.DataFrame(
        {
            "trade_time": [
                datetime.datetime(2024, 1, 2, 14, 59),
                datetime.datetime(2024, 1, 2, 15, 0),
                datetime.datetime(2024, 1, 3, 9, 30),
                datetime.datetime(2024, 1, 3, 9, 31),
            ],
            "ts_code": ["000001.SZ"] * 4,
            "close": [100.0, 101.0, 200.0, 202.0],
        }
    )

    out = compute_intraday_fwd_returns(df, periods=[1])

    values = out["fwd_ret_1bar"].to_list()
    assert values[0] == pytest.approx(0.01)
    assert values[1] is None
    assert values[2] == pytest.approx(0.01)
    assert values[3] is None
