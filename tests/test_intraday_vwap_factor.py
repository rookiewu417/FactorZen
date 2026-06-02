"""tests/test_intraday_vwap_factor.py"""

import datetime
import unittest.mock as mock

import polars as pl

from factorzen.intraday.data.context import IntradayDataContext
from workspace.factors.intraday.vwap_deviation import VwapDeviation


def _make_ctx(df: pl.DataFrame):
    ctx = mock.MagicMock(spec=IntradayDataContext)
    ctx.minute = df.lazy()
    return ctx


def _make_minute_df(n_bars: int = 20) -> pl.DataFrame:
    base = datetime.datetime(2026, 5, 16, 9, 30)
    rows = []
    for ts in ["000001.SZ", "000002.SZ"]:
        for i in range(n_bars):
            price = 10.0 + i * 0.05
            vol = 1000.0 + i * 10
            rows.append(
                {
                    "ts_code": ts,
                    "trade_time": base + datetime.timedelta(minutes=i),
                    "close": price,
                    "vol": vol,
                    "amount": price * vol,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_time").cast(pl.Datetime))


def test_columns():
    factor = VwapDeviation()
    result = factor.compute(_make_ctx(_make_minute_df()))
    assert set(["trade_time", "ts_code", "factor_value"]).issubset(result.columns)


def test_no_cross_stock():
    factor = VwapDeviation()
    result = factor.compute(_make_ctx(_make_minute_df()))
    assert set(result["ts_code"].unique().to_list()) == {"000001.SZ", "000002.SZ"}


def test_first_bar_zero():
    """第一根 bar 时 VWAP == close，偏离为 0。"""
    factor = VwapDeviation()
    result = factor.compute(_make_ctx(_make_minute_df()))
    first = (
        result.filter(pl.col("ts_code") == "000001.SZ")
        .sort("trade_time")
        .head(1)["factor_value"][0]
    )
    assert abs(first) < 1e-9


def test_registered():
    from factorzen.intraday.factors.registry import get_factor

    assert get_factor("vwap_deviation") is VwapDeviation
