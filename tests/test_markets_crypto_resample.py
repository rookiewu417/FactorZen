"""重采样/对齐 ground truth 单测:全部手算期望值,不用被测函数自导自演。"""
from datetime import date, datetime

import polars as pl

from factorzen.markets.crypto.resample import align_funding, align_open_interest, resample_bars


def _bars_1m() -> pl.DataFrame:
    # BTCUSDT 4 根 1m bar:00:00/00:01 属 15m bar0,00:15/00:16 属 15m bar1
    return pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 4,
        "trade_date": [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 0, 1),
                       datetime(2026, 5, 1, 0, 15), datetime(2026, 5, 1, 0, 16)],
        "open":  [100.0, 101.0, 103.0, 102.0],
        "high":  [102.0, 104.0, 103.5, 105.0],
        "low":   [ 99.0, 100.5, 101.0, 101.5],
        "close": [101.0, 103.0, 102.0, 104.0],
        "vol":   [10.0, 20.0, 5.0, 15.0],
        "amount": [1000.0, 2000.0, 500.0, 1500.0],
        "taker_buy_volume": [6.0, 8.0, 2.0, 9.0],
    }).with_columns(pl.col("trade_date").cast(pl.Datetime("us")))


def test_resample_15m_ground_truth():
    out = resample_bars(_bars_1m(), "15m")
    assert out["trade_date"].to_list() == [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 0, 15)]
    # bar0: open=首根 open,close=末根 close,high=max,low=min,量额=sum
    assert out["open"].to_list() == [100.0, 103.0]
    assert out["close"].to_list() == [103.0, 104.0]
    assert out["high"].to_list() == [104.0, 105.0]
    assert out["low"].to_list() == [99.0, 101.0]
    assert out["vol"].to_list() == [30.0, 20.0]
    assert out["amount"].to_list() == [3000.0, 2000.0]
    assert out["taker_buy_volume"].to_list() == [14.0, 11.0]


def test_resample_daily_casts_date():
    out = resample_bars(_bars_1m(), "daily")
    assert out.schema["trade_date"] == pl.Date
    assert out["trade_date"].to_list() == [date(2026, 5, 1)]
    assert out["open"].to_list() == [100.0]
    assert out["close"].to_list() == [104.0]
    assert out["vol"].to_list() == [50.0]


def test_resample_empty_passthrough():
    assert resample_bars(_bars_1m().head(0), "1h").is_empty()


def _funding_events() -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 3,
        "event_time": [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 8, 0),
                       datetime(2026, 5, 1, 16, 0)],
        "funding_rate": [0.0001, 0.0002, 0.0003],
    }).with_columns(pl.col("event_time").cast(pl.Datetime("us")))


def test_align_funding_daily_sums_three_legs():
    out = align_funding(_funding_events(), "daily")
    assert out.schema["trade_date"] == pl.Date
    assert out["trade_date"].to_list() == [date(2026, 5, 1)]
    assert abs(out["funding_rate"][0] - 0.0006) < 1e-12  # 现日频行为:三档和


def test_align_funding_1h_lands_on_settlement_bars():
    out = align_funding(_funding_events(), "1h").sort("trade_date")
    assert out["trade_date"].to_list() == [
        datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 8, 0), datetime(2026, 5, 1, 16, 0)]
    assert out["funding_rate"].to_list() == [0.0001, 0.0002, 0.0003]


def test_align_open_interest_last_in_bar():
    metrics = pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 3,
        "event_time": [datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 0, 5),
                       datetime(2026, 5, 1, 0, 20)],
        "open_interest": [10.0, 20.0, 30.0],
    }).with_columns(pl.col("event_time").cast(pl.Datetime("us")))
    out15 = align_open_interest(metrics, "15m").sort("trade_date")
    assert out15["open_interest"].to_list() == [20.0, 30.0]  # bar 内最后一笔
    outd = align_open_interest(metrics, "daily")
    assert outd["open_interest"].to_list() == [30.0]  # 当日最后值
