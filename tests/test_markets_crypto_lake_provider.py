"""湖 provider:读湖+重采样+freq 分派;mini-lake fixture 供全链路测试复用。"""
from datetime import date, datetime

import polars as pl
import pytest

from factorzen.markets.crypto.lake import CryptoLake
from factorzen.markets.crypto.lake_provider import CryptoLakeProvider


def make_mini_lake(root, symbols=("BTCUSDT", "ETHUSDT"), days=(1, 2)) -> CryptoLake:
    """2 标的 × N 日、每日 00:00-01:59 共 120 根 1m bar 的最小湖。"""
    lake = CryptoLake(root)
    for si, sym in enumerate(symbols):
        frames = []
        for d in days:
            ts = [datetime(2026, 5, d, h, m) for h in (0, 1) for m in range(60)]
            base = 100.0 * (si + 1)
            px = [base + i * 0.1 for i in range(len(ts))]
            frames.append(pl.DataFrame({
                "trade_date": ts, "open": px, "high": [p + 0.5 for p in px],
                "low": [p - 0.5 for p in px], "close": [p + 0.2 for p in px],
                "vol": [1.0] * len(ts), "amount": [p * 1.0 for p in px],
                "taker_buy_volume": [0.6] * len(ts),
            }).with_columns(pl.col("trade_date").cast(pl.Datetime("us"))))
        lake.write_klines(sym, "2026-05", pl.concat(frames))
        lake.write_funding(sym, "2026-05", pl.DataFrame({
            "event_time": [datetime(2026, 5, d, 0, 0) for d in days],
            "funding_rate": [0.0001] * len(days),
        }).with_columns(pl.col("event_time").cast(pl.Datetime("us"))))
        for d in days:
            lake.write_metrics(sym, f"202605{d:02d}", pl.DataFrame({
                "event_time": [datetime(2026, 5, d, 0, 5)], "open_interest": [1000.0 + d],
            }).with_columns(pl.col("event_time").cast(pl.Datetime("us"))))
    lake.write_meta(pl.DataFrame({
        "ts_code": list(symbols), "name": [s[:-4] for s in symbols],
        "list_date": [date(2024, 1, 1)] * len(symbols)}))
    return lake


def test_fetch_bars_daily_date_key(tmp_path):
    make_mini_lake(tmp_path)
    p = CryptoLakeProvider(lake_root=tmp_path)
    bars = p.fetch_bars(["BTCUSDT"], "20260501", "20260502", "daily")
    assert bars.schema["trade_date"] == pl.Date
    assert bars.height == 2 and bars["vol"].to_list() == [120.0, 120.0]


def test_fetch_bars_15m_datetime_key(tmp_path):
    make_mini_lake(tmp_path)
    p = CryptoLakeProvider(lake_root=tmp_path)
    bars = p.fetch_bars(["BTCUSDT", "ETHUSDT"], "20260501", "20260501", "15m")
    assert bars.schema["trade_date"] == pl.Datetime("us")
    assert bars.filter(pl.col("ts_code") == "BTCUSDT").height == 8  # 2h → 8 根 15m
    assert bars.filter(pl.col("ts_code") == "BTCUSDT")["vol"].to_list() == [15.0] * 8


def test_fetch_funding_and_oi_freq(tmp_path):
    make_mini_lake(tmp_path)
    p = CryptoLakeProvider(lake_root=tmp_path)
    fd = p.fetch_funding(["BTCUSDT"], "20260501", "20260502", "daily")
    assert fd.schema["trade_date"] == pl.Date and fd.height == 2
    f15 = p.fetch_funding(["BTCUSDT"], "20260501", "20260501", "15m")
    assert f15["trade_date"].to_list() == [datetime(2026, 5, 1, 0, 0)]
    oi = p.fetch_open_interest(["BTCUSDT"], "20260501", "20260501", "15m")
    assert oi["open_interest"].to_list() == [1001.0]


def test_empty_lake_raises(tmp_path):
    p = CryptoLakeProvider(lake_root=tmp_path / "nope")
    with pytest.raises(RuntimeError, match="backfill"):
        p.fetch_bars(["BTCUSDT"], "20260501", "20260502", "daily")


def test_meta_roundtrip(tmp_path):
    make_mini_lake(tmp_path)
    meta = CryptoLakeProvider(lake_root=tmp_path).fetch_symbol_meta()
    assert set(meta["ts_code"].to_list()) == {"BTCUSDT", "ETHUSDT"}
