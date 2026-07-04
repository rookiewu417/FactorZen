"""数据湖读写 roundtrip + 区间过滤 + 缺标的空帧。"""
from datetime import datetime

import polars as pl

from factorzen.markets.crypto.lake import CryptoLake, day_range, month_range


def _k(day: int) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [datetime(2026, 5, day, 0, 0), datetime(2026, 5, day, 0, 1)],
        "open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5],
        "close": [1.5, 2.5], "vol": [10.0, 20.0], "amount": [15.0, 50.0],
        "taker_buy_volume": [4.0, 9.0],
    }).with_columns(pl.col("trade_date").cast(pl.Datetime("us")))


def test_month_and_day_range():
    assert month_range("20250715", "20251003") == ["2025-07", "2025-08", "2025-09", "2025-10"]
    assert day_range("2026-05", "20260530", "20260602") == ["20260530", "20260531"]


def test_kline_roundtrip_and_filter(tmp_path):
    lake = CryptoLake(tmp_path)
    lake.write_klines("BTCUSDT", "2026-05", pl.concat([_k(1), _k(2)]))
    lake.write_klines("ETHUSDT", "2026-05", _k(1))
    out = lake.read_klines(["BTCUSDT"], "20260502", "20260502")
    assert out["ts_code"].unique().to_list() == ["BTCUSDT"]
    assert out.height == 2  # 只有 5/2 的两根
    assert lake.read_klines(["XRPUSDT"], "20260501", "20260502").is_empty()  # 缺标的→空帧
    assert sorted(lake.symbols()) == ["BTCUSDT", "ETHUSDT"]


def test_funding_meta_manifest_roundtrip(tmp_path):
    lake = CryptoLake(tmp_path)
    ev = pl.DataFrame({"event_time": [datetime(2026, 5, 1, 8)], "funding_rate": [0.0001]}
                      ).with_columns(pl.col("event_time").cast(pl.Datetime("us")))
    lake.write_funding("BTCUSDT", "2026-05", ev)
    got = lake.read_funding(["BTCUSDT"], "20260501", "20260501")
    assert got["funding_rate"].to_list() == [0.0001] and got["ts_code"][0] == "BTCUSDT"
    meta = pl.DataFrame({"ts_code": ["BTCUSDT"], "name": ["BTC"],
                         "list_date": [datetime(2020, 1, 1).date()]})
    lake.write_meta(meta)
    assert lake.read_meta()["ts_code"].to_list() == ["BTCUSDT"]
    lake.write_manifest({"gaps": []})
    assert lake.read_manifest() == {"gaps": []}
