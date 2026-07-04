"""MC0 Task 3: crypto CCXT DataProvider（离线 fake，无网络）。"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from factorzen.markets.base import DataProvider
from factorzen.markets.crypto.provider import CryptoDataProvider


def _ms(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


class FakeCCXT:
    """模拟 ccxt binanceusdm 的最小子集（结构对齐官方 unified API）。"""

    def __init__(self):
        # unified symbol -> 日线 [ms, o,h,l,c,v]
        self._ohlcv = {
            "BTC/USDT:USDT": [
                [_ms(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 10.0],
                [_ms(2024, 1, 2), 105.0, 120.0, 104.0, 118.0, 12.0],
                [_ms(2024, 1, 3), 118.0, 119.0, 108.0, 110.0, 8.0],
            ],
            "ETH/USDT:USDT": [
                [_ms(2024, 1, 1), 50.0, 55.0, 48.0, 52.0, 20.0],
                [_ms(2024, 1, 2), 52.0, 53.0, 49.0, 50.0, 22.0],
            ],
        }
        # unified symbol -> funding 事件（每日 3 次，8h 一档）
        self._funding = {
            "BTC/USDT:USDT": [
                {"timestamp": _ms(2024, 1, 1), "fundingRate": 0.0001},
                {"timestamp": _ms(2024, 1, 1) + 8 * 3600_000, "fundingRate": 0.0002},
                {"timestamp": _ms(2024, 1, 1) + 16 * 3600_000, "fundingRate": -0.0001},
                {"timestamp": _ms(2024, 1, 2), "fundingRate": 0.0003},
            ],
        }
        self._oi = {
            "BTC/USDT:USDT": [
                {"timestamp": _ms(2024, 1, 1), "openInterestAmount": 1000.0},
                {"timestamp": _ms(2024, 1, 2), "openInterestAmount": 1100.0},
            ],
        }

    def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=1000):
        data = self._ohlcv.get(symbol, [])
        if since is not None:
            data = [r for r in data if r[0] >= since]
        return data[:limit]

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        data = self._funding.get(symbol, [])
        if since is not None:
            data = [r for r in data if r["timestamp"] >= since]
        return data[:limit]

    def fetch_open_interest_history(self, symbol, timeframe="1d", since=None, limit=1000):
        data = self._oi.get(symbol, [])
        if since is not None:
            data = [r for r in data if r["timestamp"] >= since]
        return data[:limit]

    def load_markets(self):
        return {
            "BTC/USDT:USDT": {
                "base": "BTC", "quote": "USDT", "swap": True,
                "info": {"onboardDate": str(_ms(2019, 9, 8))},
            },
            "ETH/USDT:USDT": {
                "base": "ETH", "quote": "USDT", "swap": True, "info": {},
            },
            "BTC/USDT": {"base": "BTC", "quote": "USDT", "swap": False, "info": {}},  # 现货，应剔除
        }


def _provider():
    return CryptoDataProvider(client=FakeCCXT())


def test_is_a_dataprovider():
    assert isinstance(_provider(), DataProvider)


def test_symbol_mapping_roundtrip():
    p = _provider()
    assert p._to_unified("BTCUSDT") == "BTC/USDT:USDT"
    assert p._to_ts_code("BTC/USDT:USDT") == "BTCUSDT"


def test_fetch_bars_schema_and_values():
    p = _provider()
    df = p.fetch_bars(["BTCUSDT", "ETHUSDT"], "20240101", "20240103")
    assert set(df.columns) >= {
        "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    }
    # BTC 3 天 + ETH 2 天 = 5 行
    assert df.height == 5
    btc = df.filter(pl.col("ts_code") == "BTCUSDT").sort("trade_date")
    # amount = close * vol
    assert btc["amount"][0] == 105.0 * 10.0
    assert str(btc["trade_date"][0]) == "2024-01-01"


def test_fetch_bars_respects_end_date():
    p = _provider()
    df = p.fetch_bars(["BTCUSDT"], "20240101", "20240102")  # 只到 1/2
    assert df.height == 2


def test_fetch_funding_daily_sum():
    """日频 funding = 当日多档 funding 之和（Binance 每 8h 一档）。"""
    p = _provider()
    fd = p.fetch_funding(["BTCUSDT"], "20240101", "20240103")
    assert set(fd.columns) >= {"ts_code", "trade_date", "funding_rate"}
    d1 = fd.filter(pl.col("trade_date") == pl.date(2024, 1, 1))
    # 0.0001 + 0.0002 - 0.0001 = 0.0002
    assert abs(d1["funding_rate"][0] - 0.0002) < 1e-12


def test_fetch_open_interest():
    p = _provider()
    oi = p.fetch_open_interest(["BTCUSDT"], "20240101", "20240103")
    assert set(oi.columns) >= {"ts_code", "trade_date", "open_interest"}
    assert oi.height == 2


def test_fetch_symbol_meta_only_swap_quote():
    p = _provider()
    meta = p.fetch_symbol_meta()
    codes = set(meta["ts_code"].to_list())
    assert "BTCUSDT" in codes and "ETHUSDT" in codes
    # 现货 BTC/USDT(swap=False) 应被剔除 —— 只有 2 个永续
    assert meta.height == 2
