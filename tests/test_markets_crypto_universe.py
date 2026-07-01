"""MC0 Task 5: crypto Universe（成交额 Top-N + 流动性/新币过滤）。"""
from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.markets.base import DataProvider, Universe
from factorzen.markets.crypto.universe import CryptoUniverse


class _FakeProvider(DataProvider):
    """返回受控 fixture bars + meta 的假 provider。"""

    def __init__(self, bars: pl.DataFrame, meta: pl.DataFrame):
        self._bars = bars
        self._meta = meta

    def fetch_bars(self, symbols, start, end, freq="daily"):
        df = self._bars
        s = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
        e = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
        df = df.filter((pl.col("trade_date") >= s) & (pl.col("trade_date") <= e))
        if symbols is not None:
            df = df.filter(pl.col("ts_code").is_in(symbols))
        return df

    def fetch_symbol_meta(self):
        return self._meta


def _fixture() -> _FakeProvider:
    # 窗口 [2024-01-11, 2024-02-10]，d=2024-02-10，lookback=30，min_list_days=30
    rows = []
    def add(code, d, amount):
        rows.append({"ts_code": code, "trade_date": d, "close": 100.0,
                     "vol": amount / 100.0, "amount": amount})
    # BTC/ETH：老币、窗口内高成交额
    for dd in [date(2024, 1, 11), date(2024, 2, 1), date(2024, 2, 10)]:
        add("BTCUSDT", dd, 5000.0)
        add("ETHUSDT", dd, 3000.0)
        add("LOWUSDT", dd, 10.0)  # 老币但成交额极低 → min_amount 剔除
    # NEW：新币（2024-02-05 才上市），成交额极高 → 应被 age 过滤剔除
    for dd in [date(2024, 2, 5), date(2024, 2, 10)]:
        add("NEWUSDT", dd, 99999.0)
    bars = pl.DataFrame(rows)
    meta = pl.DataFrame({
        "ts_code": ["BTCUSDT", "ETHUSDT", "LOWUSDT", "NEWUSDT"],
        "name": ["BTC", "ETH", "LOW", "NEW"],
        "list_date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1), date(2024, 2, 5)],
    })
    return _FakeProvider(bars, meta)


def test_is_a_universe():
    u = CryptoUniverse(provider=_fixture())
    assert isinstance(u, Universe)


def test_snapshot_topn_liquidity_age_filter():
    u = CryptoUniverse(provider=_fixture(), top_n=3, lookback_days=30,
                       min_amount=100.0, min_list_days=30)
    snap = u.snapshot("20240210")
    # NEW 被 age 剔除，LOW 被 min_amount 剔除 → 剩 BTC/ETH，按成交额降序
    assert snap == ["BTCUSDT", "ETHUSDT"]


def test_snapshot_topn_caps_count():
    u = CryptoUniverse(provider=_fixture(), top_n=1, lookback_days=30,
                       min_amount=0.0, min_list_days=30)
    snap = u.snapshot("20240210")
    assert snap == ["BTCUSDT"]  # 成交额第一


def test_benchmark_returns_btc_close_series():
    u = CryptoUniverse(provider=_fixture(), benchmark_symbol="BTCUSDT")
    bench = u.benchmark("20240101", "20240210")
    assert set(bench.columns) >= {"trade_date", "close"}
    assert bench.height == 3  # BTC 3 根
