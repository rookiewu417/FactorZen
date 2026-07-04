"""湖 Provider:读本地数据湖 + 重采样,实现 DataProvider Port(纯离线,不联网)。

与 CCXT provider 的关系:湖是默认数据源(REST 当前 451 不可达且慢);ccxt 类
保留作可选补数。fetch_funding/fetch_open_interest 为 crypto 扩展方法,带 freq。
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from factorzen.markets.base import DataProvider
from factorzen.markets.crypto.frequency import normalize_freq
from factorzen.markets.crypto.lake import CryptoLake
from factorzen.markets.crypto.resample import (
    align_funding,
    align_open_interest,
    resample_bars,
)


class CryptoLakeProvider(DataProvider):
    def __init__(self, lake: CryptoLake | None = None,
                 lake_root: str | Path = "workspace/crypto_lake") -> None:
        self.lake = lake or CryptoLake(lake_root)

    def _require_lake(self) -> None:
        if not (self.lake.root / "klines_1m").is_dir():
            raise RuntimeError(
                f"crypto 数据湖为空({self.lake.root}):先运行 fz data crypto backfill")

    def fetch_bars(self, symbols: list[str] | None, start: str, end: str,
                   freq: str = "daily") -> pl.DataFrame:
        f = normalize_freq(freq)
        self._require_lake()
        bars = self.lake.read_klines(symbols, start, end)
        if bars.is_empty():
            key = pl.Date() if f == "daily" else pl.Datetime("us")
            return pl.DataFrame(schema={
                "ts_code": pl.String, "trade_date": key, "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                "vol": pl.Float64, "amount": pl.Float64, "taker_buy_volume": pl.Float64})
        return resample_bars(bars, f)

    def fetch_funding(self, symbols: list[str] | None, start: str, end: str,
                      freq: str = "daily") -> pl.DataFrame:
        self._require_lake()
        return align_funding(self.lake.read_funding(symbols, start, end), freq)

    def fetch_open_interest(self, symbols: list[str] | None, start: str, end: str,
                            freq: str = "daily") -> pl.DataFrame:
        self._require_lake()
        return align_open_interest(self.lake.read_metrics(symbols, start, end), freq)

    def fetch_symbol_meta(self) -> pl.DataFrame:
        return self.lake.read_meta()
