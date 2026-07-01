"""crypto 标的池：近 N 日成交额 Top-K + 流动性/新币过滤。

- 成交额 = bars.amount（quote 计价成交额代理，见 provider）。
- 新币过滤：上市不足 ``min_list_days`` 天（按 symbol_meta.list_date）剔除。
- 流动性过滤：窗口内总成交额 < ``min_amount`` 剔除。
- 基准：默认 BTCUSDT 收盘（MC3 可换市值加权指数）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from factorzen.markets.base import DataProvider, Universe


def _to_date(d: date | str) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y%m%d").date()
    return d


class CryptoUniverse(Universe):
    def __init__(
        self,
        provider: DataProvider,
        top_n: int = 50,
        lookback_days: int = 30,
        min_amount: float = 0.0,
        min_list_days: int = 30,
        benchmark_symbol: str = "BTCUSDT",
    ) -> None:
        self.provider = provider
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.min_amount = min_amount
        self.min_list_days = min_list_days
        self.benchmark_symbol = benchmark_symbol

    def snapshot(self, d: date | str) -> list[str]:
        d = _to_date(d)
        window_start = d - timedelta(days=self.lookback_days)
        bars = self.provider.fetch_bars(
            None, window_start.strftime("%Y%m%d"), d.strftime("%Y%m%d")
        )
        if bars.is_empty():
            return []
        vol = bars.group_by("ts_code").agg(pl.col("amount").sum().alias("tot_amount"))
        meta = self.provider.fetch_symbol_meta().select("ts_code", "list_date")
        vol = vol.join(meta, on="ts_code", how="left")
        keep = (
            vol.filter(
                (pl.col("tot_amount") >= self.min_amount)
                & pl.col("list_date").is_not_null()
                & ((pl.lit(d) - pl.col("list_date")).dt.total_days() >= self.min_list_days)
            )
            .sort("tot_amount", descending=True)
            .head(self.top_n)
        )
        return keep["ts_code"].to_list()

    def benchmark(self, start: str, end: str) -> pl.DataFrame:
        bars = self.provider.fetch_bars([self.benchmark_symbol], start, end)
        return bars.select("trade_date", "close").sort("trade_date")
