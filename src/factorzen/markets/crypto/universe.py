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
        meta = self.provider.fetch_symbol_meta()
        symbols = meta["ts_code"].to_list()
        if not symbols:
            return []
        bars = self.provider.fetch_bars(
            symbols, window_start.strftime("%Y%m%d"), d.strftime("%Y%m%d")
        )
        if bars.is_empty():
            return []
        vol = bars.group_by("ts_code").agg(pl.col("amount").sum().alias("tot_amount"))
        vol = vol.join(meta.select("ts_code", "list_date"), on="ts_code", how="left")
        keep = (
            vol.filter(
                # 零成交额恒不可交易,与阈值无关:合约下架/迁移后 Vision 仍生成价格
                # 冻结的 bar(实测 FTMUSDT 2025-01-06 后、MKRUSDT 2025-09-08 后),
                # 若只判 `>= min_amount(默认 0.0)` 它们会静默进池污染截面。
                (pl.col("tot_amount") > 0)
                & (pl.col("tot_amount") >= self.min_amount)
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
