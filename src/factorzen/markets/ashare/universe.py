"""A 股 Universe port —— 委托现有 core.universe + 指数基准。

在线（需 Tushare/缓存）：MC0 仅提供委托骨架，离线不测。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.markets.base import Universe


class AShareUniverse(Universe):
    def __init__(self, universe_name: str = "all_a", benchmark_index: str = "000300.SH") -> None:
        self.universe_name = universe_name
        self.benchmark_index = benchmark_index

    def snapshot(self, d: date | str) -> list[str]:
        from factorzen.core.universe import get_universe_snapshot

        date_str = d if isinstance(d, str) else d.strftime("%Y%m%d")
        snap = get_universe_snapshot(date_str, self.universe_name)
        # 剔除停牌，返回可交易标的
        if "is_suspended" in snap.columns:
            snap = snap.filter(~pl.col("is_suspended"))
        return snap["ts_code"].to_list()

    def benchmark(self, start: str, end: str) -> pl.DataFrame:
        from factorzen.core.loader import fetch_index_daily

        idx = fetch_index_daily(self.benchmark_index, start, end)
        return idx.select("trade_date", "close").sort("trade_date")
