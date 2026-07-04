"""A 股 DataProvider port —— 委托现有 core.loader（Tushare）。

Tushare 相关惰性 import，保持 ``import factorzen.markets`` 轻量、离线可导入。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import DataProvider


class AShareDataProvider(DataProvider):
    def fetch_bars(
        self, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
    ) -> pl.DataFrame:
        from factorzen.core.loader import fetch_daily

        return fetch_daily(start, end, ts_codes=symbols)

    def fetch_symbol_meta(self) -> pl.DataFrame:
        from factorzen.core.loader import fetch_stock_basic

        return fetch_stock_basic()
