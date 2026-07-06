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
        # 本 adapter 仅经 core.loader.fetch_daily 取日频；非 daily freq 显式报错，
        # 不静默返回日频数据（否则 weekly/monthly/intraday 请求会拿到错频数据而不自知）。
        if freq != "daily":
            raise ValueError(
                f"AShareDataProvider 仅支持 freq='daily'（经 Tushare fetch_daily），"
                f"收到 freq={freq!r}"
            )
        from factorzen.core.loader import fetch_daily

        return fetch_daily(start, end, ts_codes=symbols)

    def fetch_symbol_meta(self) -> pl.DataFrame:
        from factorzen.core.loader import fetch_stock_basic

        return fetch_stock_basic()
