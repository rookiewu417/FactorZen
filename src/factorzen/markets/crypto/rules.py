"""crypto perps 交易约束：近乎空约束 + T+0 + 可做空。

与 A 股不同：无涨跌停、无停牌次日封板、无 T+1。唯一不可交易情形是该 bar
无成交（``vol==0``，退市/极端缺流动性）。撮合走 next-bar 收盘。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import TradingRules


class CryptoTradingRules(TradingRules):
    @property
    def allow_short(self) -> bool:
        return True

    @property
    def settlement_lag(self) -> int:
        return 0  # T+0

    @property
    def execution_price_col(self) -> str:
        return "close"

    def tradable_mask(self, bars: pl.DataFrame, side: str) -> pl.Series:
        # 买卖对称：仅 vol>0 可交易（无涨跌停不对称）
        return (bars["vol"] > 0).rename("tradable")
