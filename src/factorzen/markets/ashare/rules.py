"""A 股 TradingRules port —— long-only + T+1 + t+1 开盘撮合。

注：完整涨跌停/停牌约束仍在 daily/evaluation/backtest.py 的慢/快双路径中；
本 port 暴露元数据 + 停牌掩码。MC1 引擎 rewire 时统一由此 port 承载，届时把
backtest.py 的板块涨跌停逻辑迁移进来（避免双路径漂移）。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import TradingRules


class AShareTradingRules(TradingRules):
    @property
    def allow_short(self) -> bool:
        return False  # A 股 long-only

    @property
    def settlement_lag(self) -> int:
        return 1  # T+1

    @property
    def execution_price_col(self) -> str:
        return "open"  # t+1 开盘撮合

    def tradable_mask(self, bars: pl.DataFrame, side: str) -> pl.Series:
        # MVP：停牌(vol==0)不可交易。完整涨跌停约束见 backtest.py（MC1 统一迁入）。
        return (bars["vol"] > 0).rename("tradable")
