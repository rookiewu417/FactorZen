"""美股交易约束：无涨跌停 + T+0 交易 + 可做空。

与 A 股不同：无涨跌停、无停牌次日封板、可当日买卖（T+0 日内交易）、可做空（借券）。
现金结算为 T+1，但**不影响信号时点**（当日可开平仓、次日交割），故 ``settlement_lag=0``
（信号→次 bar 收盘撮合，同 crypto/futures MVP 口径）。唯一不可交易情形是该 bar 无成交
（``vol==0``，退市/极端缺流动性）。做空的借券成本在 costs.py 建模。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import TradingRules


class USTradingRules(TradingRules):
    @property
    def allow_short(self) -> bool:
        return True  # 美股可借券做空

    @property
    def settlement_lag(self) -> int:
        # T+0 交易（现金 T+1 交割不影响信号时点，可日内开平），撮合口径同 crypto/futures
        return 0

    @property
    def execution_price_col(self) -> str:
        return "close"

    def tradable_mask(self, bars: pl.DataFrame, side: str) -> pl.Series:
        # 无涨跌停、买卖对称：仅 vol>0 可交易
        return (bars["vol"] > 0).rename("tradable")
