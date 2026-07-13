"""国内商品期货交易约束：T+0 + 可做空 + 有涨跌停（MVP 近似）。

与 A 股不同：T+0（当日可平）、允许做空、无停牌次日封板；有涨跌停但各品种/时段幅度不一，
**MVP 用统一 ±7% 近似**（诚实标注：真实各交易所各品种阈值不同、临近交割会调整，此处不逐品种建模）。
撮合走 next-bar 收盘（同 crypto MVP 口径）。唯一不可交易情形是该 bar 无成交（``vol==0``）。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import TradingRules

# MVP 统一涨跌停近似（诚实标注：真实阈值逐品种/逐交易所不同，未逐一建模）
LIMIT_PCT_APPROX = 0.07


class FuturesTradingRules(TradingRules):
    @property
    def allow_short(self) -> bool:
        return True  # 期货天然双向

    @property
    def settlement_lag(self) -> int:
        return 0  # T+0

    @property
    def execution_price_col(self) -> str:
        return "close"

    def tradable_mask(self, bars: pl.DataFrame, side: str) -> pl.Series:
        # MVP：仅按有无成交判（vol>0）。涨跌停不对称约束留二期，此处诚实标注不建模。
        return (bars["vol"] > 0).rename("tradable")
