"""A 股 CostModel port —— 复用 daily.evaluation.backtest.CostModel（单一公式来源）。

A 股：佣金+滑点（买入），卖出再加印花税；持有成本为融券利息（仅空头，本 MVP long-only）。
core CostModel 惰性 import，保持模块导入轻量。
"""
from __future__ import annotations

from typing import Any

from factorzen.markets.base import CostModel


class AShareCostModel(CostModel):
    def __init__(self, core: Any = None) -> None:
        if core is None:
            from factorzen.daily.evaluation.backtest import CostModel as _Core

            core = _Core()
        self._core = core

    def trade_cost(self, side: str, notional: float, is_maker: bool = False) -> float:
        # A 股无 maker/taker 区分；卖出含印花税，买入不含。
        rate = self._core.sell_cost() if side == "sell" else self._core.one_way_cost()
        return abs(notional) * rate

    def carry_cost(
        self, position_value: float, periods: int, funding_rate: float = 0.0
    ) -> float:
        # A 股无 funding；long-only 无持有成本，空头计融券利息。
        if position_value >= 0:
            return 0.0
        return abs(position_value) * self._core.borrow_rate_per_period("daily") * periods
