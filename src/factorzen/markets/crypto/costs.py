"""crypto perps 成本模型：maker/taker + 滑点 + funding carry。

与 A 股不同：**无印花税、买卖对称**（A 股卖出单边印花税）。持有成本来自
永续资金费（funding rate）而非融券利息——多头付正 funding、空头收。
"""
from __future__ import annotations

from factorzen.markets.base import CostModel


class CryptoCostModel(CostModel):
    def __init__(
        self,
        maker: float = 0.0002,
        taker: float = 0.0005,
        slippage: float = 0.0005,
    ) -> None:
        self.maker = maker
        self.taker = taker
        self.slippage = slippage

    def trade_cost(self, side: str, notional: float, is_maker: bool = False) -> float:
        fee = self.maker if is_maker else self.taker
        return abs(notional) * (fee + self.slippage)

    def carry_cost(
        self, position_value: float, periods: int, funding_rate: float = 0.0
    ) -> float:
        # position_value 带符号（多正空负）。funding>0 时多头付费（正成本）、空头收（负）。
        return position_value * funding_rate * periods
