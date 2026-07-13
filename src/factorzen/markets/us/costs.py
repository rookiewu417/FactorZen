"""美股成本模型：佣金零 + 滑点常数 bp + 做空借券成本（MVP 常数）。

与 A 股不同：**无印花税、佣金零**（主流券商零佣金）、买卖对称。持有成本仅来自
**做空的借券费**（借券成本 = 空头名义 × 常数年化借券率 × 持有期占比）；多头无 carry。
借券率 MVP 用统一常数（真实按标的可借券难度差异化，留二期，诚实标注）。
"""
from __future__ import annotations

from factorzen.markets.base import CostModel

_TRADING_DAYS = 252.0  # 借券率年化 → 日频换算


class USCostModel(CostModel):
    def __init__(self, commission: float = 0.0, slippage: float = 0.0005,
                 borrow_rate: float = 0.03) -> None:
        # commission/slippage 为成交额比例；borrow_rate 为年化借券率（MVP 常数）
        self.commission = commission
        self.slippage = slippage
        self.borrow_rate = borrow_rate

    def trade_cost(self, side: str, notional: float, is_maker: bool = False) -> float:
        return abs(notional) * (self.commission + self.slippage)

    def carry_cost(
        self, position_value: float, periods: int, funding_rate: float = 0.0
    ) -> float:
        # position_value 带符号（多正空负）。仅空头付借券费（正成本）；多头 0。
        short_notional = -position_value if position_value < 0 else 0.0
        return short_notional * self.borrow_rate * periods / _TRADING_DAYS
