"""国内商品期货成本模型：手续费 + 滑点常数 bp（MVP，品种差异留二期）。

与 A 股不同：无印花税、买卖对称、双向可开。持有成本 MVP 记 0（不建模仓储/展期基差成本，
诚实标注：真实展期有基差损益，本 Phase 只做挖掘链路，成本用常数近似）。品种级差异化手续费
（按成交额比例 vs 按手固定）留二期。
"""
from __future__ import annotations

from factorzen.markets.base import CostModel


class FuturesCostModel(CostModel):
    def __init__(self, fee: float = 0.0003, slippage: float = 0.0005) -> None:
        # fee/slippage 均为成交额比例（MVP 常数 bp；真实各品种手续费口径不一，未逐品种建模）
        self.fee = fee
        self.slippage = slippage

    def trade_cost(self, side: str, notional: float, is_maker: bool = False) -> float:
        return abs(notional) * (self.fee + self.slippage)

    def carry_cost(
        self, position_value: float, periods: int, funding_rate: float = 0.0
    ) -> float:
        # MVP：不建模展期基差/仓储成本，持有成本记 0（诚实标注的近似）。
        return 0.0
