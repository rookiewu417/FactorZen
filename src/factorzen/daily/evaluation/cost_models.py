"""成本模型：线性成本（向后兼容）和平方根冲击模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


class CostModelBase(ABC):
    """成本模型抽象基类。"""

    @abstractmethod
    def trade_cost(
        self,
        delta_weight: float,
        price: float = 1.0,
        adv: float | None = None,
    ) -> float:
        """计算单笔交易成本（占 NAV 比例）。"""

    def borrow_rate_per_period(self, frequency: str = "daily") -> float:
        """融券日/周/月费率（按年化利率折算）。子类可覆盖。"""
        return 0.0


@dataclass
class LinearCostModel(CostModelBase):
    """线性成本模型（向后兼容原 CostModel）。"""

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.001
    annual_borrow_rate: float = 0.08
    trading_days_per_year: int = 252

    def trade_cost(
        self,
        delta_weight: float,
        price: float = 1.0,
        adv: float | None = None,
    ) -> float:
        if delta_weight == 0:
            return 0.0
        abs_delta = abs(delta_weight)
        commission = abs_delta * self.commission_rate
        stamp = abs_delta * self.stamp_tax_rate if delta_weight < 0 else 0.0  # 卖出印花税
        slippage = abs_delta * self.slippage_rate
        return commission + stamp + slippage

    def borrow_rate_per_period(self, frequency: str = "daily") -> float:
        days = {"daily": 1, "weekly": 5, "monthly": 21}.get(frequency, 1)
        return self.annual_borrow_rate * days / self.trading_days_per_year


@dataclass
class SquareRootImpactCostModel(CostModelBase):
    """平方根冲击成本模型。

    总成本 = 线性成本（佣金 + 印花税 + 滑点）+ 冲击成本
    冲击成本 = alpha * |delta_weight|^1.5

    当 adv 提供时，用 adv_normalized = adv / fallback_adv 对冲击成本做缩放：
    冲击成本 = alpha * |delta_weight|^1.5 / sqrt(adv_normalized)

    Args:
        alpha: 冲击系数，默认 0.1
        commission_rate: 佣金率
        stamp_tax_rate: 印花税（卖出）
        slippage_rate: 固定滑点
        annual_borrow_rate: 融券年化成本
        trading_days_per_year: 年交易日
        fallback_adv: ADV 缺失时的参考值（元），默认 1e7（1000 万）
    """

    alpha: float = 0.1
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.001
    annual_borrow_rate: float = 0.08
    trading_days_per_year: int = 252
    fallback_adv: float = 1e7

    def trade_cost(
        self,
        delta_weight: float,
        price: float = 1.0,
        adv: float | None = None,
    ) -> float:
        if delta_weight == 0:
            return 0.0

        abs_delta = abs(delta_weight)

        # 平方根冲击：impact = alpha * |delta|^1.5
        # 当 adv 已知时，按 adv_normalized = adv / fallback_adv 做缩放
        effective_adv = adv if (adv is not None and adv > 0) else self.fallback_adv
        adv_normalized = effective_adv / self.fallback_adv
        impact = self.alpha * (abs_delta**1.5) / max(np.sqrt(adv_normalized), 1e-6)

        commission = abs_delta * self.commission_rate
        stamp = abs_delta * self.stamp_tax_rate if delta_weight < 0 else 0.0
        slippage = abs_delta * self.slippage_rate

        return commission + stamp + slippage + impact

    def borrow_rate_per_period(self, frequency: str = "daily") -> float:
        days = {"daily": 1, "weekly": 5, "monthly": 21}.get(frequency, 1)
        return self.annual_borrow_rate * days / self.trading_days_per_year
