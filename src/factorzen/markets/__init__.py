"""市场抽象层：共享引擎 + 每市场隔离 adapter。

- ``base``：7 个 Port 抽象接口 + ``MarketProfile``。
- ``registry``：市场注册表（``register`` / ``get`` / ``list_markets``）。
- 各市场 adapter 子包（``crypto`` / ``ashare``）在导入时把自己注册进 registry。
"""
from __future__ import annotations

from factorzen.markets import registry
from factorzen.markets.base import (
    Calendar,
    CostModel,
    DataProvider,
    FactorSet,
    MarketProfile,
    RiskModel,
    TradingRules,
    Universe,
)

__all__ = [
    "Calendar",
    "CostModel",
    "DataProvider",
    "FactorSet",
    "MarketProfile",
    "RiskModel",
    "TradingRules",
    "Universe",
    "registry",
]
