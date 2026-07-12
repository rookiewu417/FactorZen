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

# 导入各市场 adapter 的 profile 模块以触发 registry 注册（import 时不联网）。
from factorzen.markets.ashare import profile as _ashare_profile  # noqa: F401
from factorzen.markets.crypto import profile as _crypto_profile  # noqa: F401
from factorzen.markets.futures import profile as _futures_profile  # noqa: F401
from factorzen.markets.us import profile as _us_profile  # noqa: F401
