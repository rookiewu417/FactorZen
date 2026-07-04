"""A 股 MarketProfile 组装 + 注册。

各 port 委托现有实现（wrap 不重写）。RiskModel 本期(MC0)传 None，MC3 填
（现有 risk/ Barra 模型将来包装成 RiskModel port）。
"""
from __future__ import annotations

from factorzen.markets import registry
from factorzen.markets.ashare.calendar import AShareCalendar
from factorzen.markets.ashare.costs import AShareCostModel
from factorzen.markets.ashare.factors import AShareFactorSet
from factorzen.markets.ashare.provider import AShareDataProvider
from factorzen.markets.ashare.rules import AShareTradingRules
from factorzen.markets.ashare.universe import AShareUniverse
from factorzen.markets.base import MarketProfile


def build_ashare_profile() -> MarketProfile:
    return MarketProfile(
        name="ashare",
        quote_currency="CNY",
        base_freq="daily",
        provider=AShareDataProvider(),
        calendar=AShareCalendar(),
        rules=AShareTradingRules(),
        costs=AShareCostModel(),
        universe=AShareUniverse(),
        factors=AShareFactorSet(),
        risk=None,
    )


registry.register("ashare", build_ashare_profile)
