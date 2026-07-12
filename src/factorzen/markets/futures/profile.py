"""国内商品期货 MarketProfile 组装 + 注册。

``registry.get("futures")`` 惰性构造（provider 用时才建 Tushare client，import 不联网）。
测试经 ``build_futures_profile(pro=fake, calendar=..., cache_root=tmp)`` 注入离线路径。
RiskModel 本 Phase 传 None（同 crypto MC0 先例，风险模型的期货适配不在本计划范围）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from factorzen.markets import registry
from factorzen.markets.base import MarketProfile
from factorzen.markets.futures.calendar import FuturesCalendar
from factorzen.markets.futures.costs import FuturesCostModel
from factorzen.markets.futures.factors import FuturesFactorSet
from factorzen.markets.futures.provider import COMMODITY_EXCHANGES, FuturesDataProvider
from factorzen.markets.futures.rules import FuturesTradingRules
from factorzen.markets.futures.universe import FuturesUniverse


def build_futures_profile(
    pro: Any = None,
    exchanges: tuple[str, ...] = COMMODITY_EXCHANGES,
    top_n: int = 40,
    lookback_days: int = 20,
    calendar: FuturesCalendar | None = None,
    cache_root: str | Path | None = None,
) -> MarketProfile:
    cal = calendar or FuturesCalendar()
    provider = FuturesDataProvider(
        exchanges=exchanges, pro=pro, calendar=cal, cache_root=cache_root
    )
    universe = FuturesUniverse(provider=provider, top_n=top_n, lookback_days=lookback_days)
    return MarketProfile(
        name="futures",
        quote_currency="CNY",
        base_freq="daily",
        provider=provider,
        calendar=cal,
        rules=FuturesTradingRules(),
        costs=FuturesCostModel(),
        universe=universe,
        factors=FuturesFactorSet(),
        risk=None,
    )


registry.register("futures", build_futures_profile)
