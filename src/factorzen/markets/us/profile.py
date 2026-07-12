"""美股 MarketProfile 组装 + 注册。

``registry.get("us")`` 惰性构造（provider 用时才建 Yahoo client，import 不联网）。
测试经 ``build_us_profile(fetch=fake, cache_root=tmp, symbols=[...])`` 注入离线路径。
RiskModel 本 Phase 传 None（同 crypto/futures MC0 先例，风险模型的美股适配不在本计划范围）。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from factorzen.markets import registry
from factorzen.markets.base import MarketProfile
from factorzen.markets.us.calendar import USCalendar
from factorzen.markets.us.costs import USCostModel
from factorzen.markets.us.factors import USFactorSet
from factorzen.markets.us.provider import USDataProvider, _http_get
from factorzen.markets.us.rules import USTradingRules
from factorzen.markets.us.universe import USUniverse


def build_us_profile(
    top_n: int | None = None,
    cache_root: str | Path | None = None,
    fetch: Callable[[str], bytes] = _http_get,
    request_interval: float = 0.3,
    symbols: list[str] | None = None,
) -> MarketProfile:
    provider = USDataProvider(
        cache_root=cache_root, fetch=fetch, request_interval=request_interval,
        universe_symbols=symbols,
    )
    universe = USUniverse(provider=provider, top_n=top_n, symbols=symbols)
    calendar = USCalendar(provider=provider)
    return MarketProfile(
        name="us",
        quote_currency="USD",
        base_freq="daily",
        provider=provider,
        calendar=calendar,
        rules=USTradingRules(),
        costs=USCostModel(),
        universe=universe,
        factors=USFactorSet(),
        risk=None,
    )


registry.register("us", build_us_profile)
