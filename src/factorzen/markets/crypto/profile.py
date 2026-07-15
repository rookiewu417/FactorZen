"""crypto perps MarketProfile 组装 + 注册。

``registry.get("crypto")`` 惰性构造（client=None → 用时才建 ccxt，不 import 即联网）。
测试经 ``build_crypto_profile(client=fake)`` 注入离线 client。
RiskModel 本期(MC0)传 None，延后 MC3 填。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from factorzen.config.settings import CRYPTO_LAKE
from factorzen.markets import registry
from factorzen.markets.base import MarketProfile
from factorzen.markets.crypto.calendar import CryptoCalendar
from factorzen.markets.crypto.costs import CryptoCostModel
from factorzen.markets.crypto.factors import CryptoFactorSet
from factorzen.markets.crypto.provider import CryptoDataProvider
from factorzen.markets.crypto.rules import CryptoTradingRules
from factorzen.markets.crypto.universe import CryptoUniverse


def build_crypto_profile(
    client: Any = None,
    exchange_id: str = "binanceusdm",
    quote: str = "USDT",
    top_n: int = 50,
    lookback_days: int = 30,
    min_amount: float = 0.0,
    min_list_days: int = 30,
    source: str | None = None,
    lake_root: str | Path = CRYPTO_LAKE,
) -> MarketProfile:
    from factorzen.markets.crypto.lake_provider import CryptoLakeProvider
    from factorzen.markets.crypto.risk import CryptoRiskModel

    # 默认走数据湖(REST 当前 451 不可达);注入 client → ccxt(既有测试零改动);source 显式优先
    resolved = source or ("ccxt" if client is not None else "lake")
    if resolved == "ccxt":
        provider: Any = CryptoDataProvider(exchange_id=exchange_id, client=client, quote=quote)
    elif resolved == "lake":
        provider = CryptoLakeProvider(lake_root=lake_root)
    else:
        raise ValueError(f"未知 source: {resolved!r},支持 'lake' / 'ccxt'")
    universe = CryptoUniverse(
        provider=provider,
        top_n=top_n,
        lookback_days=lookback_days,
        min_amount=min_amount,
        min_list_days=min_list_days,
    )
    return MarketProfile(
        name="crypto",
        quote_currency=quote,
        base_freq="daily",
        provider=provider,
        calendar=CryptoCalendar(),
        rules=CryptoTradingRules(),
        costs=CryptoCostModel(),
        universe=universe,
        factors=CryptoFactorSet(),
        risk=CryptoRiskModel(),
    )


registry.register("crypto", build_crypto_profile)
