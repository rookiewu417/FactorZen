"""MC0 Task 7: crypto MarketProfile 注册 + 离线端到端 smoke。"""
from __future__ import annotations

from factorzen.markets import registry
from factorzen.markets.base import MarketProfile
from factorzen.markets.crypto.profile import build_crypto_profile
from tests.test_markets_crypto_provider import FakeCCXT


def test_registry_get_crypto():
    p = registry.get("crypto")
    assert isinstance(p, MarketProfile)
    assert p.name == "crypto"
    assert p.quote_currency == "USDT"
    assert p.base_freq == "daily"
    from factorzen.markets.crypto.risk import CryptoRiskModel
    assert isinstance(p.risk, CryptoRiskModel)  # MC3 填入 crypto 风险模型
    assert p.calendar.periods_per_year() == 365.0


def test_offline_end_to_end_pipeline():
    """注入 FakeCCXT，走 provider→factors→universe 端到端，不联网。"""
    p = build_crypto_profile(client=FakeCCXT())
    bars = p.provider.fetch_bars(["BTCUSDT", "ETHUSDT"], "20240101", "20240103")
    assert bars.height == 5
    enriched = p.factors.derived_columns(bars)
    assert {"vwap", "log_vol", "ret_1d"} <= set(enriched.columns)
    snap = p.universe.snapshot("20240103")
    assert "BTCUSDT" in snap  # 非空、schema 正确
