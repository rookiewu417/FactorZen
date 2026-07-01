"""MC0 Task 1: 市场抽象地基 —— Port 接口 + MarketProfile + registry。"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import polars as pl
import pytest

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


# ── 最小 concrete 子类（用于构造 DummyProfile）────────────────────────────────
class _DP(DataProvider):
    def fetch_bars(self, symbols, start, end, freq="daily"):
        return pl.DataFrame({"ts_code": [], "trade_date": []})

    def fetch_symbol_meta(self):
        return pl.DataFrame({"ts_code": []})


class _CAL(Calendar):
    def sessions(self, start, end):
        return [date(2024, 1, 1)]

    def is_session(self, d):
        return True

    def next_session(self, d, n=1):
        return date(2024, 1, 2)

    def prev_session(self, d, n=1):
        return date(2023, 12, 31)

    def periods_per_year(self, freq="daily"):
        return 365.0


class _RULES(TradingRules):
    @property
    def allow_short(self):
        return True

    @property
    def settlement_lag(self):
        return 0

    @property
    def execution_price_col(self):
        return "close"

    def tradable_mask(self, bars, side):
        return pl.Series([True] * bars.height)


class _COST(CostModel):
    def trade_cost(self, side, notional, is_maker=False):
        return 0.0

    def carry_cost(self, position_value, periods, funding_rate=0.0):
        return 0.0


class _UNI(Universe):
    def snapshot(self, d):
        return ["BTCUSDT"]

    def benchmark(self, start, end):
        return pl.DataFrame({"trade_date": [], "close": []})


class _FS(FactorSet):
    def leaf_features(self):
        return {"close": "close"}

    def basic_features(self):
        return set()

    def derived_columns(self, bars):
        return bars


class _RISK(RiskModel):
    def style_factors(self):
        return {}

    def sector_classification(self, symbols, d):
        return pl.DataFrame()


def _dummy_profile() -> MarketProfile:
    return MarketProfile(
        name="dummy",
        quote_currency="XXX",
        base_freq="daily",
        provider=_DP(),
        calendar=_CAL(),
        rules=_RULES(),
        costs=_COST(),
        universe=_UNI(),
        factors=_FS(),
        risk=_RISK(),
    )


# ── 测试 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cls",
    [DataProvider, Calendar, TradingRules, CostModel, Universe, FactorSet, RiskModel],
)
def test_ports_are_abstract(cls):
    """7 个 Port 均为抽象类，不可直接实例化。"""
    with pytest.raises(TypeError):
        cls()  # type: ignore[abstract]


def test_market_profile_bundles_ports():
    """MarketProfile 打包 7 个 port + 元数据，且 frozen。"""
    p = _dummy_profile()
    assert p.name == "dummy"
    assert p.quote_currency == "XXX"
    assert p.base_freq == "daily"
    assert isinstance(p.provider, DataProvider)
    assert isinstance(p.calendar, Calendar)
    assert p.calendar.periods_per_year() == 365.0
    with pytest.raises(FrozenInstanceError):
        p.name = "changed"  # type: ignore[misc]  # frozen


def test_risk_is_optional():
    """RiskModel 可为 None（crypto 本期延后到 MC3 填）。"""
    p = _dummy_profile()
    p2 = MarketProfile(
        name="norisk",
        quote_currency="XXX",
        base_freq="daily",
        provider=p.provider,
        calendar=p.calendar,
        rules=p.rules,
        costs=p.costs,
        universe=p.universe,
        factors=p.factors,
    )
    assert p2.risk is None


def test_registry_register_get_list():
    """registry：register→get 返回同一 profile（缓存），list 含其名。"""
    registry.register("dummy", _dummy_profile)
    got = registry.get("dummy")
    assert got.name == "dummy"
    assert registry.get("dummy") is got  # 缓存：同一实例
    assert "dummy" in registry.list_markets()


def test_registry_unknown_raises():
    """未注册市场 get 抛 KeyError。"""
    with pytest.raises(KeyError):
        registry.get("__nonexistent_market__")
