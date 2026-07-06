"""MC0 Task 8: A 股 adapter wrap parity（离线可验证部分）。"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from factorzen.config.constants import (
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
)
from factorzen.discovery.operators import BASIC_FEATURES, LEAF_FEATURES
from factorzen.markets import registry
from factorzen.markets.ashare.calendar import AShareCalendar
from factorzen.markets.ashare.costs import AShareCostModel
from factorzen.markets.ashare.factors import AShareFactorSet
from factorzen.markets.ashare.rules import AShareTradingRules
from factorzen.markets.base import MarketProfile


def test_calendar_periods_per_year_252():
    cal = AShareCalendar()
    assert cal.periods_per_year() == 252.0
    assert cal.periods_per_year("daily") == 252.0
    assert abs(cal.periods_per_year("monthly") - 12.0) < 1e-9


def test_factorset_leaves_match_operators():
    """叶子字典与 discovery.operators 同源（避免漂移）。"""
    fs = AShareFactorSet()
    assert fs.leaf_features() == LEAF_FEATURES
    assert fs.basic_features() == BASIC_FEATURES


def test_factorset_derived_columns_ashare_convention():
    """A 股派生：vwap=amount/vol, log_vol=ln(vol+1), ret_1d 用 close_adj。"""
    fs = AShareFactorSet()
    bars = pl.DataFrame({
        "ts_code": ["000001.SZ"] * 3,
        "trade_date": [date(2024, 1, i) for i in (2, 3, 4)],
        "close_adj": [10.0, 11.0, 10.5],
        "vol": [100.0, 200.0, 50.0],
        "amount": [1000.0, 2200.0, 525.0],
    })
    out = fs.derived_columns(bars).sort("trade_date")
    assert out["vwap"].to_list() == [10.0, 11.0, 10.5]
    np.testing.assert_allclose(out["log_vol"].to_list(), np.log(np.array([100.0, 200.0, 50.0]) + 1))
    assert out["ret_1d"][0] is None
    assert abs(out["ret_1d"][1] - 0.1) < 1e-12


def test_cost_stamp_tax_asymmetry():
    """卖出含印花税、买入不含（A 股关键差异）。"""
    c = AShareCostModel()
    buy = c.trade_cost("buy", 10000.0)
    sell = c.trade_cost("sell", 10000.0)
    assert abs(buy - 10000.0 * (COMMISSION_RATE + SLIPPAGE_RATE)) < 1e-6
    assert abs(sell - 10000.0 * (COMMISSION_RATE + SLIPPAGE_RATE + STAMP_TAX_RATE)) < 1e-6
    # 卖出比买入贵一个印花税
    assert abs((sell - buy) - 10000.0 * STAMP_TAX_RATE) < 1e-6


def test_cost_carry_long_only():
    """long-only：多头无持有成本；空头计融券利息。"""
    c = AShareCostModel()
    assert c.carry_cost(10000.0, 5) == 0.0  # 多头无成本
    assert c.carry_cost(-10000.0, 5) > 0.0  # 空头有融券利息


def test_rules_long_only_t1():
    r = AShareTradingRules()
    assert r.allow_short is False
    assert r.settlement_lag == 1
    assert r.execution_price_col == "open"
    bars = pl.DataFrame({"ts_code": ["A", "B"], "vol": [10.0, 0.0]})
    assert r.tradable_mask(bars, "buy").to_list() == [True, False]


def test_ashare_provider_fetch_bars_rejects_non_daily():
    """AShareDataProvider 仅经 fetch_daily 取日频；非 daily freq 须显式报错，
    而非静默返回日频数据（与 CryptoDataProvider.fetch_funding 的守卫一致）。
    """
    from factorzen.markets.ashare.provider import AShareDataProvider

    p = AShareDataProvider()
    for bad in ["weekly", "monthly", "1min", "60min"]:
        with pytest.raises(ValueError):
            p.fetch_bars(None, "20240101", "20240131", freq=bad)


def test_ashare_provider_fetch_bars_daily_delegates(monkeypatch):
    """freq='daily'（默认）委托 core.loader.fetch_daily，参数透传。"""
    import factorzen.core.loader as loader
    from factorzen.markets.ashare.provider import AShareDataProvider

    sentinel = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date(2024, 1, 2)]})
    called: dict = {}

    def _fake_fetch_daily(start, end, ts_codes=None):
        called["args"] = (start, end, ts_codes)
        return sentinel

    monkeypatch.setattr(loader, "fetch_daily", _fake_fetch_daily)
    out = AShareDataProvider().fetch_bars(["000001.SZ"], "20240101", "20240131")
    assert out.equals(sentinel)
    assert called["args"] == ("20240101", "20240131", ["000001.SZ"])


def test_registry_get_ashare():
    p = registry.get("ashare")
    assert isinstance(p, MarketProfile)
    assert p.name == "ashare"
    assert p.quote_currency == "CNY"
    assert p.risk is None
    assert p.calendar.periods_per_year() == 252.0
