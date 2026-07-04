"""MC0 Task 4: crypto TradingRules + CostModel（T+0/可空 + maker/taker/funding）。"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import CostModel, TradingRules
from factorzen.markets.crypto.costs import CryptoCostModel
from factorzen.markets.crypto.rules import CryptoTradingRules


def test_rules_are_tradingrules():
    assert isinstance(CryptoTradingRules(), TradingRules)


def test_rules_t0_short_execution():
    r = CryptoTradingRules()
    assert r.allow_short is True
    assert r.settlement_lag == 0  # T+0
    assert r.execution_price_col == "close"  # next-bar close 撮合


def test_tradable_mask_blocks_zero_volume():
    r = CryptoTradingRules()
    bars = pl.DataFrame({"ts_code": ["A", "B", "C"], "vol": [10.0, 0.0, 5.0]})
    buy = r.tradable_mask(bars, "buy")
    sell = r.tradable_mask(bars, "sell")
    assert buy.to_list() == [True, False, True]
    # crypto 买卖对称（无涨跌停不对称）
    assert sell.to_list() == [True, False, True]


def test_costs_are_costmodel():
    assert isinstance(CryptoCostModel(), CostModel)


def test_trade_cost_symmetric_maker_taker():
    c = CryptoCostModel(maker=0.0002, taker=0.0005, slippage=0.0005)
    # taker 卖：10000*(0.0005+0.0005)=10.0
    assert abs(c.trade_cost("sell", 10000.0, is_maker=False) - 10.0) < 1e-9
    # maker 买：10000*(0.0002+0.0005)=7.0
    assert abs(c.trade_cost("buy", 10000.0, is_maker=True) - 7.0) < 1e-9
    # 买卖对称：同 side 参数下成本相等（无印花税不对称）
    assert c.trade_cost("buy", 10000.0) == c.trade_cost("sell", 10000.0)


def test_carry_cost_funding_sign():
    c = CryptoCostModel()
    # 多头 pos=+10000, funding=0.0001, 3 期 → 付费 +3.0
    assert abs(c.carry_cost(10000.0, 3, funding_rate=0.0001) - 3.0) < 1e-9
    # 空头 pos=-10000 同 funding → 收费 -3.0
    assert abs(c.carry_cost(-10000.0, 3, funding_rate=0.0001) - (-3.0)) < 1e-9
    # funding=0 → 无 carry
    assert c.carry_cost(10000.0, 5, funding_rate=0.0) == 0.0
