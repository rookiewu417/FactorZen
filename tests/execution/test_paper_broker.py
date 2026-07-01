from datetime import date

from factorzen.execution.broker import Order
from factorzen.execution.brokers.paper import PaperBroker


def _mkt(open_, pre_close, close, vol, adv=1e12):
    # adv 极大 → 容量不绑定；聚焦其它摩擦
    return {"X.SZ": {"open": open_, "pre_close": pre_close, "close": close, "vol": vol, "adv": adv}}


def test_buy_fill_updates_cash_and_position():
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 10.0, 1e6))
    acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    fills = b.poll_fills()
    assert acks[0].accepted and fills[0].filled_volume == 1000
    pos = b.get_positions()["X.SZ"]
    assert pos.volume == 1000
    # 现金 = 100万 - 1000*10 - 成本
    assert b.get_cash().available < 1_000_000.0 - 10_000.0 + 1e-6


def test_suspended_rejects_order():
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 10.0, 0.0))  # vol=0 停牌
    acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    assert not acks[0].accepted and acks[0].reason == "suspended"
    assert b.poll_fills() == []
    assert "X.SZ" not in b.get_positions()


def test_limit_up_rejects_buy():
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.99, 10.0, 11.0, 1e6))  # 开盘+9.9%涨停
    acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    assert not acks[0].accepted and acks[0].reason == "limit_up"


def test_lot_rounding_drops_remainder():
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 10.0, 1e6))
    # 下 150 股 → 整手向零取整到 100
    acks = b.place_orders([Order("X.SZ", "buy", 150, "market", None)])
    assert b.poll_fills()[0].filled_volume == 100
    assert acks[0].reason == "lot_round"


def test_insufficient_cash_caps_buy():
    b = PaperBroker(initial_cash=1_050.0)  # 只够 100 股(1000元)+成本
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 10.0, 1e6))
    b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    fill = b.poll_fills()[0]
    assert fill.filled_volume == 100 and b.get_cash().available >= 0.0


def test_t1_frozen_blocks_same_day_sell():
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 10.0, 1e6))
    b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    b.poll_fills()
    # 同日卖：can_use_volume 当日买入部分为 0（T+1）
    assert b.get_positions()["X.SZ"].can_use_volume == 0
    acks = b.place_orders([Order("X.SZ", "sell", 1000, "market", None)])
    assert not acks[0].accepted and acks[0].reason == "t1_frozen"


def test_total_asset_marks_to_close():
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 12.0, 1e6))  # close=12
    b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    b.poll_fills()
    cash = b.get_cash()
    # 持仓市值按 close=12 标记 = 1000*12 = 12000
    assert abs(cash.market_value - 12_000.0) < 1e-6
