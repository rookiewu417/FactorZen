from datetime import date

from factorzen.execution.broker import Order
from factorzen.execution.brokers.paper import PaperBroker


def _mkt(open_, pre_close, close, vol, adv=1e12):
    return {"X.SZ": {"open": open_, "pre_close": pre_close, "close": close, "vol": vol, "adv": adv}}


def test_frictionless_fills_suspended_fully_at_close():
    b = PaperBroker(initial_cash=1_000_000.0, frictionless=True)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 11.0, 0.0))  # vol=0 停牌
    b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    f = b.poll_fills()[0]
    assert f.filled_volume == 1000        # 停牌也全额（frictionless）
    assert abs(f.price - 11.0) < 1e-9     # 按 close 成交
    assert f.cost == 0.0                  # 零成本


def test_frictionless_ignores_cash_and_lot():
    b = PaperBroker(initial_cash=500.0, frictionless=True)  # 现金远不够
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 10.0, 1e6))
    b.place_orders([Order("X.SZ", "buy", 1550, "market", None)])  # 非整百
    assert b.poll_fills()[0].filled_volume == 1550   # 不整手、不受现金限
    assert b.get_cash().available < 0                # 现金可为负


def test_frictionless_false_unchanged():
    # 回归：默认非 frictionless 行为不变（停牌拒单）
    b = PaperBroker(initial_cash=1_000_000.0)
    b.advance_to(date(2026, 1, 5), _mkt(10.0, 10.0, 11.0, 0.0))
    acks = b.place_orders([Order("X.SZ", "buy", 1000, "market", None)])
    assert not acks[0].accepted and acks[0].reason == "suspended"
