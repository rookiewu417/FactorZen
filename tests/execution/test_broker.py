from datetime import date

import pytest

from factorzen.execution.broker import (
    BrokerAdapter,
    Cash,
    Fill,
    Order,
    OrderAck,
    Position,
    round_lot,
)


def test_round_lot_floors_to_hundred():
    assert round_lot(150) == 100
    assert round_lot(199.9) == 100
    assert round_lot(-150) == -100      # 卖单同向缩小
    assert round_lot(50) == 0
    assert round_lot(300) == 300

def test_dataclasses_construct():
    assert Position("X.SZ", 200, 200, 10.0).volume == 200
    assert Cash(1e5, 1e6, 9e5).total_asset == 1e6
    assert Order("X.SZ", "buy", 100, "limit", 10.0).side == "buy"
    assert OrderAck("o1", "X.SZ", True, "").accepted is True
    assert Fill("o1", "X.SZ", "buy", 100, 10.0, 2.5, date(2026, 1, 5)).filled_volume == 100

def test_broker_adapter_is_abstract():
    with pytest.raises(TypeError):
        BrokerAdapter()  # 抽象类不可实例化

def test_concrete_subclass_ok():
    class Dummy(BrokerAdapter):
        def get_positions(self): return {}
        def get_cash(self): return Cash(0.0, 0.0, 0.0)
        def place_orders(self, orders): return []
        def poll_fills(self): return []
    d = Dummy()
    assert d.get_cash().total_asset == 0.0
