import pytest

from factorzen.execution.broker import (
    BrokerAdapter,
    round_lot,
)


def test_round_lot_floors_to_hundred():
    assert round_lot(150) == 100
    assert round_lot(199.9) == 100
    assert round_lot(-150) == -100      # 卖单同向缩小
    assert round_lot(50) == 0
    assert round_lot(300) == 300

def test_round_lot_absorbs_float_ulp():
    """权重空间往返(shares→delta_w→shares)的浮点 ulp 不应吃掉整手。"""
    # 12900 整手在往返后常低 1-2 ulp（如 12899.999999999998）→ 旧代码 floor 掉一手
    assert round_lot(12899.999999999998) == 12900
    assert round_lot(-12899.999999999998) == -12900
    assert round_lot(9999.99999999999) == 10000
    # 真实的非整手小数仍向零取整（ulp 容差远小于 1 股）
    assert round_lot(12950.4) == 12900
    assert round_lot(12899.5) == 12800

def test_broker_adapter_is_abstract():
    with pytest.raises(TypeError):
        BrokerAdapter()  # 抽象类不可实例化

