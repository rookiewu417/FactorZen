from datetime import date

from factorzen.execution.broker import BrokerAdapter, Cash, Fill, OrderAck, Position
from factorzen.execution.engine import build_orders, step


def test_build_orders_sell_before_buy():
    positions = {"A.SZ": Position("A.SZ", 1000, 1000, 10.0)}  # 现持 A 1000
    target = {"A.SZ": 300, "B.SZ": 500}  # A 减到 300, B 建 500
    orders = build_orders(target, positions)
    sides = [(o.ts_code, o.side, o.volume) for o in orders]
    # 卖单必须排在买单前
    assert sides.index(("A.SZ", "sell", 700)) < sides.index(("B.SZ", "buy", 500))


def test_build_orders_skips_zero_delta():
    positions = {"A.SZ": Position("A.SZ", 300, 300, 10.0)}
    assert build_orders({"A.SZ": 300}, positions) == []


class FakeBroker(BrokerAdapter):
    def __init__(self):
        self._acks = []
        self._orders = []

    def get_positions(self):
        return {}

    def get_cash(self):
        return Cash(1_000_000.0, 1_000_000.0, 0.0)

    def place_orders(self, orders):
        self._orders = orders
        self._acks = [OrderAck(f"o{i}", o.ts_code, True, "") for i, o in enumerate(orders)]
        return self._acks

    def poll_fills(self):
        return [
            Fill(f"o{i}", o.ts_code, o.side, o.volume, 10.0, 0.0, date(2026, 1, 5))
            for i, o in enumerate(self._orders)
        ]


def test_step_sizes_target_shares_from_nav():
    b = FakeBroker()
    # NAV=100万, 目标权重 A=0.3, ref_price=10 → 目标股数 = round_lot(0.3*1e6/10)=30000
    rec = step(b, {"A.SZ": 0.3}, {"A.SZ": 10.0})
    buy = next(o for o in b._orders if o.ts_code == "A.SZ")
    assert buy.volume == 30000 and buy.side == "buy"
    assert rec["nav_before"] == 1_000_000.0
    assert len(rec["fills"]) == 1
