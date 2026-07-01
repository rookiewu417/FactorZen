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
        self._positions: dict[str, Position] = {}
        self._cash = Cash(1_000_000.0, 1_000_000.0, 0.0)

    def get_positions(self):
        return self._positions

    def get_cash(self):
        return self._cash

    def place_orders(self, orders):
        self._orders = orders
        self._acks = [OrderAck(f"o{i}", o.ts_code, True, "") for i, o in enumerate(orders)]
        return self._acks

    def poll_fills(self):
        fills = [
            Fill(f"o{i}", o.ts_code, o.side, o.volume, 10.0, 0.0, date(2026, 1, 5))
            for i, o in enumerate(self._orders)
        ]
        # 模拟成交落地：更新持仓/现金，供 broker_state 断言用（不是 step() 自身的镜像计算）。
        for order, fill in zip(self._orders, fills, strict=True):
            cur = self._positions.get(order.ts_code)
            cur_vol = cur.volume if cur else 0
            signed = fill.filled_volume if order.side == "buy" else -fill.filled_volume
            new_vol = cur_vol + signed
            self._positions[order.ts_code] = Position(order.ts_code, new_vol, new_vol, fill.price)
            cost = fill.filled_volume * fill.price
            delta_available = -cost if order.side == "buy" else cost
            delta_market_value = cost if order.side == "buy" else -cost
            self._cash = Cash(
                self._cash.available + delta_available,
                self._cash.total_asset,
                self._cash.market_value + delta_market_value,
            )
        return fills


def test_step_sizes_target_shares_from_nav():
    b = FakeBroker()
    # NAV=100万, 目标权重 A=0.3, ref_price=10 → 目标股数 = round_lot(0.3*1e6/10)=30000
    rec = step(b, {"A.SZ": 0.3}, {"A.SZ": 10.0})
    buy = next(o for o in b._orders if o.ts_code == "A.SZ")
    assert buy.volume == 30000 and buy.side == "buy"
    assert rec["nav_before"] == 1_000_000.0
    assert len(rec["fills"]) == 1


def test_step_record_includes_broker_state():
    b = FakeBroker()
    rec = step(b, {"A.SZ": 0.3}, {"A.SZ": 10.0})
    # broker_state 是 step() 专为满足 store.append() 契约而附带的字段，结构需锁死。
    assert set(rec["broker_state"]) == {"positions", "cash"}
    assert set(rec["broker_state"]["cash"]) == {"available", "total_asset", "market_value"}
    positions = rec["broker_state"]["positions"]
    assert set(positions) == {"A.SZ"}
    assert set(positions["A.SZ"]) == {"ts_code", "volume", "can_use_volume", "avg_cost"}
    # 买入 30000 股 @10 后，broker_state 须真实反映持仓与现金变化（非恒真占位）。
    assert positions["A.SZ"]["volume"] == 30000
    assert rec["broker_state"]["cash"]["available"] == 1_000_000.0 - 30000 * 10.0
