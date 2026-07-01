"""有状态逐日 stepper：目标权重 → 目标持仓 → reconcile 下单 → 收成交 → StepRecord。

broker 无关：只调 BrokerAdapter 4 方法，不碰约束逻辑。
"""
from __future__ import annotations

from dataclasses import asdict

from factorzen.execution.broker import BrokerAdapter, Order, Position, round_lot


def build_orders(
    target_shares: dict[str, int], positions: dict[str, Position]
) -> list[Order]:
    """目标持仓 - 现持仓 → 差额单；先卖后买（先腾现金）。"""
    sells: list[Order] = []
    buys: list[Order] = []
    codes = set(target_shares) | set(positions)
    for code in sorted(codes):
        cur = positions[code].volume if code in positions else 0
        tgt = target_shares.get(code, 0)
        delta = tgt - cur
        if delta == 0:
            continue
        if delta < 0:
            sells.append(Order(code, "sell", -delta, "market", None))
        else:
            buys.append(Order(code, "buy", delta, "market", None))
    return sells + buys


def step(
    broker: BrokerAdapter,
    target_weights: dict[str, float],
    ref_price: dict[str, float],
) -> dict:
    """按目标权重 reconcile 一步：查现状 → 算目标股数 → 差额下单 → 收成交 → 记账。"""
    positions = broker.get_positions()
    nav_before = broker.get_cash().total_asset
    target_shares = {
        code: round_lot(w * nav_before / ref_price[code])
        for code, w in target_weights.items()
        if code in ref_price and ref_price[code] > 0
    }
    orders = build_orders(target_shares, positions)
    acks = broker.place_orders(orders)
    fills = broker.poll_fills()
    nav_after = broker.get_cash().total_asset
    return {
        "orders": [asdict(o) for o in orders],
        "acks": [asdict(a) for a in acks],
        "fills": [{**asdict(f), "ts": f.ts.isoformat()} for f in fills],
        "nav_before": nav_before,
        "nav_after": nav_after,
        "broker_state": {
            "positions": {code: asdict(p) for code, p in broker.get_positions().items()},
            "cash": asdict(broker.get_cash()),
        },
    }
