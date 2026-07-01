"""PaperBroker：用真实历史行情 + 共享约束内核撮合的「模拟券商」。

{停牌/涨跌停/容量} 走共享内核；{整手/现金占用/T+1} 由本类新增。
"""
from __future__ import annotations

from datetime import date

from factorzen.daily.evaluation.backtest import BacktestConfig, CostModel
from factorzen.daily.evaluation.trade_constraints import apply_trade_constraints
from factorzen.execution.broker import (
    BrokerAdapter,
    Cash,
    Fill,
    Order,
    OrderAck,
    Position,
    round_lot,
)


class PaperBroker(BrokerAdapter):
    def __init__(
        self,
        initial_cash: float,
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        slippage_bps: float = 0.0,
        frictionless: bool = False,
    ) -> None:
        self._cash = float(initial_cash)
        self._config = config if config is not None else BacktestConfig()
        self._cost = cost_model if cost_model is not None else CostModel()
        self._slip = slippage_bps / 10_000.0
        self._frictionless = frictionless
        # 持仓：ts_code -> {volume, can_use_volume, avg_cost}
        self._pos: dict[str, dict[str, float]] = {}
        self._as_of: date | None = None
        self._market: dict[str, dict[str, float]] = {}
        self._fill_buf: list[Fill] = []
        self._order_seq = 0

    # ---- paper 专用时钟（不进共享接口）----
    def advance_to(self, as_of: date, market: dict[str, dict[str, float]]) -> None:
        # 进入新交易日：上一日持仓全部解冻（T+1）
        if self._as_of is None or as_of != self._as_of:
            for p in self._pos.values():
                p["can_use_volume"] = p["volume"]
        self._as_of = as_of
        self._market = market

    # ---- BrokerAdapter 接口 ----
    def get_positions(self) -> dict[str, Position]:
        return {
            code: Position(code, int(p["volume"]), int(p["can_use_volume"]), p["avg_cost"])
            for code, p in self._pos.items()
            if p["volume"] > 0
        }

    def get_cash(self) -> Cash:
        mv = 0.0
        for code, p in self._pos.items():
            close = self._market.get(code, {}).get("close")
            if close is not None:
                mv += p["volume"] * float(close)
        return Cash(available=self._cash, total_asset=self._cash + mv, market_value=mv)

    def place_orders(self, orders: list[Order]) -> list[OrderAck]:
        acks: list[OrderAck] = []
        total_asset = self.get_cash().total_asset
        for od in orders:
            self._order_seq += 1
            oid = f"paper-{self._order_seq}"
            ack = self._exec_one(oid, od, total_asset)
            acks.append(ack)
        return acks

    def poll_fills(self) -> list[Fill]:
        out = self._fill_buf
        self._fill_buf = []
        return out

    # ---- 撮合核心 ----
    def _exec_one(self, oid: str, od: Order, total_asset: float) -> OrderAck:
        rec = self._market.get(od.ts_code)
        if rec is None or rec.get("open") is None:
            return OrderAck(oid, od.ts_code, False, "missing_price")
        if self._frictionless:
            close = rec.get("close")
            price = float(close) if close is not None else float(rec["open"])
            signed = od.volume if od.side == "buy" else -od.volume
            # 全额、零成本、无约束/整手/现金/T+1
            self._apply_fill(oid, od, signed, price, 0.0)
            return OrderAck(oid, od.ts_code, True, "")
        open_px = float(rec["open"])
        exec_price = open_px * (1.0 + self._slip if od.side == "buy" else 1.0 - self._slip)
        signed = od.volume if od.side == "buy" else -od.volume

        # 1) 共享内核：停牌/涨跌停/容量（权重空间）
        delta_w = signed * exec_price / total_asset if total_asset > 0 else 0.0
        price_map = {od.ts_code: {"open": open_px, "pre_close": rec.get("pre_close"), "vol": rec.get("vol")}}
        ach_w, reason = apply_trade_constraints(
            code=od.ts_code, delta=delta_w, price_map=price_map,
            portfolio_value=total_asset, config=self._config, adv=rec.get("adv"),
        )
        if ach_w == 0.0 and reason:
            return OrderAck(oid, od.ts_code, False, reason)
        ach_shares_raw = ach_w * total_asset / exec_price

        # 2) 整手向零取整
        capped_shares = round_lot(ach_shares_raw)
        lot_dropped = abs(capped_shares) < abs(ach_shares_raw) - 1e-9

        # 3) 现金/持仓约束
        reason_out = "capacity" if reason == "capacity" else ("lot_round" if lot_dropped else "")
        if od.side == "buy":
            affordable = self._cash / (exec_price * (1.0 + self._cost.one_way_cost()))
            fill_shares = min(capped_shares, round_lot(affordable))
            if fill_shares < capped_shares:
                reason_out = "insufficient_cash"
        else:  # sell
            held = self._pos.get(od.ts_code, {})
            can_use = int(held.get("can_use_volume", 0))
            fill_shares = -min(abs(capped_shares), can_use)
            if can_use == 0:
                return OrderAck(oid, od.ts_code, False, "t1_frozen")
            if abs(fill_shares) < abs(capped_shares):
                reason_out = "t1_frozen"

        if fill_shares == 0:
            return OrderAck(oid, od.ts_code, False, reason_out or "no_fill")

        # 4) 成本 + 落账
        cost = self._trade_cost(fill_shares, exec_price, od.side)
        self._apply_fill(oid, od, fill_shares, exec_price, cost)
        return OrderAck(oid, od.ts_code, True, reason_out)

    def _trade_cost(self, shares: int, price: float, side: str) -> float:
        rate = self._cost.sell_cost() if side == "sell" else self._cost.one_way_cost()
        return abs(shares) * price * rate

    def _apply_fill(self, oid: str, od: Order, shares: int, price: float, cost: float) -> None:
        p = self._pos.setdefault(
            od.ts_code, {"volume": 0.0, "can_use_volume": 0.0, "avg_cost": 0.0}
        )
        if shares > 0:  # 买
            new_vol = p["volume"] + shares
            p["avg_cost"] = (p["avg_cost"] * p["volume"] + price * shares) / new_vol
            p["volume"] = new_vol
            # 当日买入不计入可卖（T+1）
            self._cash -= shares * price + cost
        else:  # 卖
            p["volume"] += shares  # shares 为负
            p["can_use_volume"] = max(0.0, p["can_use_volume"] + shares)
            self._cash += (-shares) * price - cost
        self._fill_buf.append(
            Fill(oid, od.ts_code, od.side, abs(shares), price, cost, self._as_of)  # type: ignore[arg-type]
        )

    # ---- 续跑态（供 store）----
    def state(self) -> dict:
        return {"cash": self._cash, "pos": self._pos, "order_seq": self._order_seq}

    def load_state(self, d: dict) -> None:
        self._cash = float(d["cash"])
        self._pos = {k: dict(v) for k, v in d["pos"].items()}
        self._order_seq = int(d.get("order_seq", 0))
