"""券商执行接口(port) + 数据类。纸面与真券商是它的两个后端。

字段照 xtquant/miniQMT 模型设计，实盘期零改动映射。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


def round_lot(shares: float) -> int:
    """向零取整到 100 股整手（绝不超量）。execution 共享，paper/engine 都 import。"""
    return int(math.floor(abs(shares) / 100.0) * 100.0) * (1 if shares >= 0 else -1)


@dataclass(frozen=True)
class Position:
    ts_code: str
    volume: int           # 总持仓股数        ← xtquant: volume
    can_use_volume: int   # 可卖股数(T+1冻结)  ← xtquant: can_use_volume
    avg_cost: float       # 成本价            ← xtquant: open_price


@dataclass(frozen=True)
class Cash:
    available: float      # 可用资金  ← xtquant: cash
    total_asset: float    # 总资产    ← xtquant: total_asset（= 货币口径 NAV）
    market_value: float   # 持仓市值  ← xtquant: market_value


@dataclass(frozen=True)
class Order:
    ts_code: str
    side: str             # "buy" | "sell"
    volume: int           # 股数，买入须 %100==0
    price_type: str       # "limit" | "market"
    price: float | None


@dataclass(frozen=True)
class OrderAck:
    order_id: str
    ts_code: str
    accepted: bool
    reason: str           # 拒单原因（paper: suspended/limit_up/…；实盘: 券商码）


@dataclass(frozen=True)
class Fill:
    order_id: str
    ts_code: str
    side: str
    filled_volume: int
    price: float
    cost: float           # 佣金+印花+滑点
    ts: date


class BrokerAdapter(ABC):
    """上层 engine 只认这 4 个方法，不知后端真假。"""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def get_cash(self) -> Cash: ...

    @abstractmethod
    def place_orders(self, orders: list[Order]) -> list[OrderAck]: ...

    @abstractmethod
    def poll_fills(self) -> list[Fill]: ...
