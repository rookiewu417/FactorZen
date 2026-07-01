"""A 股 Calendar port —— 委托现有 core.calendar（SSE 日历）。"""
from __future__ import annotations

from datetime import date

from factorzen.config.constants import (
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,
    TRADING_DAYS_PER_YEAR,
)
from factorzen.markets.base import Calendar

# 年化周期数（A 股按交易日：日频 252）
_PERIODS_PER_YEAR: dict[str, float] = {
    "daily": float(TRADING_DAYS_PER_YEAR),
    "weekly": float(TRADING_DAYS_PER_YEAR) / TRADING_DAYS_PER_WEEK,   # 252/5 ≈ 50.4
    "monthly": float(TRADING_DAYS_PER_YEAR) / TRADING_DAYS_PER_MONTH,  # 252/21 = 12
}


class AShareCalendar(Calendar):
    def sessions(self, start: str, end: str) -> list[date]:
        from factorzen.core.calendar import get_trade_dates

        return get_trade_dates(start, end)

    def is_session(self, d: date | str) -> bool:
        from factorzen.core.calendar import is_trade_date

        return is_trade_date(d)

    def next_session(self, d: date | str, n: int = 1) -> date:
        from factorzen.core.calendar import next_trade_date

        return next_trade_date(d, n)

    def prev_session(self, d: date | str, n: int = 1) -> date:
        from factorzen.core.calendar import prev_trade_date

        return prev_trade_date(d, n)

    def periods_per_year(self, freq: str = "daily") -> float:
        if freq not in _PERIODS_PER_YEAR:
            raise ValueError(f"未知频率: {freq!r}，支持 {sorted(_PERIODS_PER_YEAR)}")
        return _PERIODS_PER_YEAR[freq]
