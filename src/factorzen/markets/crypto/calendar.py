"""crypto 24/7 连续交易日历。

与 A 股 SSE 日历不同：无休市、无节假日，每个自然日都是交易日。
年化周期数按频率取（日频 365，替代 A 股硬编的 252）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from factorzen.markets.base import Calendar

# 年化周期数（无休市 → 365 天/年）
_PERIODS_PER_YEAR: dict[str, float] = {
    "daily": 365.0,
    "hourly": 365.0 * 24,
    "weekly": 52.0,
    "monthly": 12.0,
}


def _to_date(d: date | str) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y%m%d").date()
    return d


class CryptoCalendar(Calendar):
    """24/7 连续日历：每个自然日均为交易日。"""

    def sessions(self, start: str, end: str) -> list[date]:
        s = _to_date(start)
        e = _to_date(end)
        if e < s:
            return []
        n = (e - s).days
        return [s + timedelta(days=i) for i in range(n + 1)]

    def is_session(self, d: date | str) -> bool:
        _to_date(d)  # 校验可解析
        return True

    def next_session(self, d: date | str, n: int = 1) -> date:
        return _to_date(d) + timedelta(days=n)

    def prev_session(self, d: date | str, n: int = 1) -> date:
        return _to_date(d) - timedelta(days=n)

    def periods_per_year(self, freq: str = "daily") -> float:
        if freq not in _PERIODS_PER_YEAR:
            raise ValueError(f"未知频率: {freq!r}，支持 {sorted(_PERIODS_PER_YEAR)}")
        return _PERIODS_PER_YEAR[freq]
