"""美股交易日历：从 bars 数据推导交易日（MVP，不引 pandas_market_calendars 依赖）。

美股周末 + 联邦假日休市，无法像 crypto 用 365 连续日历。MVP 从**基准标的（SPY）的 bar
日期**反推交易日集合——SPY 每交易日必有行情，其日期并集即 NYSE/Nasdaq 交易日。
``periods_per_year("daily") = 252``（美股约定）。

- 注入 ``sessions``（离线测试/已知日历）→ 直接用，不联网。
- 否则惰性经 ``provider.fetch_bars([ref_symbol], ...)`` 拉一个宽窗口反推（仅在真正调用
  日历方法时触发；挖掘评估链路市场无关、不依赖日历，故一般不触发联网）。

**注**：挖掘评估链路（run_session/护栏）不年化、不用交易日历（见计划 Phase 1 侦查：
挖掘链路 IR=ic_mean/ic_std 无 252）；本日历为 Port 完整性与 backtest_window 服务。
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from factorzen.markets.base import Calendar

_PERIODS_PER_YEAR: dict[str, float] = {"daily": 252.0}
# 反推交易日的宽窗口下限（SPY 1993 上市；用 2000 起足够覆盖 MVP 6 年窗口）
_DERIVE_START = "20000101"
_DERIVE_END = "20351231"


def _to_date(d: date | str) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y%m%d").date()
    return d


class USCalendar(Calendar):
    def __init__(
        self,
        sessions: list[date] | None = None,
        provider: Any = None,
        ref_symbol: str = "SPY",
    ) -> None:
        self._sessions = sorted(sessions) if sessions is not None else None
        self._provider = provider
        self.ref_symbol = ref_symbol

    def _all_sessions(self) -> list[date]:
        if self._sessions is not None:
            return self._sessions
        if self._provider is None:
            raise RuntimeError(
                "USCalendar 无 sessions 且无 provider：无法反推交易日（注入 sessions 或 provider）。"
            )
        bars = self._provider.fetch_bars([self.ref_symbol], _DERIVE_START, _DERIVE_END)
        self._sessions = sorted(set(bars["trade_date"].to_list())) if not bars.is_empty() else []
        return self._sessions

    def sessions(self, start: str, end: str) -> list[date]:
        s, e = _to_date(start), _to_date(end)
        return [d for d in self._all_sessions() if s <= d <= e]

    def is_session(self, d: date | str) -> bool:
        return _to_date(d) in set(self._all_sessions())

    def next_session(self, d: date | str, n: int = 1) -> date:
        d0 = _to_date(d)
        future = [x for x in self._all_sessions() if x > d0]
        if len(future) < n:
            raise ValueError(f"不足 {n} 个后续交易日（从 {d} 往后）")
        return future[n - 1]

    def prev_session(self, d: date | str, n: int = 1) -> date:
        d0 = _to_date(d)
        past = [x for x in self._all_sessions() if x < d0]
        if len(past) < n:
            raise ValueError(f"不足 {n} 个前序交易日（从 {d} 往前）")
        return past[-n]

    def periods_per_year(self, freq: str = "daily") -> float:
        if freq not in _PERIODS_PER_YEAR:
            raise ValueError(f"us 未知频率: {freq!r}，支持 {sorted(_PERIODS_PER_YEAR)}")
        return _PERIODS_PER_YEAR[freq]
