"""MC0 Task 2: crypto 24/7 连续交易日历。"""
from __future__ import annotations

from datetime import date

from factorzen.markets.base import Calendar
from factorzen.markets.crypto.calendar import CryptoCalendar


def test_is_a_calendar():
    assert isinstance(CryptoCalendar(), Calendar)


def test_sessions_are_continuous_including_weekends():
    """24/7：区间内每个自然日都是交易日（含周末）。"""
    cal = CryptoCalendar()
    days = cal.sessions("20240101", "20240107")  # 2024-01-06=周六, 01-07=周日
    assert days == [date(2024, 1, d) for d in range(1, 8)]
    assert len(days) == 7


def test_is_session_always_true():
    cal = CryptoCalendar()
    assert cal.is_session(date(2024, 1, 6)) is True  # 周六
    assert cal.is_session("20240107") is True  # 周日


def test_next_prev_session_are_natural_days():
    cal = CryptoCalendar()
    assert cal.next_session(date(2024, 1, 1)) == date(2024, 1, 2)
    assert cal.next_session("20240105", n=2) == date(2024, 1, 7)  # 跨周末
    assert cal.prev_session("20240101", n=2) == date(2023, 12, 30)


def test_periods_per_year():
    cal = CryptoCalendar()
    assert cal.periods_per_year() == 365.0
    assert cal.periods_per_year("daily") == 365.0
    assert cal.periods_per_year("hourly") == 365.0 * 24
    assert cal.periods_per_year("weekly") == 52.0
    assert cal.periods_per_year("monthly") == 12.0
