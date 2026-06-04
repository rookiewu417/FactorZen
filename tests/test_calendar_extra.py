"""core/calendar.py 离线补测：缓存有效性、Tushare 拉取、周/月快照、交易时段、边界异常。

补 test_calendar.py 未覆盖的分支，全部离线（mock tushare / 重定向缓存文件）。
"""

import time
from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

import factorzen.core.calendar as cal_mod


def _make_calendar(start: date, days: int) -> pl.DataFrame:
    """生成连续 days 天的日历，工作日 is_open=1，周末=0。"""
    rows = []
    d = start
    for _ in range(days):
        rows.append(
            {"cal_date": d, "is_open": 0 if d.weekday() >= 5 else 1, "pretrade_date": ""}
        )
        d += timedelta(days=1)
    return pl.DataFrame(rows)


@pytest.fixture
def long_calendar(tmp_path, monkeypatch):
    """2024-01-01 起 60 天日历，重定向 _CAL_FILE 并强制缓存有效。"""
    cal_file = tmp_path / "trade_cal.parquet"
    _make_calendar(date(2024, 1, 1), 60).write_parquet(cal_file)
    monkeypatch.setattr(cal_mod, "_CAL_FILE", cal_file)
    monkeypatch.setattr(cal_mod, "_is_cache_valid", lambda: True)
    return cal_mod


# ══════════════════════════════════════════════════════════
# _is_cache_valid
# ══════════════════════════════════════════════════════════


def test_cache_valid_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_mod, "_CAL_FILE", tmp_path / "nope.parquet")
    assert cal_mod._is_cache_valid() is False


def test_cache_valid_fresh_file(tmp_path, monkeypatch):
    f = tmp_path / "trade_cal.parquet"
    pl.DataFrame({"cal_date": [date(2024, 1, 1)], "is_open": [1]}).write_parquet(f)
    monkeypatch.setattr(cal_mod, "_CAL_FILE", f)
    assert cal_mod._is_cache_valid() is True


def test_cache_valid_expired_file(tmp_path, monkeypatch):
    f = tmp_path / "trade_cal.parquet"
    pl.DataFrame({"cal_date": [date(2024, 1, 1)], "is_open": [1]}).write_parquet(f)
    old = time.time() - 30 * 86400
    import os

    os.utime(f, (old, old))
    monkeypatch.setattr(cal_mod, "_CAL_FILE", f)
    assert cal_mod._is_cache_valid() is False


# ══════════════════════════════════════════════════════════
# _fetch_from_tushare / _load_calendar
# ══════════════════════════════════════════════════════════


def test_fetch_from_tushare_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_mod, "_CAL_FILE", tmp_path / "trade_cal.parquet")
    monkeypatch.setattr(cal_mod, "DATA_CACHE", tmp_path)
    monkeypatch.setattr(cal_mod, "ensure_token", lambda: "dummy")

    import tushare as ts

    fake_pro = MagicMock()
    fake_pro.trade_cal.return_value = pd.DataFrame(
        {"cal_date": ["20240102", "20240103"], "is_open": [1, 1]}
    )
    monkeypatch.setattr(ts, "set_token", lambda t: None)
    monkeypatch.setattr(ts, "pro_api", lambda: fake_pro)

    out = cal_mod._fetch_from_tushare()
    assert out["cal_date"].dtype == pl.Date
    assert out["is_open"].dtype == pl.Int8
    assert (tmp_path / "trade_cal.parquet").exists()


def test_fetch_from_tushare_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_mod, "_CAL_FILE", tmp_path / "trade_cal.parquet")
    monkeypatch.setattr(cal_mod, "DATA_CACHE", tmp_path)
    monkeypatch.setattr(cal_mod, "ensure_token", lambda: "dummy")

    import tushare as ts

    fake_pro = MagicMock()
    fake_pro.trade_cal.return_value = pd.DataFrame()
    monkeypatch.setattr(ts, "set_token", lambda t: None)
    monkeypatch.setattr(ts, "pro_api", lambda: fake_pro)

    with pytest.raises(RuntimeError, match="返回空数据"):
        cal_mod._fetch_from_tushare()


def test_load_calendar_cache_miss_fetches(monkeypatch):
    monkeypatch.setattr(cal_mod, "_is_cache_valid", lambda: False)
    sentinel = pl.DataFrame({"cal_date": [date(2024, 1, 2)], "is_open": [1]})
    monkeypatch.setattr(cal_mod, "_fetch_from_tushare", lambda: sentinel)
    assert cal_mod._load_calendar().equals(sentinel)


# ══════════════════════════════════════════════════════════
# is_trade_date / prev / next 边界
# ══════════════════════════════════════════════════════════


def test_is_trade_date_unknown_date_false(long_calendar):
    """日历中不存在的日期 → False。"""
    assert long_calendar.is_trade_date(date(2050, 1, 1)) is False


def test_prev_trade_date_insufficient_raises(long_calendar):
    """日历最早日之前没有足够交易日 → ValueError。"""
    with pytest.raises(ValueError, match="不足"):
        long_calendar.prev_trade_date(date(2024, 1, 2), n=99)


def test_next_trade_date_insufficient_raises(long_calendar):
    with pytest.raises(ValueError, match="不足"):
        long_calendar.next_trade_date(date(2024, 2, 28), n=99)


def test_next_trade_date_multi_step(long_calendar):
    """从周一往后第 5 个交易日应跨周。"""
    result = long_calendar.next_trade_date(date(2024, 1, 1), n=5)  # 周一
    assert result.weekday() < 5  # 落在工作日


def test_prev_trade_date_string_input(long_calendar):
    """字符串入参应被解析为日期。"""
    assert long_calendar.prev_trade_date("20240103", n=1) == date(2024, 1, 2)


def test_next_trade_date_string_input(long_calendar):
    assert long_calendar.next_trade_date("20240105", n=1) == date(2024, 1, 8)


# ══════════════════════════════════════════════════════════
# get_trade_dates / 时段 / 周月快照
# ══════════════════════════════════════════════════════════


def test_get_trade_dates_excludes_weekends(long_calendar):
    dates = long_calendar.get_trade_dates("20240101", "20240107")
    assert all(d.weekday() < 5 for d in dates)
    assert date(2024, 1, 6) not in dates  # 周六


def test_get_trading_sessions():
    sessions = cal_mod.get_trading_sessions()
    assert len(sessions) == 2
    assert sessions[0][0].hour == 9 and sessions[0][0].minute == 30


def test_weekly_snapshot_takes_last_trade_day_per_week(long_calendar):
    snaps = long_calendar.get_weekly_snapshot_dates("20240101", "20240131")
    # 每个快照日应是其 ISO 周内最后一个交易日（通常为周五）
    assert snaps == sorted(snaps)
    assert all(d.weekday() <= 4 for d in snaps)
    # 第一周 (2024-01-01~05) 的快照应为周五 2024-01-05
    assert date(2024, 1, 5) in snaps


def test_weekly_snapshot_empty_when_no_trades(monkeypatch):
    monkeypatch.setattr(cal_mod, "get_trade_dates", lambda s, e: [])
    assert cal_mod.get_weekly_snapshot_dates("20240101", "20240131") == []


def test_monthly_snapshot_takes_last_trade_day_per_month(long_calendar):
    snaps = cal_mod.get_monthly_snapshot_dates("20240101", "20240229")
    # 1 月最后交易日应为 2024-01-31（周三）
    assert date(2024, 1, 31) in snaps
    assert snaps == sorted(snaps)


def test_monthly_snapshot_empty_when_no_trades(monkeypatch):
    monkeypatch.setattr(cal_mod, "get_trade_dates", lambda s, e: [])
    assert cal_mod.get_monthly_snapshot_dates("20240101", "20240131") == []
