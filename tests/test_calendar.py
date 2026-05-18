"""common/calendar.py 的单元测试（使用本地缓存 mock，不调用 Tushare）。"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _make_mock_calendar(tmp_path: Path) -> pl.DataFrame:
    """生成 2024-01-01 ~ 2024-01-10 的模拟交易日历，工作日为交易日。"""
    from datetime import date, timedelta

    rows = []
    d = date(2024, 1, 1)
    for _ in range(10):
        rows.append(
            {
                "cal_date": d,
                "is_open": 0 if d.weekday() >= 5 else 1,
                "pretrade_date": "",
            }
        )
        d += timedelta(days=1)
    df = pl.DataFrame(rows)
    cal_file = tmp_path / "trade_cal.parquet"
    df.write_parquet(cal_file)
    return df


@pytest.fixture()
def mock_calendar(tmp_path, monkeypatch):
    """将 _CAL_FILE 和 _is_cache_valid 重定向到 tmp 目录。"""
    import common.calendar as cal_mod

    cal_file = tmp_path / "trade_cal.parquet"
    _make_mock_calendar(tmp_path)

    monkeypatch.setattr(cal_mod, "_CAL_FILE", cal_file)

    def _always_valid():
        return True

    monkeypatch.setattr(cal_mod, "_is_cache_valid", _always_valid)
    return cal_mod


def test_is_trade_date_weekday(mock_calendar):
    assert mock_calendar.is_trade_date(date(2024, 1, 2)) is True  # 周二


def test_is_trade_date_weekend(mock_calendar):
    assert mock_calendar.is_trade_date(date(2024, 1, 6)) is False  # 周六


def test_is_trade_date_string_input(mock_calendar):
    assert mock_calendar.is_trade_date("20240102") is True


def test_prev_trade_date(mock_calendar):
    # 2024-01-03（周三）的前一个交易日是 2024-01-02（周二）
    result = mock_calendar.prev_trade_date(date(2024, 1, 3), n=1)
    assert result == date(2024, 1, 2)


def test_next_trade_date(mock_calendar):
    # 2024-01-05（周五）的下一个交易日是 2024-01-08（周一，下一周）
    result = mock_calendar.next_trade_date(date(2024, 1, 5), n=1)
    assert result == date(2024, 1, 8)


def test_get_trade_calendar_filter(mock_calendar):
    cal = mock_calendar.get_trade_calendar(start="20240103", end="20240105")
    assert cal.shape[0] == 3
    assert cal["cal_date"].min() == date(2024, 1, 3)
    assert cal["cal_date"].max() == date(2024, 1, 5)
