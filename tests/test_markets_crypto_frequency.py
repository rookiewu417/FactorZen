"""频率表单测:别名/年化/未知 raise(全链路唯一事实源)。"""
import pytest

from factorzen.markets.crypto.frequency import BAR_FREQS, normalize_freq, periods_per_year


def test_normalize_known_and_alias():
    assert normalize_freq("daily") == "daily"
    assert normalize_freq("1h") == "1h"
    assert normalize_freq("hourly") == "1h"  # 别名
    assert normalize_freq("15m") == "15m"


def test_normalize_unknown_raises():
    with pytest.raises(ValueError, match="未知频率"):
        normalize_freq("3m")


def test_periods_per_year_values():
    assert periods_per_year("1m") == 365.0 * 24 * 60
    assert periods_per_year("5m") == 365.0 * 24 * 12
    assert periods_per_year("15m") == 365.0 * 24 * 4
    assert periods_per_year("1h") == 365.0 * 24
    assert periods_per_year("daily") == 365.0
    assert periods_per_year("hourly") == 365.0 * 24  # 别名走 1h
    assert periods_per_year("weekly") == 52.0  # calendar 兼容
    assert periods_per_year("monthly") == 12.0


def test_bar_freqs_polars_every():
    assert BAR_FREQS["daily"].every == "1d"
    assert BAR_FREQS["15m"].every == "15m"
    assert BAR_FREQS["daily"].timeframe == "1d"  # ccxt timeframe
