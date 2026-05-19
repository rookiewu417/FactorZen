"""测试 IntradayDataContext。"""

from datetime import date
from unittest.mock import patch

import polars as pl
import pytest

from intraday.data.context import IntradayDataContext


@pytest.fixture(autouse=True)
def mock_prev_trade_date(monkeypatch):
    """Keep these unit tests offline; calendar integration is covered separately."""

    def _fake_prev_trade_date(d: str, n: int = 1) -> date:
        assert n == 5
        return {
            "20260514": date(2026, 5, 7),
            "20260105": date(2025, 12, 25),
        }[d]

    monkeypatch.setattr("intraday.data.context.prev_trade_date", _fake_prev_trade_date)

# ── 构造与默认值 ──────────────────────────────────────────────────────────


def test_construction_defaults():
    """基本构造：默认属性与预期一致。"""
    ctx = IntradayDataContext("20260514", "20260514")
    assert ctx.start == "20260514"
    assert ctx.end == "20260514"
    assert ctx.bar_size == "1min"
    assert ctx.required_data == ["minute"]
    assert ctx.universe is None
    assert ctx.max_bars == 10_000
    assert ctx._minute is None


def test_construction_custom():
    """自定义参数正确存储。"""
    ctx = IntradayDataContext(
        "20260501",
        "20260510",
        bar_size="5min",
        required_data=["minute", "daily_basic"],
        universe=["000001.SZ", "000002.SZ"],
        max_bars=2000,
    )
    assert ctx.bar_size == "5min"
    assert ctx.required_data == ["minute", "daily_basic"]
    assert ctx.universe == ["000001.SZ", "000002.SZ"]
    assert ctx.max_bars == 2000


# ── expanded_start ────────────────────────────────────────────────────────


def test_expanded_start():
    """expanded_start 向前扩展 5 天。"""
    ctx = IntradayDataContext("20260514", "20260514")
    assert ctx.expanded_start == "20260507"

    ctx2 = IntradayDataContext("20260105", "20260110")
    assert ctx2.expanded_start == "20251225"


# ── required_data 校验 ────────────────────────────────────────────────────


def test_minute_not_declared_raises():
    """minute 不在 required_data 中时，访问 .minute 抛出 ValueError。"""
    ctx = IntradayDataContext("20260514", "20260514", required_data=["other"])
    with pytest.raises(ValueError, match="minute data not declared"):
        _ = ctx.minute


# ── 惰性加载与缓存 ────────────────────────────────────────────────────────


def test_minute_lazy_loading():
    """访问 .minute 触发 load_parquet 调用，结果缓存到 _minute。"""
    synthetic = pl.DataFrame(
        {
            "trade_time": ["2026-05-14 09:30:00"],
            "ts_code": ["000001.SZ"],
        }
    ).lazy()

    with patch("intraday.data.context.load_parquet", return_value=synthetic) as mock_load:
        ctx = IntradayDataContext("20260514", "20260514")
        assert ctx._minute is None

        result = ctx.minute
        mock_load.assert_called_once()
        assert isinstance(result, pl.LazyFrame)
        assert ctx._minute is not None

        # 再次访问应命中缓存，不重复调用
        _ = ctx.minute
        mock_load.assert_called_once()


def test_minute_universe_filter():
    """传入 universe 时对 LazyFrame 添加 .is_in() 过滤。"""
    synthetic = pl.DataFrame(
        {
            "trade_time": ["2026-05-14 09:30:00", "2026-05-14 09:31:00"],
            "ts_code": ["000001.SZ", "999999.SZ"],
        }
    ).lazy()

    with patch("intraday.data.context.load_parquet", return_value=synthetic):
        ctx = IntradayDataContext("20260514", "20260514", universe=["000001.SZ"])
        lf = ctx.minute
        collected = lf.collect()
        assert collected.height == 1
        assert collected["ts_code"][0] == "000001.SZ"


# ── load_all ──────────────────────────────────────────────────────────────


def test_load_all():
    """load_all() 触发 minute 数据加载。"""
    synthetic = pl.DataFrame(
        {
            "trade_time": ["2026-05-14 09:30:00"],
            "ts_code": ["000001.SZ"],
        }
    ).lazy()

    with patch("intraday.data.context.load_parquet", return_value=synthetic):
        ctx = IntradayDataContext("20260514", "20260514")
        assert ctx._minute is None
        ctx.load_all()
        assert ctx._minute is not None
