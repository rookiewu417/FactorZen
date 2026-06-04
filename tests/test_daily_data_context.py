"""daily/data/context.py 的离线单测。

FactorDataContext 仅依赖 load_parquet 与 calendar，全部 monkeypatch，
不触碰真实 data/ 或 Tushare。重点覆盖复权 join、adj 缺失回退、universe 过滤、
未声明数据报错、快照下采样与惰性缓存。
"""

from datetime import date

import polars as pl
import pytest

from factorzen.daily.data import context as ctx_mod
from factorzen.daily.data.context import FactorDataContext


def _daily_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "close": [10.0, 11.0, 20.0],
            "open": [9.0, 10.0, 19.0],
            "high": [10.5, 11.5, 20.5],
            "low": [8.5, 9.5, 18.5],
            "vol": [100.0, 200.0, 300.0],
        }
    )


def _adj_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "adj_factor": [2.0, 2.0, 1.0],
        }
    )


def _basic_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "pe": [15.0, 16.0, 30.0],
        }
    )


@pytest.fixture
def patched(monkeypatch):
    """重定向 prev_trade_date 与 load_parquet，提供 daily/daily_basic/adj 合成数据。"""
    monkeypatch.setattr(ctx_mod, "prev_trade_date", lambda d, n: date(2023, 12, 1))

    def fake_load(category, start=None, end=None):
        if category == "daily":
            return _daily_df().lazy()
        if category == "daily_basic":
            return _basic_df().lazy()
        if category == "adj_factor":
            return _adj_df().lazy()
        raise ValueError(f"未知 category: {category}")

    monkeypatch.setattr(ctx_mod, "load_parquet", fake_load)
    return monkeypatch


# ══════════════════════════════════════════════════════════
# expanded_start
# ══════════════════════════════════════════════════════════


def test_expanded_start_uses_prev_trade_date(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", lookback_days=20)
    assert ctx.expanded_start == "20231201"


# ══════════════════════════════════════════════════════════
# daily 属性：复权 join / 回退 / universe / 缓存 / 未声明
# ══════════════════════════════════════════════════════════


def test_daily_applies_adj_factor(patched):
    ctx = FactorDataContext(start="20240102", end="20240103")
    df = ctx.daily.collect().sort(["ts_code", "trade_date"])
    # A 在 2024-01-02：close 10 × adj 2.0 = 20.0
    row = df.filter((pl.col("ts_code") == "A") & (pl.col("trade_date") == date(2024, 1, 2)))
    assert row["close_adj"].item() == 20.0
    assert "adj_factor" not in df.columns  # join 后已 drop


def test_daily_fallback_when_adj_missing(patched, monkeypatch):
    """adj_factor 未落盘（load 抛异常）时，close_adj 回退为原始价格。"""

    def fake_load(category, start=None, end=None):
        if category == "daily":
            return _daily_df().lazy()
        raise FileNotFoundError("adj_factor 未落盘")

    monkeypatch.setattr(ctx_mod, "load_parquet", fake_load)
    ctx = FactorDataContext(start="20240102", end="20240103")
    df = ctx.daily.collect()
    row = df.filter((pl.col("ts_code") == "A") & (pl.col("trade_date") == date(2024, 1, 2)))
    assert row["close_adj"].item() == 10.0  # 回退原值


def test_daily_filters_universe(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", universe=["A"])
    df = ctx.daily.collect()
    assert set(df["ts_code"].to_list()) == {"A"}


def test_daily_not_declared_raises(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", required_data=["daily_basic"])
    with pytest.raises(ValueError, match="daily data not declared"):
        _ = ctx.daily


def test_daily_lazy_cached(patched):
    ctx = FactorDataContext(start="20240102", end="20240103")
    assert ctx.daily is ctx.daily  # 第二次命中缓存，同一对象


# ══════════════════════════════════════════════════════════
# daily_basic
# ══════════════════════════════════════════════════════════


def test_daily_basic_loads(patched):
    ctx = FactorDataContext(
        start="20240102", end="20240103", required_data=["daily", "daily_basic"]
    )
    df = ctx.daily_basic.collect()
    assert "pe" in df.columns
    assert df.height == 3


def test_daily_basic_not_declared_raises(patched):
    ctx = FactorDataContext(start="20240102", end="20240103", required_data=["daily"])
    with pytest.raises(ValueError, match="daily_basic data not declared"):
        _ = ctx.daily_basic


def test_daily_basic_filters_universe(patched):
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily_basic"],
        universe=["B"],
    )
    df = ctx.daily_basic.collect()
    assert set(df["ts_code"].to_list()) == {"B"}


# ══════════════════════════════════════════════════════════
# snapshot_dates 三种模式
# ══════════════════════════════════════════════════════════


def test_snapshot_dates_daily_mode(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_trade_dates",
        lambda s, e: [date(2024, 1, 2), date(2024, 1, 3)],
    )
    ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="daily")
    assert ctx.snapshot_dates == [date(2024, 1, 2), date(2024, 1, 3)]


def test_snapshot_dates_weekly_mode(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_weekly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="weekly")
    assert ctx.snapshot_dates == [date(2024, 1, 3)]


def test_snapshot_dates_monthly_mode(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_monthly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="monthly")
    assert ctx.snapshot_dates == [date(2024, 1, 3)]


# ══════════════════════════════════════════════════════════
# 下采样属性：weekly / monthly / *_basic
# ══════════════════════════════════════════════════════════


def test_weekly_downsamples_to_snapshot(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_weekly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="weekly")
    df = ctx.weekly.collect()
    assert df["trade_date"].unique().to_list() == [date(2024, 1, 3)]
    assert ctx.weekly is ctx.weekly  # 缓存


def test_monthly_downsamples_to_snapshot(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_monthly_snapshot_dates",
        lambda s, e: [date(2024, 1, 2)],
    )
    ctx = FactorDataContext(start="20240102", end="20240103", snapshot_mode="monthly")
    df = ctx.monthly.collect()
    assert df["trade_date"].unique().to_list() == [date(2024, 1, 2)]


def test_weekly_basic_downsamples(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_weekly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily_basic"],
        snapshot_mode="weekly",
    )
    df = ctx.weekly_basic.collect()
    assert df["trade_date"].unique().to_list() == [date(2024, 1, 3)]


def test_monthly_basic_downsamples(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_monthly_snapshot_dates",
        lambda s, e: [date(2024, 1, 2)],
    )
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily_basic"],
        snapshot_mode="monthly",
    )
    df = ctx.monthly_basic.collect()
    assert df["trade_date"].unique().to_list() == [date(2024, 1, 2)]


# ══════════════════════════════════════════════════════════
# load_all
# ══════════════════════════════════════════════════════════


def test_load_all_daily_and_basic(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_weekly_snapshot_dates",
        lambda s, e: [date(2024, 1, 3)],
    )
    ctx = FactorDataContext(
        start="20240102",
        end="20240103",
        required_data=["daily", "daily_basic"],
        snapshot_mode="weekly",
    )
    ctx.load_all()
    # 所有惰性缓存均已填充
    assert ctx._daily is not None
    assert ctx._daily_basic is not None
    assert ctx._weekly_snapshot is not None


def test_load_all_monthly_mode(patched, monkeypatch):
    monkeypatch.setattr(
        "factorzen.core.calendar.get_monthly_snapshot_dates",
        lambda s, e: [date(2024, 1, 2)],
    )
    ctx = FactorDataContext(
        start="20240102", end="20240103", required_data=["daily"], snapshot_mode="monthly"
    )
    ctx.load_all()
    assert ctx._monthly_snapshot is not None
