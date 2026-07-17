"""tests/test_resample_equiv.py — resample_intraday 等价性（优化前 ground truth 锁定）。

以本文件内嵌的旧实现为 ground truth；边界 fixture + 可选真实 2024-06 截断样本。
行序契约：输出按 (ts_code, trade_time) 排序；消费方多会再 sort，此处仍比齐。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.intraday.sessions import (
    ASHARE_BAR_FREQS,
    normalize_freq,
    resample_intraday,
    session_bar_index,
)
from factorzen.intraday.sessions import (  # noqa: PLC0415 — 测私有辅助
    _bucket_end_minutes,
    _canonical_mask,
    _empty_bars,
)


def _dt(h: int, m: int, day: int = 2, month: int = 1) -> datetime:
    return datetime(2024, month, day, h, m, 0)


def _resample_intraday_ref(
    bars: pl.DataFrame,
    freq: str,
    *,
    time_col: str = "trade_time",
) -> pl.DataFrame:
    """W3-A 优化前的 resample 实现（锁定语义）。"""
    freq = normalize_freq(freq)
    k = ASHARE_BAR_FREQS[freq].minutes

    if bars.is_empty():
        return _empty_bars()

    filtered = bars.filter(_canonical_mask(time_col))
    if filtered.is_empty():
        return _empty_bars()

    idx = session_bar_index(time_col)
    work = (
        filtered.sort([time_col])
        .with_columns(
            idx.alias("_idx"),
            pl.col(time_col).dt.date().alias("_date"),
        )
        .with_columns(
            pl.when(pl.col("_idx") == 0)
            .then(pl.lit(0, dtype=pl.Int32))
            .otherwise(((pl.col("_idx") - 1) // k).cast(pl.Int32))
            .alias("_bucket")
        )
        .filter(pl.col("_idx").is_not_null())
    )

    if work.is_empty():
        return _empty_bars()

    return (
        work.group_by(["ts_code", "_date", "_bucket"], maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("vol").sum().cast(pl.Int64).alias("vol"),
            pl.col("amount").sum().cast(pl.Float64).alias("amount"),
        )
        .with_columns(
            ((pl.col("_bucket") + 1) * k).alias("_end_idx"),
        )
        .with_columns(
            (
                pl.col("_date").cast(pl.Datetime("us"))
                + pl.duration(minutes=_bucket_end_minutes(pl.col("_end_idx")))
            ).alias(time_col)
        )
        .select(
            pl.col("ts_code").cast(pl.String),
            pl.col(time_col).cast(pl.Datetime("us")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("vol").cast(pl.Int64),
            pl.col("amount").cast(pl.Float64),
        )
        .sort(["ts_code", time_col])
    )


def _assert_equiv(src: pl.DataFrame, freq: str = "5min") -> None:
    keys = ["ts_code", "trade_time"]
    exp = _resample_intraday_ref(src, freq).sort(keys)
    got = resample_intraday(src, freq).sort(keys)
    assert_frame_equal(got, exp, check_column_order=True, check_dtypes=True)


class TestResampleBoundaryFixtures:
    def test_lunch_boundary_bars(self) -> None:
        """午休边界：11:30 与 13:01 不跨桶吞并。"""
        rows = [
            ("000001.SZ", _dt(11, 28), 10.0, 10.1, 9.9, 10.0, 10, 100.0),
            ("000001.SZ", _dt(11, 29), 10.0, 10.2, 9.9, 10.1, 20, 200.0),
            ("000001.SZ", _dt(11, 30), 10.1, 10.3, 10.0, 10.2, 30, 300.0),
            ("000001.SZ", _dt(13, 1), 10.2, 10.4, 10.1, 10.3, 40, 400.0),
            ("000001.SZ", _dt(13, 2), 10.3, 10.5, 10.2, 10.4, 50, 500.0),
        ]
        df = pl.DataFrame(
            {
                "ts_code": [r[0] for r in rows],
                "trade_time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[2] for r in rows],
                "high": [r[3] for r in rows],
                "low": [r[4] for r in rows],
                "close": [r[5] for r in rows],
                "vol": pl.Series([r[6] for r in rows], dtype=pl.Int64),
                "amount": [r[7] for r in rows],
            }
        )
        _assert_equiv(df, "5min")
        out = resample_intraday(df, "5min")
        labels = [
            (t.hour, t.minute) for t in out["trade_time"].to_list()  # type: ignore[union-attr]
        ]
        assert (11, 30) in labels
        assert (13, 5) in labels
        assert (13, 0) not in labels

    def test_open_close_session_labels(self) -> None:
        """开盘竞价 + 收盘 15:00 标签。"""
        rows = [
            ("000001.SZ", _dt(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
            ("000001.SZ", _dt(9, 31), 10.0, 10.2, 9.9, 10.1, 200, 2000.0),
            ("000001.SZ", _dt(9, 35), 10.1, 10.3, 10.0, 10.2, 50, 500.0),
            ("000001.SZ", _dt(14, 58), 11.0, 11.1, 10.9, 11.0, 10, 100.0),
            ("000001.SZ", _dt(14, 59), 11.0, 11.0, 11.0, 11.0, 0, 0.0),
            ("000001.SZ", _dt(15, 0), 11.0, 11.2, 10.9, 11.1, 500, 5500.0),
            ("000001.SZ", _dt(15, 5), 11.1, 11.1, 11.1, 11.1, 1, 1.0),  # drop
        ]
        df = pl.DataFrame(
            {
                "ts_code": [r[0] for r in rows],
                "trade_time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[2] for r in rows],
                "high": [r[3] for r in rows],
                "low": [r[4] for r in rows],
                "close": [r[5] for r in rows],
                "vol": pl.Series([r[6] for r in rows], dtype=pl.Int64),
                "amount": [r[7] for r in rows],
            }
        )
        _assert_equiv(df, "5min")

    def test_partial_bucket_and_single_stock_day(self) -> None:
        """不足一个完整 bar 的残段 + 单股票单日。"""
        rows = [
            ("000001.SZ", _dt(10, 1), 10.0, 10.1, 9.9, 10.05, 10, 100.0),
            ("000001.SZ", _dt(10, 2), 10.05, 10.2, 10.0, 10.1, 20, 200.0),
            # 10:01-10:02 → 桶 end 10:05，残段 3 分钟缺失
        ]
        df = pl.DataFrame(
            {
                "ts_code": [r[0] for r in rows],
                "trade_time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[2] for r in rows],
                "high": [r[3] for r in rows],
                "low": [r[4] for r in rows],
                "close": [r[5] for r in rows],
                "vol": pl.Series([r[6] for r in rows], dtype=pl.Int64),
                "amount": [r[7] for r in rows],
            }
        )
        _assert_equiv(df, "5min")
        out = resample_intraday(df, "5min")
        assert out.height == 1
        assert out["vol"][0] == 30
        assert out["open"][0] == pytest.approx(10.0)
        assert out["close"][0] == pytest.approx(10.1)

    def test_unsorted_multi_stock_input(self) -> None:
        """乱序多股票输入：first/last 仍按时间正确。"""
        rows = [
            ("000002.SZ", _dt(9, 32), 20.0, 20.1, 19.9, 20.05, 5, 100.0),
            ("000001.SZ", _dt(9, 31), 10.0, 10.1, 9.9, 10.05, 10, 100.0),
            ("000001.SZ", _dt(9, 33), 10.1, 10.2, 10.0, 10.15, 20, 200.0),
            ("000002.SZ", _dt(9, 31), 20.0, 20.2, 19.8, 20.1, 8, 160.0),
            ("000001.SZ", _dt(9, 32), 10.05, 10.15, 10.0, 10.1, 15, 150.0),
        ]
        df = pl.DataFrame(
            {
                "ts_code": [r[0] for r in rows],
                "trade_time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[2] for r in rows],
                "high": [r[3] for r in rows],
                "low": [r[4] for r in rows],
                "close": [r[5] for r in rows],
                "vol": pl.Series([r[6] for r in rows], dtype=pl.Int64),
                "amount": [r[7] for r in rows],
            }
        )
        _assert_equiv(df, "5min")

    def test_empty_frame(self) -> None:
        _assert_equiv(_empty_bars(), "5min")


class TestResampleRealMonthSample:
    @pytest.fixture(scope="module")
    def minute_sample(self) -> pl.DataFrame | None:
        path = Path("data/raw/minute_1min/year=2024/month=06/data.parquet")
        if not path.exists():
            return None
        # 截断到若干千只股票以控时
        codes = (
            pl.scan_parquet(path)
            .select("ts_code")
            .unique()
            .collect()
            .head(800)["ts_code"]
            .to_list()
        )
        return (
            pl.scan_parquet(path)
            .filter(pl.col("ts_code").is_in(codes))
            .select(["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"])
            .collect()
        )

    def test_real_2024_06_truncated_equiv(self, minute_sample: pl.DataFrame | None) -> None:
        if minute_sample is None:
            pytest.skip("no data/raw/minute_1min/year=2024/month=06")
        _assert_equiv(minute_sample, "5min")
