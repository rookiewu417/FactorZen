"""tests/test_intraday_audit.py — 分钟 bar 口径审计纯函数单测。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from factorzen.intraday.audit import (
    coverage_report,
    infer_label_convention,
    reconcile_with_daily,
    timestamp_census,
)


def _dt(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, 0)


def _minute_healthy(code: str = "000001.SZ", day: date = date(2024, 1, 2)) -> pl.DataFrame:
    """合成分钟帧：vol 合计 = 日线 vol × 100，amount 合计 = 日线 amount × 1000。"""
    # 两根 bar：09:30 + 15:00，便于 open/close 对齐
    rows = [
        (code, _dt(day.year, day.month, day.day, 9, 30), 10.0, 10.5, 9.8, 10.2, 6000, 60000.0),
        (code, _dt(day.year, day.month, day.day, 15, 0), 10.2, 10.6, 10.0, 10.4, 4000, 40000.0),
    ]
    return pl.DataFrame(
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


def _daily_healthy(code: str = "000001.SZ", day: date = date(2024, 1, 2)) -> pl.DataFrame:
    """与 _minute_healthy 闭合：vol=100 手，amount=100 千元。"""
    # minute vol sum = 10000 股 → daily vol = 100 手 (×100)
    # minute amount sum = 100000 元 → daily amount = 100 千元 (×1000)
    return pl.DataFrame(
        {
            "ts_code": [code],
            "trade_date": pl.Series([day], dtype=pl.Date),
            "open": [10.0],
            "high": [10.6],
            "low": [9.8],
            "close": [10.4],
            "vol": pl.Series([100], dtype=pl.Int64),
            "amount": [100.0],
        }
    )


class TestTimestampCensus:
    def test_groups_by_year_and_board(self) -> None:
        df = pl.concat(
            [
                _minute_healthy("000001.SZ", date(2024, 1, 2)),
                _minute_healthy("600000.SH", date(2024, 1, 2)).with_columns(
                    # 加一根盘后 bar
                    pl.col("trade_time")
                ),
            ]
        )
        # 北交所盘后
        bj = pl.DataFrame(
            {
                "ts_code": ["920001.BJ"],
                "trade_time": pl.Series(
                    [_dt(2024, 1, 2, 15, 10)], dtype=pl.Datetime("us")
                ),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "vol": pl.Series([5], dtype=pl.Int64),
                "amount": [5.0],
            }
        )
        df = pl.concat([df, bj], how="vertical_relaxed")
        census = timestamp_census(df)
        boards = set(census["board"].to_list())
        assert "000" in boards
        assert "600" in boards
        assert "920" in boards
        bj_row = census.filter(pl.col("board") == "920")
        assert bj_row["n_after_1500"][0] == 1
        assert bj_row["vol_after_1500"][0] == 5


class TestReconcileWithDaily:
    def test_healthy_multipliers(self) -> None:
        recon = reconcile_with_daily(_minute_healthy(), _daily_healthy())
        assert recon.height == 1
        assert recon["vol_multiplier"][0] == pytest.approx(100.0)
        assert recon["amount_multiplier"][0] == pytest.approx(1000.0)
        assert recon["open_match"][0] is True
        assert recon["close_match"][0] is True
        assert recon["high_match"][0] is True
        assert recon["low_match"][0] is True

    def test_detects_wrong_unit_frame(self) -> None:
        """故意构造 ×1 错误单位帧，应检出 multiplier≈1。"""
        minute = _minute_healthy()
        # 日线也用「股/元」→ 倍率 ≈ 1
        daily = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": pl.Series([date(2024, 1, 2)], dtype=pl.Date),
                "open": [10.0],
                "high": [10.6],
                "low": [9.8],
                "close": [10.4],
                "vol": pl.Series([10000], dtype=pl.Int64),  # 股而非手
                "amount": [100000.0],  # 元而非千元
            }
        )
        recon = reconcile_with_daily(minute, daily)
        assert recon["vol_multiplier"][0] == pytest.approx(1.0)
        assert recon["amount_multiplier"][0] == pytest.approx(1.0)
        # OHLC 仍应匹配
        assert recon["open_match"][0] is True
        assert recon["close_match"][0] is True


class TestInferLabelConvention:
    def test_end_convention(self) -> None:
        times = [
            _dt(2024, 1, 2, 9, 30),
            _dt(2024, 1, 2, 9, 31),
            _dt(2024, 1, 2, 11, 30),
            _dt(2024, 1, 2, 13, 1),
            _dt(2024, 1, 2, 15, 0),
        ]
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * len(times),
                "trade_time": pl.Series(times, dtype=pl.Datetime("us")),
            }
        )
        result = infer_label_convention(df)
        assert result["label_convention"] == "end"
        assert result["has_0930"] is True
        assert result["has_after_1500"] is False
        assert result["first_time"] == "09:30"
        assert result["last_time"] == "15:00"

    def test_start_convention(self) -> None:
        # start 标签：有 13:00 无 11:30
        times = [
            _dt(2024, 1, 2, 9, 30),
            _dt(2024, 1, 2, 9, 31),
            _dt(2024, 1, 2, 11, 29),
            _dt(2024, 1, 2, 13, 0),
            _dt(2024, 1, 2, 14, 59),
        ]
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * len(times),
                "trade_time": pl.Series(times, dtype=pl.Datetime("us")),
            }
        )
        result = infer_label_convention(df)
        assert result["label_convention"] == "start"

    def test_after_1500_flag(self) -> None:
        times = [_dt(2024, 1, 2, 11, 30), _dt(2024, 1, 2, 15, 10)]
        df = pl.DataFrame(
            {
                "ts_code": ["920001.BJ"] * 2,
                "trade_time": pl.Series(times, dtype=pl.Datetime("us")),
            }
        )
        result = infer_label_convention(df)
        assert result["has_after_1500"] is True


def _write_month_partition(
    base: Path, year: int, month: int, days: list[date], codes: list[str]
) -> None:
    """写一个月分区 parquet（最小 schema）。"""
    rows_code: list[str] = []
    rows_time: list[datetime] = []
    for d in days:
        for c in codes:
            rows_code.append(c)
            rows_time.append(datetime(d.year, d.month, d.day, 9, 31))
    df = pl.DataFrame(
        {
            "ts_code": rows_code,
            "trade_time": pl.Series(rows_time, dtype=pl.Datetime("us")),
            "open": [1.0] * len(rows_code),
            "high": [1.0] * len(rows_code),
            "low": [1.0] * len(rows_code),
            "close": [1.0] * len(rows_code),
            "vol": pl.Series([1] * len(rows_code), dtype=pl.Int64),
            "amount": [1.0] * len(rows_code),
        }
    )
    part = base / f"year={year}" / f"month={month:02d}"
    part.mkdir(parents=True, exist_ok=True)
    df.write_parquet(part / "data.parquet")


class TestCoverageReport:
    def test_detects_missing_range_and_merges(self, tmp_path: Path) -> None:
        """注入 trade_dates，挖掉一段连续缺失，验证区间合并。"""
        # 期望 10 个交易日：1/2..1/11（人工日历）
        expected = [date(2024, 1, d) for d in range(2, 12)]
        # 有数据：1/2,1/3,1/4 和 1/9,1/10,1/11；缺 1/5..1/8
        present = [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 9),
            date(2024, 1, 10),
            date(2024, 1, 11),
        ]
        _write_month_partition(tmp_path, 2024, 1, present, ["000001.SZ"])

        report = coverage_report(
            "20240102",
            "20240111",
            base_dir=tmp_path,
            trade_dates=expected,
        )
        assert report["n_expected_days"] == 10
        assert report["n_present_days"] == 6
        assert report["missing_ranges"] == [["2024-01-05", "2024-01-08"]]
        assert "2024-01" in report["months_present"]
        assert report["per_month_rows"]["2024-01"] == 6

    def test_two_disjoint_gaps_merge_separately(self, tmp_path: Path) -> None:
        expected = [date(2024, 2, d) for d in range(1, 11)]
        # 有：1,2,5,6,9,10；缺 3-4 与 7-8
        present = [
            date(2024, 2, 1),
            date(2024, 2, 2),
            date(2024, 2, 5),
            date(2024, 2, 6),
            date(2024, 2, 9),
            date(2024, 2, 10),
        ]
        _write_month_partition(tmp_path, 2024, 2, present, ["000001.SZ", "600000.SH"])

        report = coverage_report(
            "20240201",
            "20240210",
            base_dir=tmp_path,
            trade_dates=expected,
        )
        assert report["missing_ranges"] == [
            ["2024-02-03", "2024-02-04"],
            ["2024-02-07", "2024-02-08"],
        ]
        # 6 days × 2 codes
        assert report["per_month_rows"]["2024-02"] == 12

    def test_full_coverage_empty_missing(self, tmp_path: Path) -> None:
        expected = [date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5)]
        _write_month_partition(tmp_path, 2024, 3, expected, ["000001.SZ"])
        report = coverage_report(
            "20240301",
            "20240305",
            base_dir=tmp_path,
            trade_dates=expected,
        )
        assert report["missing_ranges"] == []
        assert report["n_present_days"] == 3
