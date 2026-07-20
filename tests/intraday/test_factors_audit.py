"""合并自: test_context_audit.py, test_intraday_factors.py
目标: test_factors_audit.py

--- 来源 test_context_audit.py ---
test_intraday_data_context.py：IntradayDataContext 构造、懒加载、universe 过滤
test_intraday_audit.py：分钟覆盖/时间戳普查/与日频对账/标签惯例推断

--- 来源 test_intraday_factors.py ---
test_intraday_factor_base.py：IntradayFactor 抽象基类与 validate 统计
test_intraday_vwap_factor.py：VwapDeviation 内建因子列与首 bar/跨股票行为
test_intraday_demo.py：Momentum1Min demo 因子导入/计算/跨日不串
test_intraday_evaluation.py：分钟 IC 分析分段、日度 IC 与空输入
test_intraday_preprocessing.py：fill_missing_bars 与 clip_outliers 预处理管线
"""

from __future__ import annotations

import unittest.mock as mock
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest

from factorzen.builtin_factors.intraday.momentum_1min import Momentum1Min
from factorzen.builtin_factors.intraday.vwap_deviation import VwapDeviation
from factorzen.intraday.audit import (
    coverage_report,
    infer_label_convention,
    reconcile_with_daily,
    timestamp_census,
)
from factorzen.intraday.data.context import IntradayDataContext
from factorzen.intraday.evaluation.ic_analysis import (
    _assign_segment,
    compute_intraday_rank_ic,
)
from factorzen.intraday.factors.base import IntradayFactor
from factorzen.intraday.preprocessing.pipeline import (
    IntradayPreprocessingPipeline,
    clip_outliers,
    fill_missing_bars,
)


# ==== 来自 test_context_audit.py ====
# ==== 来自 test_intraday_data_context.py ====
@pytest.fixture(autouse=True)
def mock_prev_trade_date(monkeypatch):
    """Keep these unit tests offline; calendar integration is covered separately."""

    def _fake_prev_trade_date(d: str, n: int = 1) -> date:
        assert n == 5
        return {
            "20260514": date(2026, 5, 7),
            "20260105": date(2025, 12, 25),
        }[d]

    monkeypatch.setattr("factorzen.intraday.data.context.prev_trade_date", _fake_prev_trade_date)

# ── 构造与默认值 ──────────────────────────────────────────────────────────


def test_intraday_context_construction_suite():
    """基本构造：默认属性与预期一致。；自定义参数正确存储。；expanded_start 向前扩展 5 天。；minute 不在 required_data 中时，访问 .minute 抛出 ValueError。；访问 .minute 触发 load_parquet 调用，结果缓存到 _minute。；传入 universe 时对 LazyFrame 添加 .is_in() 过滤。；load_all() 触发 minute 数据加载。"""
    # -- 原 test_construction_defaults --
    def _section_0_test_construction_defaults():
        ctx = IntradayDataContext("20260514", "20260514")
        assert ctx.start == "20260514"
        assert ctx.end == "20260514"
        assert ctx.bar_size == "1min"
        assert ctx.required_data == ["minute"]
        assert ctx.universe is None
        assert ctx.max_bars == 10_000
        assert ctx._minute is None

    _section_0_test_construction_defaults()

    # -- 原 test_construction_custom --
    def _section_1_test_construction_custom():
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

    _section_1_test_construction_custom()

    # -- 原 test_expanded_start --
    def _section_2_test_expanded_start():
        ctx = IntradayDataContext("20260514", "20260514")
        assert ctx.expanded_start == "20260507"

        ctx2 = IntradayDataContext("20260105", "20260110")
        assert ctx2.expanded_start == "20251225"

    _section_2_test_expanded_start()

    # -- 原 test_minute_not_declared_raises --
    def _section_3_test_minute_not_declared_raises():
        ctx = IntradayDataContext("20260514", "20260514", required_data=["other"])
        with pytest.raises(ValueError, match="minute data not declared"):
            _ = ctx.minute

    _section_3_test_minute_not_declared_raises()

    # -- 原 test_minute_lazy_loading --
    def _section_4_test_minute_lazy_loading():
        synthetic = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"],
                "ts_code": ["000001.SZ"],
            }
        ).lazy()

        with patch("factorzen.intraday.data.context.load_parquet", return_value=synthetic) as mock_load:
            ctx = IntradayDataContext("20260514", "20260514")
            assert ctx._minute is None

            result = ctx.minute
            mock_load.assert_called_once()
            # 存储 data_type 必须按 freq 命名空间 minute_{bar_size}（对齐 loader.fetch_minute，
            # 否则读到空的 "minute" 分区——双路径漂移）。
            assert mock_load.call_args.args[0] == "minute_1min"
            assert isinstance(result, pl.LazyFrame)
            assert ctx._minute is not None

            # 再次访问应命中缓存，不重复调用
            _ = ctx.minute
            mock_load.assert_called_once()

    _section_4_test_minute_lazy_loading()

    # -- 原 test_minute_universe_filter --
    def _section_5_test_minute_universe_filter():
        synthetic = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00", "2026-05-14 09:31:00"],
                "ts_code": ["000001.SZ", "999999.SZ"],
            }
        ).lazy()

        with patch("factorzen.intraday.data.context.load_parquet", return_value=synthetic):
            ctx = IntradayDataContext("20260514", "20260514", universe=["000001.SZ"])
            lf = ctx.minute
            collected = lf.collect()
            assert collected.height == 1
            assert collected["ts_code"][0] == "000001.SZ"

    _section_5_test_minute_universe_filter()

    # -- 原 test_load_all --
    def _section_6_test_load_all():
        synthetic = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"],
                "ts_code": ["000001.SZ"],
            }
        ).lazy()

        with patch("factorzen.intraday.data.context.load_parquet", return_value=synthetic):
            ctx = IntradayDataContext("20260514", "20260514")
            assert ctx._minute is None
            ctx.load_all()
            assert ctx._minute is not None

    _section_6_test_load_all()


# ── expanded_start ────────────────────────────────────────────────────────


# ── required_data 校验 ────────────────────────────────────────────────────


# ── 惰性加载与缓存 ────────────────────────────────────────────────────────


# ── load_all ──────────────────────────────────────────────────────────────


# ==== 来自 test_intraday_audit.py ====
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


def test_timestamp_census_and_reconcile_suite():
    """test_groups_by_year_and_board；test_healthy_multipliers；故意构造 ×1 错误单位帧，应检出 multiplier≈1。"""
    # -- 原 test_groups_by_year_and_board --
    def _section_0_test_groups_by_year_and_board():
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

    _section_0_test_groups_by_year_and_board()

    # -- 原 test_healthy_multipliers --
    def _section_1_test_healthy_multipliers():
        recon = reconcile_with_daily(_minute_healthy(), _daily_healthy())
        assert recon.height == 1
        assert recon["vol_multiplier"][0] == pytest.approx(100.0)
        assert recon["amount_multiplier"][0] == pytest.approx(1000.0)
        assert recon["open_match"][0] is True
        assert recon["close_match"][0] is True
        assert recon["high_match"][0] is True
        assert recon["low_match"][0] is True

    _section_1_test_healthy_multipliers()

    # -- 原 test_detects_wrong_unit_frame --
    def _section_2_test_detects_wrong_unit_frame():
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

    _section_2_test_detects_wrong_unit_frame()

class TestInferLabelConvention:
    def test_infer_label_convention_suite(self):
        """test_end_convention；test_start_convention；test_after_1500_flag"""
        # -- 原 test_end_convention --
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

        # -- 原 test_start_convention --
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

        # -- 原 test_after_1500_flag --
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
    def test_coverage_report_suite(self, tmp_path):
        """注入 trade_dates，挖掉一段连续缺失，验证区间合并。；test_two_disjoint_gaps_merge_separately；test_full_coverage_empty_missing"""
        # -- 原 test_detects_missing_range_and_merges --
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

        # -- 原 test_two_disjoint_gaps_merge_separately --
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

        # -- 原 test_full_coverage_empty_missing --
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


# ==== 来自 test_intraday_factors.py ====
# ==== 来自 test_intraday_factor_base.py ====
# ── helpers ─────────────────────────────────────────────────────────────────

@dataclass
class _ConcreteFactor(IntradayFactor):
    """测试用的具体因子类。"""

    name: str = "test_factor"
    description: str = "Test factor for unit testing"

    def compute(self, ctx):
        return pl.DataFrame(
            schema={"trade_time": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64}
        )

def _make_result(values: list[float], ts_codes: list[str] | None = None) -> pl.DataFrame:
    """构建模拟的 factor 结果 DataFrame。"""
    ts_codes = ts_codes or [f"00000{i}.SZ" for i in range(len(values))]
    return pl.DataFrame(
        {
            "trade_time": ["2026-05-14 09:30:00"] * len(values),
            "ts_code": ts_codes,
            "factor_value": values,
        }
    )

# ── ABC 约束 ────────────────────────────────────────────────────────────────

def test_intraday_factor_base_suite():
    """无法直接实例化抽象类 IntradayFactor。；实现所有抽象方法后可以实例化。；未实现 compute() 的子类无法实例化。；默认属性值与预期一致。；空 DataFrame 返回 error。；正常因子结果返回正确的统计信息。；含 null 值时 coverage 不为 1。；含 inf 值时 inf_count > 0。；缺少 factor_value 列时 null_count 和 inf_count 为 0。"""
    # -- 原 test_cannot_instantiate_abstract__intraday_factor_base --
    def _section_0_test_cannot_instantiate_abstract__intraday_factor_base():
        with pytest.raises(TypeError, match="abstract"):
            IntradayFactor()  # type: ignore[abstract]

    _section_0_test_cannot_instantiate_abstract__intraday_factor_base()

    # -- 原 test_can_instantiate_concrete --
    def _section_1_test_can_instantiate_concrete():
        factor = _ConcreteFactor()
        assert factor.name == "test_factor"
        assert isinstance(factor, IntradayFactor)

    _section_1_test_can_instantiate_concrete()

    # -- 原 test_compute_is_abstract --
    def _section_2_test_compute_is_abstract():
        class _BadFactor(IntradayFactor):
            pass

        with pytest.raises(TypeError, match="abstract"):
            _BadFactor()  # type: ignore[abstract]

    _section_2_test_compute_is_abstract()

    # -- 原 test_default_attributes__intraday_factor_base --
    def _section_3_test_default_attributes__intraday_factor_base():
        factor = _ConcreteFactor()
        assert factor.required_data == ["minute"]
        assert factor.lookback_bars == 500
        assert factor.description == "Test factor for unit testing"

    _section_3_test_default_attributes__intraday_factor_base()

    # -- 原 test_validate_empty --
    def _section_4_test_validate_empty():
        factor = _ConcreteFactor()
        result = factor.validate(pl.DataFrame())
        assert result["error"] == "Empty DataFrame"

    _section_4_test_validate_empty()

    # -- 原 test_validate_normal --
    def _section_5_test_validate_normal():
        factor = _ConcreteFactor()
        result = factor.validate(_make_result([1.0, 2.0, 3.0]))
        assert result["coverage"] == 1.0
        assert result["n_stocks"] == 3

    _section_5_test_validate_normal()

    # -- 原 test_validate_nulls --
    def _section_6_test_validate_nulls():
        factor = _ConcreteFactor()
        result = factor.validate(_make_result([1.0, None, 3.0]))
        assert result["null_count"] == 1
        assert result["coverage"] == pytest.approx(2 / 3)

    _section_6_test_validate_nulls()

    # -- 原 test_validate_inf --
    def _section_7_test_validate_inf():
        factor = _ConcreteFactor()
        result = factor.validate(_make_result([1.0, float("inf")]))
        assert result["inf_count"] == 1

    _section_7_test_validate_inf()

    # -- 原 test_validate_missing_factor_column --
    def _section_8_test_validate_missing_factor_column():
        factor = _ConcreteFactor()
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"],
                "ts_code": ["000001.SZ"],
            }
        )
        result = factor.validate(df)
        assert result["null_count"] == 0
        assert result["inf_count"] == 0

    _section_8_test_validate_missing_factor_column()


# ── 默认属性 ────────────────────────────────────────────────────────────────


# ── validate() ──────────────────────────────────────────────────────────────


# ==== 来自 test_intraday_vwap_factor.py ====
def _make_ctx(df: pl.DataFrame):
    ctx = mock.MagicMock(spec=IntradayDataContext)
    ctx.minute = df.lazy()
    return ctx

def _make_minute_df(n_bars: int = 20) -> pl.DataFrame:
    base = datetime(2026, 5, 16, 9, 30)
    rows = []
    for ts in ["000001.SZ", "000002.SZ"]:
        for i in range(n_bars):
            price = 10.0 + i * 0.05
            vol = 1000.0 + i * 10
            rows.append(
                {
                    "ts_code": ts,
                    "trade_time": base + timedelta(minutes=i),
                    "close": price,
                    "vol": vol,
                    "amount": price * vol,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("trade_time").cast(pl.Datetime))

def test_intraday_demo_factor_suite():
    """test_no_cross_stock；第一根 bar 时 VWAP == close，偏离为 0。；test_registered；默认属性与预期一致。；compute() 返回包含 trade_time, ts_code, factor_value 的 DataFrame。；factor_value 在合理范围内（正值，因模拟数据持续上涨）。；多股票同时计算时每只股票独立计算。；前 5 根 bar 的 null 值已被过滤。；validate() 返回覆盖率等统计信息。；5-bar 动量不得跨交易日：每个交易日前 5 根 bar 的 factor_value 应为 null（被过滤掉）。"""
    # -- 原 test_no_cross_stock --
    def _section_0_test_no_cross_stock():
        factor = VwapDeviation()
        result = factor.compute(_make_ctx(_make_minute_df()))
        assert set(result["ts_code"].unique().to_list()) == {"000001.SZ", "000002.SZ"}

    _section_0_test_no_cross_stock()

    # -- 原 test_first_bar_zero --
    def _section_1_test_first_bar_zero():
        factor = VwapDeviation()
        result = factor.compute(_make_ctx(_make_minute_df()))
        first = (
            result.filter(pl.col("ts_code") == "000001.SZ")
            .sort("trade_time")
            .head(1)["factor_value"][0]
        )
        assert abs(first) < 1e-9

    _section_1_test_first_bar_zero()

    # -- 原 test_registered --
    def _section_2_test_registered():
        from factorzen.intraday.factors.registry import get_factor

        assert get_factor("vwap_deviation") is VwapDeviation

    _section_2_test_registered()

    # -- 原 test_default_attributes__intraday_demo --
    def _section_3_test_default_attributes__intraday_demo():
        factor = Momentum1Min()
        assert factor.name == "momentum_1min"
        assert factor.bar_size == "1min"
        assert factor.frequency == "minute"
        assert factor.lookback_bars == 6
        assert factor.description == "5-bar momentum: close(t) / close(t-5) - 1"
        assert factor.required_data == ["minute"]

    _section_3_test_default_attributes__intraday_demo()

    # -- 原 test_compute_returns_correct_schema --
    def _section_4_test_compute_returns_correct_schema():
        factor = Momentum1Min()
        mock_data = _make_mock_minute(n_bars=20)
        ctx = _MockContext(_minute_data=mock_data)
        result = factor.compute(ctx)
        assert isinstance(result, pl.DataFrame)
        assert "trade_time" in result.columns
        assert "ts_code" in result.columns
        assert "factor_value" in result.columns
        assert result.height > 0

    _section_4_test_compute_returns_correct_schema()

    # -- 原 test_compute_factor_range --
    def _section_5_test_compute_factor_range():
        factor = Momentum1Min()
        mock_data = _make_mock_minute(n_bars=20)
        ctx = _MockContext(_minute_data=mock_data)
        result = factor.compute(ctx)
        assert result["factor_value"].min() > -1.0
        assert result["factor_value"].max() < 10.0
        assert result["factor_value"].mean() > 0

    _section_5_test_compute_factor_range()

    # -- 原 test_compute_multi_stock --
    def _section_6_test_compute_multi_stock():
        factor = Momentum1Min()
        ts_codes = ["000001.SZ", "000002.SZ", "000004.SZ"]
        mock_data = _make_mock_minute(n_bars=20, ts_codes=ts_codes)
        ctx = _MockContext(_minute_data=mock_data)
        result = factor.compute(ctx)
        codes_in_result = result["ts_code"].unique().to_list()
        for code in ts_codes:
            assert code in codes_in_result

    _section_6_test_compute_multi_stock()

    # -- 原 test_compute_filters_nulls --
    def _section_7_test_compute_filters_nulls():
        factor = Momentum1Min()
        mock_data = _make_mock_minute(n_bars=20)
        ctx = _MockContext(_minute_data=mock_data)
        result = factor.compute(ctx)
        assert result["factor_value"].null_count() == 0

    _section_7_test_compute_filters_nulls()

    # -- 原 test_validate_returns_stats --
    def _section_8_test_validate_returns_stats():
        factor = Momentum1Min()
        mock_data = _make_mock_minute(n_bars=20)
        ctx = _MockContext(_minute_data=mock_data)
        result = factor.compute(ctx)
        stats = factor.validate(result)
        assert "coverage" in stats
        assert stats["coverage"] == 1.0
        assert stats["n_stocks"] == 1

    _section_8_test_validate_returns_stats()

    # -- 原 test_momentum_does_not_cross_trading_days --
    def _section_9_test_momentum_does_not_cross_trading_days():
        factor = Momentum1Min()
        result = factor.compute(_MockContext(_minute_data=_make_two_day_minute(bars_per_day=10)))

        # trade_time 字符串前 10 位是日期
        result = result.with_columns(pl.col("trade_time").cast(pl.Utf8).str.slice(0, 10).alias("_d"))
        per_day = {d[0]: sub.height for d, sub in result.group_by("_d")}
        # 每日 10 根，前 5 根 null 被过滤 → 每日各剩 5 根有效值；共 2 天 → 10 行
        assert per_day.get("2026-05-14") == 5, per_day
        assert per_day.get("2026-05-15") == 5, per_day  # 次日不会因跨日多出有效值
        assert result.height == 10

    _section_9_test_momentum_does_not_cross_trading_days()


# ==== 来自 test_intraday_demo.py ====
# ── Import & Class Structure ─────────────────────────────────────────────────


# ── compute() 结构测试 ───────────────────────────────────────────────────────

@dataclass
class _MockContext:
    """模拟 IntradayDataContext，提供 minute LazyFrame。"""

    _minute_data: pl.DataFrame = field(default_factory=pl.DataFrame)

    @property
    def minute(self) -> pl.LazyFrame:
        return self._minute_data.lazy()

def _make_mock_minute(n_bars: int = 20, ts_codes: list[str] | None = None) -> pl.DataFrame:
    """构造模拟的 1 分钟 bar 数据（按 ts_code + trade_time 排序）。"""
    if ts_codes is None:
        ts_codes = ["000001.SZ"]
    rows = []
    for code in ts_codes:
        base_close = 100.0 + hash(code) % 100
        for i in range(n_bars):
            hour = 9 + (i + 30) // 60
            minute = (i + 30) % 60
            trade_time = f"2026-05-14 {hour:02d}:{minute:02d}:00"
            close = base_close + i * 1.0
            rows.append(
                {
                    "trade_time": trade_time,
                    "ts_code": code,
                    "open": close - 0.5,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "vol": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    df = pl.DataFrame(rows)
    return df.sort(["ts_code", "trade_time"])


def _make_two_day_minute(code: str = "000001.SZ", bars_per_day: int = 10) -> pl.DataFrame:
    """构造两个交易日、每日 bars_per_day 根的分钟数据（trade_time 为字符串，模拟隔夜缺口）。"""
    rows = []
    for day in ("2026-05-14", "2026-05-15"):
        for i in range(bars_per_day):
            hour = 9 + (i + 30) // 60
            minute = (i + 30) % 60
            close = 100.0 + i * 1.0 + (0.0 if day == "2026-05-14" else 50.0)  # 隔夜跳空
            rows.append({"trade_time": f"{day} {hour:02d}:{minute:02d}:00", "ts_code": code,
                         "open": close, "high": close, "low": close, "close": close,
                         "vol": 1000.0, "amount": close * 1000.0})
    return pl.DataFrame(rows).sort(["ts_code", "trade_time"])


# ==== 来自 test_intraday_evaluation.py ====
def _make_intraday_data(n_stocks: int = 30, n_days: int = 5, seed: int = 0):
    """Generates synthetic minute-bar factor + return DataFrames."""
    rng = np.random.default_rng(seed)
    from datetime import timedelta

    base_times = [
        datetime(2024, 1, 2) + timedelta(days=d, hours=9, minutes=30 + m)
        for d in range(n_days)
        for m in range(0, 90, 5)  # every 5 min, 09:30–11:00 (18 bars/day)
    ]
    minutes_per_day = base_times

    factor_rows = []
    ret_rows = []
    for ts in minutes_per_day:
        for i in range(n_stocks):
            code = f"{i:06d}.SH"
            factor_rows.append(
                {
                    "trade_time": ts,
                    "ts_code": code,
                    "factor_value": float(rng.standard_normal()),
                }
            )
            ret_rows.append(
                {
                    "trade_time": ts,
                    "ts_code": code,
                    "fwd_ret_1bar": float(rng.standard_normal() * 0.002),
                }
            )

    return pl.DataFrame(factor_rows), pl.DataFrame(ret_rows)

@pytest.fixture()
def intraday_data():
    return _make_intraday_data()

def test_intraday_ic_suite(intraday_data):
    """test_ic_is_finite；test_daily_ic_has_correct_dates；test_summary_string；test_empty_input_returns_zeros"""
    # -- 原 test_ic_is_finite --
    def _section_0_test_ic_is_finite(intraday_data):
        factor_df, ret_df = intraday_data
        result = compute_intraday_rank_ic(factor_df, ret_df)
        assert np.isfinite(result.ic_mean)
        assert np.isfinite(result.ic_std)

    _section_0_test_ic_is_finite(intraday_data)

    # -- 原 test_daily_ic_has_correct_dates --
    def _section_1_test_daily_ic_has_correct_dates(intraday_data):
        factor_df, ret_df = intraday_data
        result = compute_intraday_rank_ic(factor_df, ret_df)
        assert not result.daily_ic.is_empty()
        # 5 trading days -> 5 rows in daily_ic
        assert result.daily_ic.shape[0] == 5

    _section_1_test_daily_ic_has_correct_dates(intraday_data)

    # -- 原 test_summary_string --
    def _section_2_test_summary_string(intraday_data):
        factor_df, ret_df = intraday_data
        result = compute_intraday_rank_ic(factor_df, ret_df)
        text = result.summary()
        assert "Intraday IC" in text
        assert "IC Mean" in text

    _section_2_test_summary_string(intraday_data)

    # -- 原 test_empty_input_returns_zeros --
    def _section_3_test_empty_input_returns_zeros():
        factor_df = pl.DataFrame(
            {
                "trade_time": pl.Series([], dtype=pl.Datetime),
                "ts_code": pl.Series([], dtype=pl.Utf8),
                "factor_value": pl.Series([], dtype=pl.Float64),
            }
        )
        ret_df = pl.DataFrame(
            {
                "trade_time": pl.Series([], dtype=pl.Datetime),
                "ts_code": pl.Series([], dtype=pl.Utf8),
                "fwd_ret_1bar": pl.Series([], dtype=pl.Float64),
            }
        )
        result = compute_intraday_rank_ic(factor_df, ret_df)
        assert result.ic_mean == 0.0
        assert result.n_periods == 0

    _section_3_test_empty_input_returns_zeros()


def test_assign_segment_labels():
    df = pl.DataFrame(
        {
            "trade_time": [
                datetime(2024, 1, 2, 9, 30),  # open
                datetime(2024, 1, 2, 11, 0),  # midday
                datetime(2024, 1, 2, 14, 45),  # close
            ]
        }
    )
    result = _assign_segment(df, "trade_time")
    segs = result["segment"].to_list()
    assert segs[0] == "open"
    assert segs[1] == "midday"
    assert segs[2] == "close"

# ==== 来自 test_intraday_preprocessing.py ====
# ── fill_missing_bars ───────────────────────────────────────────────────────

class TestFillMissingBars:
    """验证 forward-fill 缺失 bar 的行为。"""

    def test_fill_missing_bars_suite(self):
        """同股票内 null 值被前一 bar 的 factor_value 填充。；forward-fill 不应跨股票。；股票第一个 bar 为 null 时，forward_fill 无法填充（无先序值）。；填充操作不应丢弃原有列。；无缺失值时 DataFrame 不变。；forward-fill 不应跨交易日。"""
        # -- 原 test_fill_null_within_group --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                    "2026-05-14 09:32:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None, 3.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        expected = [1.0, 1.0, 3.0]
        assert result["factor_value"].to_list() == expected

        # -- 原 test_fill_cross_group_boundary --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                    "2026-05-14 09:30:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "factor_value": [1.0, None, None],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        values = result["factor_value"].to_list()
        assert values[0] == 1.0
        assert values[1] == 1.0
        assert values[2] is None

        # -- 原 test_leading_null_remains_null --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [None, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        values = result["factor_value"].to_list()
        assert values[0] is None
        assert values[1] == 2.0

        # -- 原 test_retains_other_columns --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None],
                "volume": [100, 200],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        assert "volume" in result.columns
        assert result["volume"].to_list() == [100, 200]

        # -- 原 test_all_present_no_change --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)
        pl_testing.assert_frame_equal(result, df)

        # -- 原 test_fill_missing_bars_does_not_cross_trading_day_boundary --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2024-01-02 15:00:00",
                    "2024-01-03 09:30:00",
                    "2024-01-03 09:31:00",
                ],
                "ts_code": ["000001.SZ"] * 3,
                "factor_value": [1.5, None, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = fill_missing_bars(df)

        assert result["factor_value"].to_list() == [1.5, None, 2.0]


# ── clip_outliers ───────────────────────────────────────────────────────────

class TestClipOutliers:
    """验证分位数截尾行为。"""

    def test_clip_outliers_suite(self):
        """上下同时截尾：超出分位数界的值被 clamp。；仅截取下界：上界 100% 不起作用。；仅截取上界：下界 0% 不起作用。；默认 1%/99% 分位数：正常数据不应被截。；截尾不应丢弃原有列。；单一值不触发截尾。"""
        # -- 原 test_clip_both_ends --
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"] * 5,
                "ts_code": [f"00000{i}.SZ" for i in range(1, 6)],
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 200.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = clip_outliers(df, lower_pct=0.0, upper_pct=60.0)
        values = result["factor_value"].to_list()
        assert -100.0 in values
        assert 200.0 not in values
        assert all(v <= 200.0 for v in values)

        # -- 原 test_clip_lower_only --
        df = pl.DataFrame(
            {
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 10.0],
            }
        )
        result = clip_outliers(df, lower_pct=20.0, upper_pct=100.0)
        clipped = sorted(result["factor_value"].to_list())
        assert clipped[-1] == 10.0

        # -- 原 test_clip_upper_only --
        df = pl.DataFrame(
            {
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 10.0],
            }
        )
        result = clip_outliers(df, lower_pct=0.0, upper_pct=80.0)
        clipped = sorted(result["factor_value"].to_list())
        assert clipped[0] == -100.0

        # -- 原 test_default_bounds_no_clip_on_normal_data --
        df = pl.DataFrame(
            {
                "factor_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = clip_outliers(df)
        pl_testing.assert_frame_equal(result, df)

        # -- 原 test_clip_preserves_other_columns --
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"] * 3,
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "factor_value": [-50.0, 2.0, 50.0],
                "volume": [100, 200, 300],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = clip_outliers(df, lower_pct=33.0, upper_pct=67.0)
        assert "volume" in result.columns
        assert result["volume"].to_list() == [100, 200, 300]

        # -- 原 test_single_value_no_clip --
        df = pl.DataFrame({"factor_value": [42.0]})
        result = clip_outliers(df)
        assert result["factor_value"][0] == 42.0


# ── IntradayPreprocessingPipeline ───────────────────────────────────────────

class TestIntradayPreprocessingPipeline:
    """验证预处理管线的构造、配置和 run() 行为。"""

    def test_intraday_preprocessing_pipeline_suite(self):
        """默认配置：fill_missing 和 clip_outliers 均开启。；自定义分位数参数正确存储。；run() 必须产出 factor_clean 列。；同时处理缺失和异常值。；do_fill_missing=False 时跳过填充。；do_clip_outliers=False 时跳过截尾。"""
        # -- 原 test_default_config --
        pipe = IntradayPreprocessingPipeline()
        assert pipe.do_fill_missing is True
        assert pipe.do_clip_outliers is True
        assert pipe.clip_lower_pct == 1.0
        assert pipe.clip_upper_pct == 99.0

        # -- 原 test_custom_config --
        pipe = IntradayPreprocessingPipeline(
            do_fill_missing=False,
            clip_lower_pct=5.0,
            clip_upper_pct=95.0,
        )
        assert pipe.do_fill_missing is False
        assert pipe.clip_lower_pct == 5.0
        assert pipe.clip_upper_pct == 95.0

        # -- 原 test_run_produces_factor_clean --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, 2.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        result = IntradayPreprocessingPipeline().run(df)
        assert "factor_clean" in result.columns
        assert result["factor_clean"].to_list() == [1.0, 2.0]

        # -- 原 test_run_with_missing_and_outliers --
        df = pl.DataFrame(
            {
                "trade_time": [
                    "2026-05-14 09:30:00",
                    "2026-05-14 09:31:00",
                    "2026-05-14 09:32:00",
                ],
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None, 100.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        pipe = IntradayPreprocessingPipeline(clip_lower_pct=0.0, clip_upper_pct=50.0)
        result = pipe.run(df)
        assert "factor_clean" in result.columns
        assert result["factor_clean"].to_list() == [1.0, 1.0, 1.0]

        # -- 原 test_run_skip_fill --
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00", "2026-05-14 09:31:00"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value": [1.0, None],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        pipe = IntradayPreprocessingPipeline(do_fill_missing=False)
        result = pipe.run(df)
        assert result["factor_clean"].to_list() == [1.0, None]

        # -- 原 test_run_skip_clip --
        df = pl.DataFrame(
            {
                "trade_time": ["2026-05-14 09:30:00"],
                "ts_code": ["000001.SZ"],
                "factor_value": [999.0],
            }
        ).with_columns(pl.col("trade_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"))

        pipe = IntradayPreprocessingPipeline(do_clip_outliers=False)
        result = pipe.run(df)
        assert result["factor_clean"][0] == 999.0


