"""test_intraday_factor_base.py：IntradayFactor 抽象基类与 validate 统计
test_intraday_vwap_factor.py：VwapDeviation 内建因子列与首 bar/跨股票行为
test_intraday_demo.py：Momentum1Min demo 因子导入/计算/跨日不串
test_intraday_evaluation.py：分钟 IC 分析分段、日度 IC 与空输入
test_intraday_preprocessing.py：fill_missing_bars 与 clip_outliers 预处理管线
"""

import unittest.mock as mock
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest

from factorzen.builtin_factors.intraday.momentum_1min import Momentum1Min
from factorzen.builtin_factors.intraday.vwap_deviation import VwapDeviation
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

def test_cannot_instantiate_abstract__intraday_factor_base():
    """无法直接实例化抽象类 IntradayFactor。"""
    with pytest.raises(TypeError, match="abstract"):
        IntradayFactor()  # type: ignore[abstract]

def test_can_instantiate_concrete():
    """实现所有抽象方法后可以实例化。"""
    factor = _ConcreteFactor()
    assert factor.name == "test_factor"
    assert isinstance(factor, IntradayFactor)

def test_compute_is_abstract():
    """未实现 compute() 的子类无法实例化。"""

    @dataclass
    class _BadFactor(IntradayFactor):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _BadFactor()  # type: ignore[abstract]

# ── 默认属性 ────────────────────────────────────────────────────────────────

def test_default_attributes__intraday_factor_base():
    """默认属性值与预期一致。"""
    factor = _ConcreteFactor()
    assert factor.required_data == ["minute"]
    assert factor.lookback_bars == 500
    assert factor.description == "Test factor for unit testing"

# ── validate() ──────────────────────────────────────────────────────────────

def test_validate_empty():
    """空 DataFrame 返回 error。"""
    factor = _ConcreteFactor()
    result = factor.validate(pl.DataFrame())
    assert result["error"] == "Empty DataFrame"

def test_validate_normal():
    """正常因子结果返回正确的统计信息。"""
    factor = _ConcreteFactor()
    result = factor.validate(_make_result([1.0, 2.0, 3.0]))
    assert result["coverage"] == 1.0
    assert result["n_stocks"] == 3

def test_validate_nulls():
    """含 null 值时 coverage 不为 1。"""
    factor = _ConcreteFactor()
    result = factor.validate(_make_result([1.0, None, 3.0]))
    assert result["null_count"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)

def test_validate_inf():
    """含 inf 值时 inf_count > 0。"""
    factor = _ConcreteFactor()
    result = factor.validate(_make_result([1.0, float("inf")]))
    assert result["inf_count"] == 1

def test_validate_missing_factor_column():
    """缺少 factor_value 列时 null_count 和 inf_count 为 0。"""
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

def test_no_cross_stock():
    factor = VwapDeviation()
    result = factor.compute(_make_ctx(_make_minute_df()))
    assert set(result["ts_code"].unique().to_list()) == {"000001.SZ", "000002.SZ"}

def test_first_bar_zero():
    """第一根 bar 时 VWAP == close，偏离为 0。"""
    factor = VwapDeviation()
    result = factor.compute(_make_ctx(_make_minute_df()))
    first = (
        result.filter(pl.col("ts_code") == "000001.SZ")
        .sort("trade_time")
        .head(1)["factor_value"][0]
    )
    assert abs(first) < 1e-9

def test_registered():
    from factorzen.intraday.factors.registry import get_factor

    assert get_factor("vwap_deviation") is VwapDeviation

# ==== 来自 test_intraday_demo.py ====
# ── Import & Class Structure ─────────────────────────────────────────────────

def test_default_attributes__intraday_demo():
    """默认属性与预期一致。"""
    factor = Momentum1Min()
    assert factor.name == "momentum_1min"
    assert factor.bar_size == "1min"
    assert factor.frequency == "minute"
    assert factor.lookback_bars == 6
    assert factor.description == "5-bar momentum: close(t) / close(t-5) - 1"
    assert factor.required_data == ["minute"]

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

def test_compute_returns_correct_schema():
    """compute() 返回包含 trade_time, ts_code, factor_value 的 DataFrame。"""
    factor = Momentum1Min()
    mock_data = _make_mock_minute(n_bars=20)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    assert isinstance(result, pl.DataFrame)
    assert "trade_time" in result.columns
    assert "ts_code" in result.columns
    assert "factor_value" in result.columns
    assert result.height > 0

def test_compute_factor_range():
    """factor_value 在合理范围内（正值，因模拟数据持续上涨）。"""
    factor = Momentum1Min()
    mock_data = _make_mock_minute(n_bars=20)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    assert result["factor_value"].min() > -1.0
    assert result["factor_value"].max() < 10.0
    assert result["factor_value"].mean() > 0

def test_compute_multi_stock():
    """多股票同时计算时每只股票独立计算。"""
    factor = Momentum1Min()
    ts_codes = ["000001.SZ", "000002.SZ", "000004.SZ"]
    mock_data = _make_mock_minute(n_bars=20, ts_codes=ts_codes)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    codes_in_result = result["ts_code"].unique().to_list()
    for code in ts_codes:
        assert code in codes_in_result

def test_compute_filters_nulls():
    """前 5 根 bar 的 null 值已被过滤。"""
    factor = Momentum1Min()
    mock_data = _make_mock_minute(n_bars=20)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    assert result["factor_value"].null_count() == 0

def test_validate_returns_stats():
    """validate() 返回覆盖率等统计信息。"""
    factor = Momentum1Min()
    mock_data = _make_mock_minute(n_bars=20)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    stats = factor.validate(result)
    assert "coverage" in stats
    assert stats["coverage"] == 1.0
    assert stats["n_stocks"] == 1

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

def test_momentum_does_not_cross_trading_days():
    """5-bar 动量不得跨交易日：每个交易日前 5 根 bar 的 factor_value 应为 null（被过滤掉）。

    否则次日开盘首根 bar 会用前一日尾盘价算动量，把隔夜跳空当成日内动量（未来函数式污染）。
    """
    factor = Momentum1Min()
    result = factor.compute(_MockContext(_minute_data=_make_two_day_minute(bars_per_day=10)))

    # trade_time 字符串前 10 位是日期
    result = result.with_columns(pl.col("trade_time").cast(pl.Utf8).str.slice(0, 10).alias("_d"))
    per_day = {d[0]: sub.height for d, sub in result.group_by("_d")}
    # 每日 10 根，前 5 根 null 被过滤 → 每日各剩 5 根有效值；共 2 天 → 10 行
    assert per_day.get("2026-05-14") == 5, per_day
    assert per_day.get("2026-05-15") == 5, per_day  # 次日不会因跨日多出有效值
    assert result.height == 10

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

def test_ic_is_finite(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert np.isfinite(result.ic_mean)
    assert np.isfinite(result.ic_std)

def test_daily_ic_has_correct_dates(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    assert not result.daily_ic.is_empty()
    # 5 trading days -> 5 rows in daily_ic
    assert result.daily_ic.shape[0] == 5

def test_summary_string(intraday_data):
    factor_df, ret_df = intraday_data
    result = compute_intraday_rank_ic(factor_df, ret_df)
    text = result.summary()
    assert "Intraday IC" in text
    assert "IC Mean" in text

def test_empty_input_returns_zeros():
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

    def test_fill_null_within_group(self):
        """同股票内 null 值被前一 bar 的 factor_value 填充。"""
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

    def test_fill_cross_group_boundary(self):
        """forward-fill 不应跨股票。"""
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

    def test_leading_null_remains_null(self):
        """股票第一个 bar 为 null 时，forward_fill 无法填充（无先序值）。"""
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

    def test_retains_other_columns(self):
        """填充操作不应丢弃原有列。"""
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

    def test_all_present_no_change(self):
        """无缺失值时 DataFrame 不变。"""
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

    def test_fill_missing_bars_does_not_cross_trading_day_boundary(self):
        """forward-fill 不应跨交易日。"""
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

    def test_clip_both_ends(self):
        """上下同时截尾：超出分位数界的值被 clamp。"""
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

    def test_clip_lower_only(self):
        """仅截取下界：上界 100% 不起作用。"""
        df = pl.DataFrame(
            {
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 10.0],
            }
        )
        result = clip_outliers(df, lower_pct=20.0, upper_pct=100.0)
        clipped = sorted(result["factor_value"].to_list())
        assert clipped[-1] == 10.0

    def test_clip_upper_only(self):
        """仅截取上界：下界 0% 不起作用。"""
        df = pl.DataFrame(
            {
                "factor_value": [-100.0, 1.0, 2.0, 3.0, 10.0],
            }
        )
        result = clip_outliers(df, lower_pct=0.0, upper_pct=80.0)
        clipped = sorted(result["factor_value"].to_list())
        assert clipped[0] == -100.0

    def test_default_bounds_no_clip_on_normal_data(self):
        """默认 1%/99% 分位数：正常数据不应被截。"""
        df = pl.DataFrame(
            {
                "factor_value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = clip_outliers(df)
        pl_testing.assert_frame_equal(result, df)

    def test_clip_preserves_other_columns(self):
        """截尾不应丢弃原有列。"""
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

    def test_single_value_no_clip(self):
        """单一值不触发截尾。"""
        df = pl.DataFrame({"factor_value": [42.0]})
        result = clip_outliers(df)
        assert result["factor_value"][0] == 42.0

# ── IntradayPreprocessingPipeline ───────────────────────────────────────────

class TestIntradayPreprocessingPipeline:
    """验证预处理管线的构造、配置和 run() 行为。"""

    def test_default_config(self):
        """默认配置：fill_missing 和 clip_outliers 均开启。"""
        pipe = IntradayPreprocessingPipeline()
        assert pipe.do_fill_missing is True
        assert pipe.do_clip_outliers is True
        assert pipe.clip_lower_pct == 1.0
        assert pipe.clip_upper_pct == 99.0

    def test_custom_config(self):
        """自定义分位数参数正确存储。"""
        pipe = IntradayPreprocessingPipeline(
            do_fill_missing=False,
            clip_lower_pct=5.0,
            clip_upper_pct=95.0,
        )
        assert pipe.do_fill_missing is False
        assert pipe.clip_lower_pct == 5.0
        assert pipe.clip_upper_pct == 95.0

    def test_run_produces_factor_clean(self):
        """run() 必须产出 factor_clean 列。"""
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

    def test_run_with_missing_and_outliers(self):
        """同时处理缺失和异常值。"""
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

    def test_run_skip_fill(self):
        """do_fill_missing=False 时跳过填充。"""
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

    def test_run_skip_clip(self):
        """do_clip_outliers=False 时跳过截尾。"""
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
