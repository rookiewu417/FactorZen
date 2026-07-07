"""测试 Intraday Demo 因子 Momentum1Min。"""

from dataclasses import dataclass, field

import polars as pl
import pytest

from factorzen.builtin_factors.intraday.momentum_1min import Momentum1Min
from factorzen.intraday.factors.base import IntradayFactor

# ── Import & Class Structure ─────────────────────────────────────────────────


def test_import_clean():
    """Momentum1Min 可正常导入。"""
    assert Momentum1Min is not None
    assert Momentum1Min.__name__ == "Momentum1Min"


def test_extends_intraday_factor():
    """Momentum1Min 继承自 IntradayFactor。"""
    factor = Momentum1Min()
    assert isinstance(factor, IntradayFactor)


def test_cannot_instantiate_abstract():
    """IntradayFactor 本身不可直接实例化。"""
    with pytest.raises(TypeError, match="abstract"):
        IntradayFactor()  # type: ignore[abstract]


def test_default_attributes():
    """默认属性与预期一致。"""
    factor = Momentum1Min()
    assert factor.name == "momentum_1min"
    assert factor.bar_size == "1min"
    assert factor.frequency == "minute"
    assert factor.lookback_bars == 6
    assert factor.description == "5-bar momentum: close(t) / close(t-5) - 1"
    assert factor.required_data == ["minute"]


def test_is_dataclass():
    """Momentum1Min 是 dataclass，可按位置/关键字实例化。"""
    assert hasattr(Momentum1Min, "__dataclass_fields__")


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
