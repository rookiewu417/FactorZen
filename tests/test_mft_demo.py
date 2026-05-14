"""测试 MFT Demo 因子 Momentum1Min。"""

from dataclasses import dataclass, field

import polars as pl
import pytest

from intraday.factors.base import MFTFactor
from intraday.factors.demo.momentum_1min import Momentum1Min

# ── Import & Class Structure ─────────────────────────────────────────────────

def test_import_clean():
    """Momentum1Min 可正常导入。"""
    assert Momentum1Min is not None
    assert Momentum1Min.__name__ == "Momentum1Min"


def test_extends_mftfactor():
    """Momentum1Min 继承自 MFTFactor。"""
    factor = Momentum1Min()
    assert isinstance(factor, MFTFactor)


def test_cannot_instantiate_abstract():
    """MFTFactor 本身不可直接实例化。"""
    with pytest.raises(TypeError, match="abstract"):
        MFTFactor()  # type: ignore[abstract]


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
    assert hasattr(Momentum1Min, '__dataclass_fields__')


# ── compute() 结构测试 ───────────────────────────────────────────────────────

@dataclass
class _MockContext:
    """模拟 MFTDataContext，提供 minute LazyFrame。"""
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
            close = base_close + i * 1.0  # 每 bar 涨 1
            rows.append({
                "trade_time": trade_time,
                "ts_code": code,
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "vol": 1000.0,
                "amount": close * 1000.0,
            })
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
    assert result.height > 0  # 至少有一行有效数据


def test_compute_factor_range():
    """factor_value 在合理范围内（正值，因模拟数据持续上涨）。"""
    factor = Momentum1Min()
    mock_data = _make_mock_minute(n_bars=20)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    assert result["factor_value"].min() > -1.0  # 至少不是 -100%
    assert result["factor_value"].max() < 10.0  # 不会出现极端值
    # 持续上涨场景下 factor_value 应为正
    assert result["factor_value"].mean() > 0


def test_compute_multi_stock():
    """多股票同时计算时每只股票独立计算。"""
    factor = Momentum1Min()
    ts_codes = ["000001.SZ", "000002.SZ", "000004.SZ"]
    mock_data = _make_mock_minute(n_bars=20, ts_codes=ts_codes)
    ctx = _MockContext(_minute_data=mock_data)
    result = factor.compute(ctx)
    # 每只股票都应出现在结果中
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
