"""测试 IntradayFactor 抽象基类。"""

from dataclasses import dataclass

import polars as pl
import pytest

from intraday.factors.base import IntradayFactor

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


def test_cannot_instantiate_abstract():
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


def test_default_attributes():
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
