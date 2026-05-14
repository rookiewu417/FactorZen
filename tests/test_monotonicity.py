"""测试因子单调性：分组收益是否单调递增/递减。"""

import polars as pl
from daily.evaluation.advanced import compute_monotonicity, MonotonicityResult


def _make_strongly_monotonic_data() -> pl.DataFrame:
    """构造强单调数据：分位 1→10 收益严格递增。"""
    n = 100
    return pl.DataFrame({
        "ts_code": [f"s{i}" for i in range(n)],
        "trade_date": ["2026-01-05"] * n,
        "factor_value": [i / n for i in range(n)],          # [0, 1) 均匀分布
        "fwd_ret": [i / n * 0.1 for i in range(n)],         # 与因子值完全正相关
    })


def test_monotonicity_returns_result_object():
    """compute_monotonicity 返回 MonotonicityResult。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(
        df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10
    )
    assert isinstance(result, MonotonicityResult)


def test_monotonicity_strongly_positive():
    """强正相关数据 → monotonicity_score 接近 1.0。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(
        df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10
    )
    assert result.monotonicity_score > 0.5
    assert result.direction == "positive"


def test_monotonicity_group_means_monotonic():
    """分组均值应为单调递增。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(
        df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10
    )
    means = result.group_means
    assert len(means) == 10
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1], f"组 {i}→{i+1} 收益不单调"


def test_monotonicity_result_fields():
    """MonotonicityResult 包含必要字段。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(
        df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10
    )
    assert hasattr(result, "monotonicity_score")
    assert hasattr(result, "group_means")
    assert hasattr(result, "direction")
    assert isinstance(result.group_means, list)
    assert all(isinstance(m, float) for m in result.group_means)