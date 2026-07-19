"""测试因子单调性：分组收益是否单调递增/递减。"""

import polars as pl

from factorzen.daily.evaluation.advanced import MonotonicityResult, compute_monotonicity


def _make_strongly_monotonic_data() -> pl.DataFrame:
    """构造强单调数据：分位 1→10 收益严格递增。"""
    n = 100
    return pl.DataFrame(
        {
            "ts_code": [f"s{i}" for i in range(n)],
            "trade_date": ["2026-01-05"] * n,
            "factor_value": [i / n for i in range(n)],  # [0, 1) 均匀分布
            "fwd_ret": [i / n * 0.1 for i in range(n)],  # 与因子值完全正相关
        }
    )


def test_monotonicity_returns_result_object():
    """compute_monotonicity 返回 MonotonicityResult。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10)
    assert isinstance(result, MonotonicityResult)


def test_monotonicity_strongly_positive():
    """强正相关数据 → monotonicity_score 接近 1.0。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10)
    assert result.monotonicity_score > 0.5
    assert result.direction == "positive"


def test_monotonicity_group_means_monotonic():
    """分组均值应为单调递增。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10)
    means = result.group_means
    assert len(means) == 10
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1], f"组 {i}→{i + 1} 收益不单调"


def test_monotonicity_result_fields():
    """MonotonicityResult 包含必要字段。"""
    df = _make_strongly_monotonic_data()
    result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10)
    assert hasattr(result, "monotonicity_score")
    assert hasattr(result, "group_means")
    assert hasattr(result, "direction")
    assert isinstance(result.group_means, list)
    assert all(isinstance(m, float) for m in result.group_means)


# ── group_daily_returns：报告层画分组净值/绩效的数据源 ────────────────────────


def test_group_daily_returns_matches_hand_computed_ground_truth():
    """逐日 × 分组收益对齐手算值。

    2 天 × 4 股 × 2 组，分组公式 ``(rank-1)*n_groups//max_rank``：
    因子 1,2,3,4 → rank 1,2,3,4 → G0={rank1,2}、G1={rank3,4}。
    收益按天独立给定，各组均值可手算，不依赖 group_means 反推（避免恒真）。
    """
    df = pl.DataFrame(
        {
            "ts_code": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "trade_date": ["2026-01-05"] * 4 + ["2026-01-06"] * 4,
            "factor_value": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            # day1: G0=(0.01+0.02)/2=0.015, G1=(0.03+0.04)/2=0.035
            # day2: G0=(0.05+0.07)/2=0.060, G1=(0.09+0.11)/2=0.100
            "fwd_ret": [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.11],
        }
    )
    result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=2)
    gdr = result.group_daily_returns

    assert set(gdr.columns) == {"trade_date", "group", "mean_ret"}
    assert gdr.height == 4, "2 天 × 2 组应为 4 行"

    actual = {
        (row["trade_date"], row["group"]): round(row["mean_ret"], 10)
        for row in gdr.to_dicts()
    }
    expected = {
        ("2026-01-05", 0): 0.015,
        ("2026-01-05", 1): 0.035,
        ("2026-01-06", 0): 0.060,
        ("2026-01-06", 1): 0.100,
    }
    assert actual == expected, f"逐日分组收益不符手算值：{actual}"


def test_group_daily_returns_is_sorted_for_cumulative_nav():
    """必须按 (group, trade_date) 有序——报告层直接 cumprod，乱序会算出错误净值。"""
    df = _make_strongly_monotonic_data()
    extra = df.with_columns(pl.lit("2026-01-02").alias("trade_date"))  # 更早的一天
    result = compute_monotonicity(
        pl.concat([df, extra]), factor_col="factor_value", ret_col="fwd_ret", n_groups=5
    )
    gdr = result.group_daily_returns
    for g in gdr["group"].unique().to_list():
        dates = gdr.filter(pl.col("group") == g)["trade_date"].to_list()
        assert dates == sorted(dates), f"组 {g} 的日期未升序：{dates}"


def test_group_daily_returns_empty_input_has_stable_schema():
    """空输入返回带正确 schema 的空表，报告层无需额外守卫。"""
    empty = pl.DataFrame(
        {
            "ts_code": pl.Series([], dtype=pl.Utf8),
            "trade_date": pl.Series([], dtype=pl.Utf8),
            "factor_value": pl.Series([], dtype=pl.Float64),
            "fwd_ret": pl.Series([], dtype=pl.Float64),
        }
    )
    result = compute_monotonicity(empty, factor_col="factor_value", ret_col="fwd_ret", n_groups=5)
    assert result.group_daily_returns.is_empty()
    assert set(result.group_daily_returns.columns) == {"trade_date", "group", "mean_ret"}
