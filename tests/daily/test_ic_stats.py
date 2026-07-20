"""
test_pearson_ic.py：Pearson IC 测试。
test_ic_factor_nan_mask.py：Rank IC 有效掩码必须同时过滤因子列的 NaN/inf（D1）。
test_hac_tstat.py：S3 防回归：验证 Newey-West HAC t-stat 修正。
test_monotonicity.py：测试因子单调性：分组收益是否单调递增/递减。
test_thin_cross_section_warning.py：截面被 _MIN_CROSS_SAMPLES 整天丢光时，必须出声。
test_advanced.py：高级评估保留能力：ic_analysis 内置 multi_period / decay，以及单调性 re-export。
"""

from __future__ import annotations

import datetime as dt
import logging
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats as scipy_stats

from factorzen.daily.evaluation import ic_analysis as ia
from factorzen.daily.evaluation.advanced import MonotonicityResult, compute_monotonicity
from factorzen.daily.evaluation.ic_analysis import _hac_maxlags, _ic_stats, compute_rank_ic


# ==== 来自 test_pearson_ic.py ====
def make_factor_ret_df(n_stocks: int = 50, n_dates: int = 20, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        factors = rng.standard_normal(n_stocks)
        rets = factors * 0.01 + rng.standard_normal(n_stocks) * 0.02  # positive IC
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(factors[s]),
                    "ret_1d": float(rets[s]),
                }
            )
    return pl.DataFrame(rows)


def test_pearson_ic_is_positive():
    """已知正信号因子的 Pearson IC 应为正。"""
    from factorzen.daily.evaluation.ic_analysis import compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="pearson")
    assert result.ic_mean > 0


def test_rank_ic_is_positive():
    from factorzen.daily.evaluation.ic_analysis import compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="rank")
    assert result.ic_mean > 0


def test_both_ic_returns_dict():
    """method='both' 应返回含 rank 和 pearson 两个 IcStats 的字典。"""
    from factorzen.daily.evaluation.ic_analysis import IcStats, compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="both")
    assert "rank" in result
    assert "pearson" in result
    assert isinstance(result["rank"], IcStats)
    assert isinstance(result["pearson"], IcStats)


def test_heavy_tail_pearson_less_than_rank():
    """重尾因子（单个极端值）Pearson IC 受影响更大，绝对值应小于 Rank IC。"""
    from factorzen.daily.evaluation.ic_analysis import compute_ic

    rng = np.random.default_rng(0)
    n_stocks = 100
    n_dates = 20
    start = date(2023, 1, 3)
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        factors = rng.standard_normal(n_stocks)
        factors[0] = 1000.0  # extreme outlier
        rets = np.sign(factors) * 0.01 + rng.standard_normal(n_stocks) * 0.02
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(factors[s]),
                    "ret_1d": float(rets[s]),
                }
            )
    df = pl.DataFrame(rows)
    pearson_res = compute_ic(df, method="pearson")
    rank_res = compute_ic(df, method="rank")
    # Pearson is disturbed by outlier; rank is more robust
    # Rank IC should be >= Pearson IC in absolute value (rank is outlier-robust)
    assert abs(rank_res.ic_mean) >= abs(pearson_res.ic_mean)


def test_ic_stats_fields():
    """IcStats 应含预期字段。"""
    from factorzen.daily.evaluation.ic_analysis import IcStats, compute_ic

    df = make_factor_ret_df()
    result = compute_ic(df, method="rank")
    assert isinstance(result, IcStats)
    assert hasattr(result, "ic_mean")
    assert hasattr(result, "ic_std")
    assert hasattr(result, "ir")
    assert hasattr(result, "ic_positive_ratio")
    assert hasattr(result, "n_periods")
    assert hasattr(result, "ic_tstat")
    assert hasattr(result, "ic_pvalue")
    assert hasattr(result, "ic_series")

# ==== 来自 test_ic_factor_nan_mask.py ====
def _panel(n_days=6, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    rows_f = []
    rows_r = []
    d0 = date(2024, 1, 1)
    for di in range(n_days):
        d = d0 + timedelta(days=di)
        x = rng.normal(0, 1, n_stocks)
        fwd = x * 0.8 + rng.normal(0, 0.3, n_stocks)  # 强相关截面
        for si in range(n_stocks):
            code = f"{si:06d}.SZ"
            rows_f.append({"trade_date": d, "ts_code": code, "factor_clean": float(x[si])})
            rows_r.append({"trade_date": d, "ts_code": code, "fwd_ret_1d": float(fwd[si])})
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def test_factor_nan_row_equivalent_to_dropped_row():
    """不变量：因子为 NaN 的行应与该行被物理删除等价——NaN 不得以最高秩污染 IC。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic

    factor_df, ret_df = _panel()
    codes8 = [f"{i:06d}.SZ" for i in range(8)]

    # A) 每日前 8 只股票的因子值置 NaN（收益不变）
    nan_df = factor_df.with_columns(
        pl.when(pl.col("ts_code").is_in(codes8))
        .then(float("nan"))
        .otherwise(pl.col("factor_clean"))
        .alias("factor_clean")
    )
    # B) 同样这 8 只股票的行直接删除（ground truth）
    drop_df = factor_df.filter(~pl.col("ts_code").is_in(codes8))

    nan_ic = compute_rank_ic(nan_df, ret_df, factor_col="factor_clean").ic_mean
    drop_ic = compute_rank_ic(drop_df, ret_df, factor_col="factor_clean").ic_mean

    assert abs(nan_ic - drop_ic) < 1e-9, (
        f"NaN 因子行应等价于删除该行，nan_ic={nan_ic:.6f} vs drop_ic={drop_ic:.6f}"
        "（修复前 NaN 被 rank 排最大参与 IC，两者不等）"
    )

# ==== 来自 test_hac_tstat.py ====
def _make_autocorr_ic(n: int = 200, ar_coef: float = 0.6, seed: int = 42) -> np.ndarray:
    """生成 AR(1) 自相关 IC 序列（模拟因子 IC 的序列相关性）。"""
    rng = np.random.default_rng(seed)
    ic = np.zeros(n)
    ic[0] = rng.normal(0.03, 0.08)
    for t in range(1, n):
        ic[t] = ar_coef * ic[t - 1] + rng.normal(0, 0.08 * np.sqrt(1 - ar_coef**2))
    return ic


class TestHACTstat:
    def test_hac_maxlags_formula(self):
        """HAC 最优滞后阶数公式：floor(4*(N/100)^(2/9))，最小为 1。"""
        assert _hac_maxlags(100) == 4
        assert _hac_maxlags(50) >= 1
        assert _hac_maxlags(500) > 4

    def test_hac_tstat_smaller_than_naive_for_autocorr_series(self):
        """对高自相关 IC 序列，HAC t-stat 应小于朴素（iid）t-stat。"""
        ic = _make_autocorr_ic(n=200, ar_coef=0.6)
        # HAC t-stat
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        # 朴素 t-stat（假设 iid）
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        # HAC 应更保守（绝对值更小）
        assert abs(hac_t) < abs(naive_t), (
            f"HAC t={abs(hac_t):.2f} 应 < 朴素 t={abs(naive_t):.2f}（AR(1) 自相关序列）"
        )

    def test_hac_correction_ratio_reasonable(self):
        """HAC 与朴素 t-stat 的比值应在 0.3~1.0 范围（30-70% 修正幅度）。"""
        ic = _make_autocorr_ic(n=300, ar_coef=0.6)
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        ratio = abs(hac_t) / (abs(naive_t) + 1e-10)
        assert 0.2 < ratio < 1.01, f"HAC/朴素 t 比值 {ratio:.2f} 超出合理范围 [0.2, 1.0]"

    def test_hac_low_autocorr_close_to_naive(self):
        """低自相关 IC 序列，HAC t-stat 应接近朴素 t-stat（修正幅度小）。"""
        rng = np.random.default_rng(99)
        ic = rng.normal(0.03, 0.08, 300)  # i.i.d.
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        ratio = abs(hac_t) / (abs(naive_t) + 1e-10)
        # i.i.d. 序列下 HAC 与朴素几乎相同（允许 30% 偏差）
        assert ratio > 0.7, (
            f"i.i.d. 序列下 HAC t={abs(hac_t):.2f} 与朴素 t={abs(naive_t):.2f} 应接近"
        )

    def test_ic_stats_returns_valid_types(self):
        """_ic_stats 返回 6 个 float，无 nan/inf。"""
        ic = _make_autocorr_ic(n=100, ar_coef=0.4)
        result = _ic_stats(ic)
        assert len(result) == 6
        for val in result:
            assert isinstance(val, float), f"返回值 {val} 应为 float"
            assert np.isfinite(val), f"返回值 {val} 包含 nan/inf"

    def test_ic_stats_empty_input(self):
        """空输入应返回零值，不崩溃。"""
        result = _ic_stats(np.array([]))
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

# ==== 来自 test_monotonicity.py ====
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

# ==== 来自 test_thin_cross_section_warning.py ====
_LOGGER = "factorzen.daily.evaluation.ic_analysis"


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    """告警只发一次（挖掘会调用它上千次）。每个测试从干净状态开始。"""
    monkeypatch.setattr(ia, "_warned_thin_cross_section", False, raising=False)


def _frame(n_stocks: int, n_days: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        for i in range(n_stocks):
            rows.append({
                "trade_date": dt.date(2023, 1, 2) + dt.timedelta(days=d),
                "ts_code": f"{i:06d}.SZ",
                "factor": float(rng.normal()),
                "fwd_ret_1d": float(rng.normal()),
            })
    return pl.DataFrame(rows)


def test_warns_when_every_day_is_dropped(caplog):
    """20 只 < 30 → 每天都被丢 → 必须 WARNING，且带上最大截面数供用户排查。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(_frame(20), "factor", "fwd_ret_1d")

    assert out.height == 0
    assert "30" in caplog.text, "告警应说明门槛值"
    assert "20" in caplog.text, "告警应说明实际最大截面数，否则用户无从下手"


def test_no_warning_when_cross_sections_are_thick(caplog):
    """40 只 → 全部保留 → 不该有任何告警。没有这条，「无脑每次都警告」也能过上一个测试。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(_frame(40), "factor", "fwd_ret_1d")

    assert out.height == 5
    assert caplog.text == ""


def test_warning_is_emitted_only_once(caplog):
    """挖掘每评估一个表达式就调一次；每次都警告会把日志淹没。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for _ in range(5):
            ia._rank_ic_by_date(_frame(20), "factor", "fwd_ret_1d")

    assert caplog.text.count("截面") == 1, f"应恰好告警一次，实得：\n{caplog.text}"


def test_no_warning_when_data_is_simply_empty(caplog):
    """全 null 因子是另一种病（不是截面太薄），不该报「截面不足」误导排查方向。"""
    df = _frame(40).with_columns(pl.lit(None, dtype=pl.Float64).alias("factor"))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(df, "factor", "fwd_ret_1d")

    assert out.height == 0
    assert caplog.text == ""


def test_partial_drop_does_not_warn(caplog):
    """只有**整天丢光**才告警。部分丢弃（早期上市股少）是正常的，警告会变噪音。"""
    thick = _frame(40, n_days=4)
    thin = _frame(20, n_days=1).with_columns(
        (pl.col("trade_date") + pl.duration(days=10)).alias("trade_date")
    )
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._rank_ic_by_date(pl.concat([thick, thin]), "factor", "fwd_ret_1d")

    assert out.height == 4, "薄的那天被丢，厚的四天保留"
    assert caplog.text == ""


def test_pearson_path_warns_too(caplog):
    """双路径：`_pearson_ic_by_date` 有它自己的过滤，同样会静默丢光。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = ia._pearson_ic_by_date(_frame(20), "factor", "fwd_ret_1d")

    assert out.height == 0
    assert "30" in caplog.text

# ==== 来自 test_advanced.py ====
_N_STOCKS = 40  # 必须 ≥ _MIN_CROSS_SAMPLES(=30)
_N_DAYS = 30


def _make_factor_and_returns(seed: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    """带递减信噪比的合成数据：|IC|(1d) > |IC|(5d) > |IC|(10d)。"""
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        pl.date(2026, 1, 5),
        pl.date(2026, 1, 5) + pl.duration(days=_N_DAYS - 1),
        interval="1d",
        eager=True,
    )
    codes = [f"{600000 + i:06d}.SH" for i in range(_N_STOCKS)]
    f = rng.standard_normal(_N_STOCKS)

    rows_f, rows_r = [], []
    for d in dates:
        for i, code in enumerate(codes):
            rows_f.append({"trade_date": d, "ts_code": code, "factor_clean": float(f[i])})
            rows_r.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "fwd_ret_1d": float(f[i] + 0.5 * rng.standard_normal()),
                    "fwd_ret_5d": float(f[i] + 2.0 * rng.standard_normal()),
                    "fwd_ret_10d": float(f[i] + 5.0 * rng.standard_normal()),
                    "fwd_ret_20d": float(f[i] + 8.0 * rng.standard_normal()),
                }
            )
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)


def test_rank_ic_multi_period_covers_horizons():
    factor, ret = _make_factor_and_returns()
    result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10, 20])

    assert sorted(result.multi_period.keys()) == [1, 5, 10, 20]
    assert sorted(result.decay.keys()) == [1, 5, 10, 20]
    for h, stats in result.multi_period.items():
        assert "ic_mean" in stats and "ir" in stats
        assert stats["ic_mean"] == pytest.approx(result.decay[h])


def test_rank_ic_decay_is_monotonically_decreasing_in_abs():
    factor, ret = _make_factor_and_returns()
    result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10])
    ic = result.decay

    assert all(v == v for v in ic.values()), f"IC 不该是 nan：{ic}"
    assert ic[1] > 0.5, f"1 日 IC 应显著为正，实得 {ic[1]:.4f}"
    assert abs(ic[1]) > abs(ic[5]) > abs(ic[10]), f"IC 未随持有期衰减：{ic}"


def test_rank_ic_series_covers_trading_days():
    factor, ret = _make_factor_and_returns()
    result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10])

    assert result.n_periods == _N_DAYS
    assert result.ic_series.height == _N_DAYS
    assert result.ic_std > 0


def test_monotonicity_reexport_runs():
    """advanced 包仍导出 compute_monotonicity（管线在用）。"""
    factor, ret = _make_factor_and_returns()
    mono_df = factor.join(
        ret.select(["trade_date", "ts_code", "fwd_ret_1d"]),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    mono = compute_monotonicity(
        mono_df, factor_col="factor_clean", ret_col="fwd_ret_1d", n_groups=5
    )
    assert isinstance(mono, MonotonicityResult)
    assert len(mono.group_means) == 5

