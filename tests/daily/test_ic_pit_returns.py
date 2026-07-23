"""
test_ic_stats.py：test_pearson_ic.py：Pearson IC 测试。
test_pit_fwd_returns.py：test_pit.py：测试 Point-In-Time 财务数据对齐。
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats as scipy_stats

from factorzen.daily.data.pit import attach_fundamentals, pit_align
from factorzen.daily.evaluation import ic_analysis as ia
from factorzen.daily.evaluation.advanced import compute_monotonicity
from factorzen.daily.evaluation.ic_analysis import (
    _hac_maxlags,
    _ic_stats,
    compute_fwd_returns,
    compute_rank_ic,
)


# ==== 来自 test_ic_stats.py ====
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


def test_ic_stats_pearson_rank_suite():
    """method='both' 应返回含 rank 和 pearson 两个 IcStats 的字典。；重尾因子（单个极端值）Pearson IC 受影响更大，绝对值应小于 Rank IC。；不变量：因子为 NaN 的行应与该行被物理删除等价——NaN 不得以最高秩污染 IC。"""
    # -- 原 test_both_ic_returns_dict --
    def _section_0_test_both_ic_returns_dict():
        from factorzen.daily.evaluation.ic_analysis import IcStats, compute_ic

        df = make_factor_ret_df()
        result = compute_ic(df, method="both")
        assert "rank" in result
        assert "pearson" in result
        assert isinstance(result["rank"], IcStats)
        assert isinstance(result["pearson"], IcStats)

    _section_0_test_both_ic_returns_dict()

    # -- 原 test_heavy_tail_pearson_less_than_rank --
    def _section_1_test_heavy_tail_pearson_less_than_rank():
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

    _section_1_test_heavy_tail_pearson_less_than_rank()

    # -- 原 test_factor_nan_row_equivalent_to_dropped_row --
    def _section_2_test_factor_nan_row_equivalent_to_dropped_row():
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

    _section_2_test_factor_nan_row_equivalent_to_dropped_row()


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
    def test_hac_tstat_suite(self):
        """HAC 最优滞后阶数公式：floor(4*(N/100)^(2/9))，最小为 1。；对高自相关 IC 序列，HAC t-stat 应小于朴素（iid）t-stat。；HAC 与朴素 t-stat 的比值应在 0.3~1.0 范围（30-70% 修正幅度）。；低自相关 IC 序列，HAC t-stat 应接近朴素 t-stat（修正幅度小）。；_ic_stats 返回 6 个 float，无 nan/inf。；空输入应返回零值，不崩溃。"""
        # -- 原 test_hac_maxlags_formula --
        assert _hac_maxlags(100) == 4
        assert _hac_maxlags(50) >= 1
        assert _hac_maxlags(500) > 4

        # -- 原 test_hac_tstat_smaller_than_naive_for_autocorr_series --
        ic = _make_autocorr_ic(n=200, ar_coef=0.6)
        # HAC t-stat
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        # 朴素 t-stat（假设 iid）
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        # HAC 应更保守（绝对值更小）
        assert abs(hac_t) < abs(naive_t), (
            f"HAC t={abs(hac_t):.2f} 应 < 朴素 t={abs(naive_t):.2f}（AR(1) 自相关序列）"
        )

        # -- 原 test_hac_correction_ratio_reasonable --
        ic = _make_autocorr_ic(n=300, ar_coef=0.6)
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        ratio = abs(hac_t) / (abs(naive_t) + 1e-10)
        assert 0.2 < ratio < 1.01, f"HAC/朴素 t 比值 {ratio:.2f} 超出合理范围 [0.2, 1.0]"

        # -- 原 test_hac_low_autocorr_close_to_naive --
        rng = np.random.default_rng(99)
        ic = rng.normal(0.03, 0.08, 300)  # i.i.d.
        _, _, _, _, hac_t, _ = _ic_stats(ic)
        naive_t, _ = scipy_stats.ttest_1samp(ic, popmean=0.0)
        ratio = abs(hac_t) / (abs(naive_t) + 1e-10)
        # i.i.d. 序列下 HAC 与朴素几乎相同（允许 30% 偏差）
        assert ratio > 0.7, (
            f"i.i.d. 序列下 HAC t={abs(hac_t):.2f} 与朴素 t={abs(naive_t):.2f} 应接近"
        )

        # -- 原 test_ic_stats_returns_valid_types --
        ic = _make_autocorr_ic(n=100, ar_coef=0.4)
        result = _ic_stats(ic)
        assert len(result) == 6
        for val in result:
            assert isinstance(val, float), f"返回值 {val} 应为 float"
            assert np.isfinite(val), f"返回值 {val} 包含 nan/inf"

        # -- 原 test_ic_stats_empty_input --
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


def test_monotonicity_suite():
    """强正相关数据 → monotonicity_score 接近 1.0。；分组均值应为单调递增。；逐日 × 分组收益对齐手算值。；必须按 (group, trade_date) 有序——报告层直接 cumprod，乱序会算出错误净值。；空输入返回带正确 schema 的空表，报告层无需额外守卫。"""
    # -- 原 test_monotonicity_strongly_positive --
    def _section_0_test_monotonicity_strongly_positive():
        df = _make_strongly_monotonic_data()
        result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10)
        assert result.monotonicity_score > 0.5
        assert result.direction == "positive"

    _section_0_test_monotonicity_strongly_positive()

    # -- 原 test_monotonicity_group_means_monotonic --
    def _section_1_test_monotonicity_group_means_monotonic():
        df = _make_strongly_monotonic_data()
        result = compute_monotonicity(df, factor_col="factor_value", ret_col="fwd_ret", n_groups=10)
        means = result.group_means
        assert len(means) == 10
        for i in range(len(means) - 1):
            assert means[i] <= means[i + 1], f"组 {i}→{i + 1} 收益不单调"

    _section_1_test_monotonicity_group_means_monotonic()

    # -- 原 test_group_daily_returns_matches_hand_computed_ground_truth --
    def _section_2_test_group_daily_returns_matches_hand_computed_ground_truth():
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

    _section_2_test_group_daily_returns_matches_hand_computed_ground_truth()

    # -- 原 test_group_daily_returns_is_sorted_for_cumulative_nav --
    def _section_3_test_group_daily_returns_is_sorted_for_cumulative_nav():
        df = _make_strongly_monotonic_data()
        extra = df.with_columns(pl.lit("2026-01-02").alias("trade_date"))  # 更早的一天
        result = compute_monotonicity(
            pl.concat([df, extra]), factor_col="factor_value", ret_col="fwd_ret", n_groups=5
        )
        gdr = result.group_daily_returns
        for g in gdr["group"].unique().to_list():
            dates = gdr.filter(pl.col("group") == g)["trade_date"].to_list()
            assert dates == sorted(dates), f"组 {g} 的日期未升序：{dates}"

    _section_3_test_group_daily_returns_is_sorted_for_cumulative_nav()

    # -- 原 test_group_daily_returns_empty_input_has_stable_schema --
    def _section_4_test_group_daily_returns_empty_input_has_stable_schema():
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

    _section_4_test_group_daily_returns_empty_input_has_stable_schema()


# ── group_daily_returns：报告层画分组净值/绩效的数据源 ────────────────────────


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


def test_advanced_ic_multi_period_suite():
    """test_rank_ic_multi_period_covers_horizons；test_rank_ic_decay_is_monotonically_decreasing_in_abs；test_rank_ic_series_covers_trading_days"""
    # -- 原 test_rank_ic_multi_period_covers_horizons --
    def _section_0_test_rank_ic_multi_period_covers_horizons():
        factor, ret = _make_factor_and_returns()
        result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10, 20])

        assert sorted(result.multi_period.keys()) == [1, 5, 10, 20]
        assert sorted(result.decay.keys()) == [1, 5, 10, 20]
        for h, stats in result.multi_period.items():
            assert "ic_mean" in stats and "ir" in stats
            assert stats["ic_mean"] == pytest.approx(result.decay[h])

    _section_0_test_rank_ic_multi_period_covers_horizons()

    # -- 原 test_rank_ic_decay_is_monotonically_decreasing_in_abs --
    def _section_1_test_rank_ic_decay_is_monotonically_decreasing_in_abs():
        factor, ret = _make_factor_and_returns()
        result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10])
        ic = result.decay

        assert all(v == v for v in ic.values()), f"IC 不该是 nan：{ic}"
        assert ic[1] > 0.5, f"1 日 IC 应显著为正，实得 {ic[1]:.4f}"
        assert abs(ic[1]) > abs(ic[5]) > abs(ic[10]), f"IC 未随持有期衰减：{ic}"

    _section_1_test_rank_ic_decay_is_monotonically_decreasing_in_abs()

    # -- 原 test_rank_ic_series_covers_trading_days --
    def _section_2_test_rank_ic_series_covers_trading_days():
        factor, ret = _make_factor_and_returns()
        result = compute_rank_ic(factor, ret, factor_col="factor_clean", horizons=[1, 5, 10])

        assert result.n_periods == _N_DAYS
        assert result.ic_series.height == _N_DAYS
        assert result.ic_std > 0

    _section_2_test_rank_ic_series_covers_trading_days()


# ==== 来自 test_pit_fwd_returns.py ====
# ==== 来自 test_pit.py ====
# ── helpers ────────────────────────────────────────────────────────────────


def _make_fina(rows: list[tuple]) -> pl.DataFrame:
    """从 (ts_code, end_date, ann_date, roe) 元组列表构造财务数据。"""
    return pl.DataFrame(
        rows,
        schema={"ts_code": pl.Utf8, "end_date": pl.Date, "ann_date": pl.Date, "roe": pl.Float64},
        orient="row",
    )


# ── correctness ─────────────────────────────────────────────────────────────


def test_pit_align_suite():
    """验证无前视偏差：快照日只使用「已公告」的财报中 end_date 最新的那条。；多股票场景：各自独立取最新已公告财报。；空 DataFrame 或空 snapshot 列表 → 返回空 DataFrame。；1000 只股票 × 40 个月频快照 → 2 秒内完成。；常规多股票多季度。；更正公告反例：后公告但 end_date 更旧，naive asof-ann 会答错。"""
    # -- 原 test_pit_align_correctness --
    def _section_0_test_pit_align_correctness():
        fina = _make_fina(
            [
                ("000001.SZ", date(2024, 6, 30), date(2024, 8, 15), 12.0),
                ("000001.SZ", date(2024, 9, 30), date(2024, 10, 30), 15.0),
            ]
        )

        snapshots = [
            date(2024, 8, 31),  # Q2 已公告，Q3 未公告 → 应取 Q2（roe=12.0）
            date(2024, 10, 31),  # Q2/Q3 均已公告 → 应取 Q3（roe=15.0）
        ]

        result = pit_align(fina, snapshots)

        # 两个快照日各返回一条
        assert result.height == 2

        row_aug = result.filter(pl.col("snapshot_date") == date(2024, 8, 31))
        assert row_aug.height == 1
        assert row_aug[0, "end_date"] == date(2024, 6, 30)
        assert row_aug[0, "roe"] == 12.0

        row_oct = result.filter(pl.col("snapshot_date") == date(2024, 10, 31))
        assert row_oct.height == 1
        assert row_oct[0, "end_date"] == date(2024, 9, 30)
        assert row_oct[0, "roe"] == 15.0

    _section_0_test_pit_align_correctness()

    # -- 原 test_pit_align_multiple_stocks --
    def _section_1_test_pit_align_multiple_stocks():
        d1, d2, d3 = date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)
        fina = _make_fina(
            [
                ("A", d1, date(2024, 4, 25), 10.0),
                ("A", d2, date(2024, 8, 28), 12.0),
                ("A", d3, date(2024, 10, 30), 14.0),
                ("B", d1, date(2024, 4, 25), 20.0),
                ("B", d2, date(2024, 8, 30), 22.0),
                # B 没有 Q3
            ]
        )

        snapshots = [date(2024, 9, 1), date(2024, 11, 1)]

        result = pit_align(fina, snapshots)

        # Sep: A 取 Q2(12.0), B 取 Q2(22.0)
        sep_a = result.filter(pl.col("snapshot_date") == date(2024, 9, 1), pl.col("ts_code") == "A")
        assert sep_a[0, "roe"] == 12.0
        sep_b = result.filter(pl.col("snapshot_date") == date(2024, 9, 1), pl.col("ts_code") == "B")
        assert sep_b[0, "roe"] == 22.0

        # Nov: A 取 Q3(14.0), B 仍取 Q2(22.0)（无 Q3）
        nov_a = result.filter(pl.col("snapshot_date") == date(2024, 11, 1), pl.col("ts_code") == "A")
        assert nov_a[0, "roe"] == 14.0
        nov_b = result.filter(pl.col("snapshot_date") == date(2024, 11, 1), pl.col("ts_code") == "B")
        assert nov_b[0, "roe"] == 22.0

        assert result.height == 4

    _section_1_test_pit_align_multiple_stocks()

    # -- 原 test_pit_align_empty_input --
    def _section_2_test_pit_align_empty_input():
        fina = _make_fina(
            [
                ("A", date(2024, 6, 30), date(2024, 8, 1), 10.0),
            ]
        )

        # 空 fina_df
        assert pit_align(pl.DataFrame(), [date(2024, 9, 1)]).is_empty()

        # 空 snapshot_dates
        assert pit_align(fina, []).is_empty()

        # 两者皆空
        assert pit_align(pl.DataFrame(), []).is_empty()

    _section_2_test_pit_align_empty_input()

    # -- 原 test_pit_align_performance --
    def _section_3_test_pit_align_performance():
        n_stocks = 1000
        n_periods = 40

        base = date(2020, 1, 1)
        # 每只股票 4 份年报（end_date 在 2020-2023）
        rows = []
        for s in range(n_stocks):
            for y in range(4):
                end_d = date(2020 + y, 12, 31)
                ann_d = date(2021 + y, 4, 30)
                rows.append((f"stock_{s:04d}", end_d, ann_d, y * 5.0 + s * 0.01))

        fina = _make_fina(rows)

        snapshots = [base + timedelta(days=30 * i) for i in range(n_periods)]

        start = time.perf_counter()
        result = pit_align(fina, snapshots)
        elapsed = time.perf_counter() - start

        assert not result.is_empty(), "结果不应为空"
        assert elapsed < 2.0, f"耗时 {elapsed:.2f}s ≥ 2s，算法效率不达标"

    _section_3_test_pit_align_performance()

    # -- 原 test_pit_align_multi_stock_multi_quarter --
    def _section_4_test_pit_align_multi_stock_multi_quarter():
        fina = pl.DataFrame(
            {
                "ts_code": [
                    "A", "A", "A",
                    "B", "B",
                    "C",
                ],
                "end_date": [
                    date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30),
                    date(2023, 3, 31), date(2023, 6, 30),
                    date(2023, 6, 30),
                ],
                "ann_date": [
                    date(2023, 4, 20), date(2023, 8, 15), date(2023, 10, 25),
                    date(2023, 4, 22), date(2023, 8, 20),
                    date(2023, 8, 10),
                ],
                "roe": [10.0, 12.0, 14.0, 20.0, 22.0, 30.0],
            }
        )
        snaps = [
            date(2023, 5, 1),
            date(2023, 9, 1),
            date(2023, 11, 1),
        ]
        expected = _pit_align_reference(fina, snaps)
        got = pit_align(fina, snaps)
        _assert_equiv(got, expected)
        # 语义抽检：9/1 时 A 应取 Q2 而非尚未公告的 Q3
        row = got.filter(
            (pl.col("snapshot_date") == date(2023, 9, 1)) & (pl.col("ts_code") == "A")
        )
        assert row[0, "end_date"] == date(2023, 6, 30)
        assert row[0, "roe"] == 12.0

    _section_4_test_pit_align_multi_stock_multi_quarter()

    # -- 原 test_pit_align_correction_later_ann_older_end --
    def _section_5_test_pit_align_correction_later_ann_older_end():
        fina = pl.DataFrame(
            {
                "ts_code": ["X", "X"],
                "end_date": [date(2023, 6, 30), date(2023, 3, 31)],
                "ann_date": [date(2023, 8, 15), date(2023, 9, 1)],  # 更正更晚
                "roe": [15.0, 99.0],  # 99 是陷阱：按 ann 最新会错取
            }
        )
        snaps = [date(2023, 8, 20), date(2023, 9, 15)]
        expected = _pit_align_reference(fina, snaps)
        got = pit_align(fina, snaps)
        _assert_equiv(got, expected)

        # 两日都应取 Q2 (end=6/30, roe=15)，绝不能取更正的 Q1
        for sd in snaps:
            row = got.filter(pl.col("snapshot_date") == sd)
            assert row.height == 1
            assert row[0, "end_date"] == date(2023, 6, 30)
            assert row[0, "roe"] == 15.0

    _section_5_test_pit_align_correction_later_ann_older_end()


# ── empty input ─────────────────────────────────────────────────────────────


# ── performance ─────────────────────────────────────────────────────────────


# ==== 来自 test_pit_align_equiv.py ====
def _pit_align_reference(
    fina_df: pl.DataFrame,
    snapshot_dates: list[date],
) -> pl.DataFrame:
    """旧实现逐字拷贝（Wave1 前 master），作 golden 基准。"""
    if fina_df.is_empty() or not snapshot_dates:
        return pl.DataFrame()

    if fina_df["ann_date"].dtype == pl.Utf8:
        fina_df = fina_df.with_columns(
            pl.col("ann_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )

    fina_df = fina_df.filter(pl.col("ann_date").is_not_null())

    fina_sorted = fina_df.sort(["ts_code", "end_date"], descending=[False, True])

    results: list[pl.DataFrame] = []
    for sd in snapshot_dates:
        valid = fina_sorted.filter(pl.col("ann_date") <= sd)
        if valid.is_empty():
            continue

        best = (
            valid.group_by("ts_code")
            .first()
            .with_columns(pl.lit(sd).cast(pl.Date).alias("snapshot_date"))
        )
        results.append(best)

    if not results:
        return pl.DataFrame()

    return pl.concat(results, how="vertical")


def _assert_equiv(got: pl.DataFrame, expected: pl.DataFrame) -> None:
    """列集合、dtype、行集合一致；行序无契约，sort 后再比。"""
    if expected.is_empty():
        assert got.is_empty(), f"expected empty, got height={got.height}"
        return
    assert not got.is_empty()
    assert set(got.columns) == set(expected.columns), (
        f"cols got={got.columns} expected={expected.columns}"
    )
    for c in expected.columns:
        assert got[c].dtype == expected[c].dtype, (
            f"dtype {c}: got={got[c].dtype} expected={expected[c].dtype}"
        )
    sort_keys = [c for c in ("snapshot_date", "ts_code") if c in expected.columns]
    g = got.select(expected.columns).sort(sort_keys)
    e = expected.select(expected.columns).sort(sort_keys)
    assert g.equals(e), (
        f"mismatch\ngot:\n{g}\nexpected:\n{e}"
    )


def test_pit_align_ann_date_null_and_string_dtype():
    """ann_date 为 null / String YYYYMMDD 两种 dtype。"""
    # String dtype
    fina_str = pl.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "end_date": [date(2023, 3, 31), date(2023, 6, 30), date(2023, 6, 30)],
            "ann_date": ["20230420", "20230815", "20230810"],
            "roe": [1.0, 2.0, 3.0],
        }
    )
    snaps = [date(2023, 5, 1), date(2023, 9, 1)]
    expected = _pit_align_reference(fina_str, snaps)
    got = pit_align(fina_str, snaps)
    _assert_equiv(got, expected)

    # null ann_date 被过滤
    fina_null = pl.DataFrame(
        {
            "ts_code": ["A", "A", "A"],
            "end_date": [date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30)],
            "ann_date": [date(2023, 4, 20), None, date(2023, 10, 25)],
            "roe": [1.0, 2.0, 3.0],
        }
    )
    snaps2 = [date(2023, 9, 1), date(2023, 11, 1)]
    expected2 = _pit_align_reference(fina_null, snaps2)
    got2 = pit_align(fina_null, snaps2)
    _assert_equiv(got2, expected2)
    # 9/1 时只有 Q1 可见（Q2 ann 为 null 被丢）
    row = got2.filter(pl.col("snapshot_date") == date(2023, 9, 1))
    assert row[0, "end_date"] == date(2023, 3, 31)


def test_pit_align_snapshot_before_any_announcement():
    """快照日早于一切公告 → 该日无输出行。"""
    fina = pl.DataFrame(
        {
            "ts_code": ["A"],
            "end_date": [date(2023, 6, 30)],
            "ann_date": [date(2023, 8, 15)],
            "roe": [10.0],
        }
    )
    snaps = [date(2023, 1, 1), date(2023, 8, 20)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)
    assert got.filter(pl.col("snapshot_date") == date(2023, 1, 1)).is_empty()
    assert got.filter(pl.col("snapshot_date") == date(2023, 8, 20)).height == 1


def test_pit_align_same_end_date_tie_break():
    """同 end_date 双记录：旧实现 sort 后 group_by().first() 的 tie-break。"""
    # 原相对顺序：先 v1 再 v2；同 end_date 应取原相对顺序第一条 (v1)
    fina = pl.DataFrame(
        {
            "ts_code": ["T", "T"],
            "end_date": [date(2023, 12, 31), date(2023, 12, 31)],
            "ann_date": [date(2024, 3, 1), date(2024, 4, 1)],
            "roe": [10.0, 20.0],
            "version": ["v1", "v2"],
        }
    )
    snaps = [date(2024, 3, 15), date(2024, 4, 15)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)

    # 两日均应取 v1（原相对顺序第一条），即便 v2 更晚公告
    for sd in snaps:
        row = got.filter(pl.col("snapshot_date") == sd)
        assert row.height == 1
        assert row[0, "roe"] == 10.0
        assert row[0, "version"] == "v1"


def test_pit_align_tie_break_later_ann_earlier_in_file():
    """同 end_date：原文件中更晚公告的行反而排在前面 → 两日可见后应取该行。"""
    fina = pl.DataFrame(
        {
            "ts_code": ["T", "T"],
            "end_date": [date(2023, 12, 31), date(2023, 12, 31)],
            # 行序：先 late 再 early（与 ann 时间相反）
            "ann_date": [date(2024, 4, 1), date(2024, 3, 1)],
            "roe": [20.0, 10.0],
            "version": ["late_first", "early_second"],
        }
    )
    snaps = [date(2024, 3, 15), date(2024, 4, 15)]
    expected = _pit_align_reference(fina, snaps)
    got = pit_align(fina, snaps)
    _assert_equiv(got, expected)

    # 3/15 只有 early_second 可见
    r1 = got.filter(pl.col("snapshot_date") == date(2024, 3, 15))
    assert r1[0, "version"] == "early_second"
    # 4/15 两者可见：原相对顺序第一条是 late_first
    r2 = got.filter(pl.col("snapshot_date") == date(2024, 4, 15))
    assert r2[0, "version"] == "late_first"


# ==== 来自 test_fundamentals_pit.py ====
def _fina() -> pl.DataFrame:
    """两份报告:Q1(end 0331)0420 公告、Q2(end 0630)0815 公告——真实数据 ann/end 为 String。

    含全套质量/成长字段,验证扩充后的叶子一并 PIT 对齐。
    """
    return pl.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ"],
        "end_date": ["20200331", "20200630"],
        "ann_date": ["20200420", "20200815"],
        "roe": [10.0, 12.0], "roa": [1.0, 1.2],
        "grossprofit_margin": [40.0, 41.0], "netprofit_margin": [20.0, 21.0],
        "debt_to_assets": [50.0, 51.0],
        "or_yoy": [8.0, 9.0], "netprofit_yoy": [15.0, 16.0], "assets_yoy": [5.0, 6.0],
    })


def _daily(dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.datetime.strptime(d, "%Y%m%d").date() for d in dates],
        "ts_code": ["000001.SZ"] * len(dates),
        "close": [10.0] * len(dates),
    })


def test_fundamentals_pit_suite():
    """t 日在 Q1 公告(0420)之前 → roe 必须是 null,绝不能把 0420 才公告的报告泄漏回 0410。；公告后取最新已公告报告:0420~0814 用 Q1(10.0);0815 起用 Q2(end 更大,12.0)。；无 finance 数据(空帧)→ 原样返回但补齐 roe/assets_yoy 为 null(表达式引用不崩)。；扩充的质量/成长字段(毛利率/营收增速等)与 roe 同套 PIT 对齐,公告后取最新报告。；全套质量/成长叶子已注册且可解析(否则 LLM/搜索碰不到、prompt 广告了却用不了)。"""
    # -- 原 test_no_future_leak_before_announcement --
    def _section_0_test_no_future_leak_before_announcement():
        out = attach_fundamentals(_daily(["20200410"]), fina_df=_fina())
        row = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10))
        assert row["roe"][0] is None, "Q1 报告在公告日前泄漏 → 未来函数!"
        assert row["assets_yoy"][0] is None

    _section_0_test_no_future_leak_before_announcement()

    # -- 原 test_uses_latest_announced_report --
    def _section_1_test_uses_latest_announced_report():
        out = attach_fundamentals(_daily(["20200410", "20200501", "20200820"]), fina_df=_fina())
        by_date = {r["trade_date"]: r["roe"] for r in out.iter_rows(named=True)}
        assert by_date[dt.date(2020, 4, 10)] is None       # 公告前
        assert by_date[dt.date(2020, 5, 1)] == 10.0         # Q1 已公告
        assert by_date[dt.date(2020, 8, 20)] == 12.0        # Q2 已公告(end_date 更大)

    _section_1_test_uses_latest_announced_report()

    # -- 原 test_missing_finance_returns_daily_with_null_cols --
    def _section_2_test_missing_finance_returns_daily_with_null_cols():
        out = attach_fundamentals(_daily(["20200501"]), fina_df=pl.DataFrame())
        assert "roe" in out.columns and "assets_yoy" in out.columns
        assert out["roe"][0] is None

    _section_2_test_missing_finance_returns_daily_with_null_cols()

    # -- 原 test_expanded_fields_pit_aligned --
    def _section_3_test_expanded_fields_pit_aligned():
        out = attach_fundamentals(_daily(["20200410", "20200820"]), fina_df=_fina())
        pre = out.filter(pl.col("trade_date") == dt.date(2020, 4, 10))
        post = out.filter(pl.col("trade_date") == dt.date(2020, 8, 20))
        for col in ("grossprofit_margin", "or_yoy", "netprofit_yoy", "debt_to_assets", "roa"):
            assert pre[col][0] is None, f"{col} 公告前泄漏 → 未来函数!"
        assert post["grossprofit_margin"][0] == 41.0   # Q2
        assert post["or_yoy"][0] == 9.0

    _section_3_test_expanded_fields_pit_aligned()

    # -- 原 test_all_fundamental_leaves_registered_and_parse --
    def _section_4_test_all_fundamental_leaves_registered_and_parse():
        from factorzen.discovery.expression import feature_names, parse_expr
        from factorzen.discovery.operators import FUNDAMENTAL_FEATURES, LEAF_FEATURES
        expected = {"roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
                    "or_yoy", "netprofit_yoy", "assets_yoy"}
        assert expected <= FUNDAMENTAL_FEATURES
        for leaf in expected:
            assert leaf in LEAF_FEATURES, f"{leaf} 未注册为叶子"
            assert leaf in feature_names(parse_expr(f"rank({leaf})")), f"{leaf} 解析不出"

    _section_4_test_all_fundamental_leaves_registered_and_parse()


# ==== 来自 test_fwd_returns.py ====
def test_fwd_returns_suite():
    """test_compute_fwd_returns_raises_on_missing_key_columns；test_compute_fwd_returns_raises_when_no_price_or_ret_column；test_fwd_ret_1d_uses_next_close_over_current_close；test_fwd_ret_5d_is_cumulative_holding_period_return；test_fwd_returns_compound_from_ret_when_close_is_absent"""
    # -- 原 test_compute_fwd_returns_raises_on_missing_key_columns --
    def _section_0_test_compute_fwd_returns_raises_on_missing_key_columns():
        df = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "close": [100.0]})
        with pytest.raises(ValueError) as exc:
            compute_fwd_returns(df, horizons=[1])
        assert "ts_code" in str(exc.value)

    _section_0_test_compute_fwd_returns_raises_on_missing_key_columns()

    # -- 原 test_compute_fwd_returns_raises_when_no_price_or_ret_column --
    def _section_1_test_compute_fwd_returns_raises_when_no_price_or_ret_column():
        df = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"]})
        with pytest.raises(ValueError) as exc:
            compute_fwd_returns(df, horizons=[1], ret_col="ret")
        msg = str(exc.value)
        assert "close" in msg and "ret" in msg

    _section_1_test_compute_fwd_returns_raises_when_no_price_or_ret_column()

    # -- 原 test_fwd_ret_1d_uses_next_close_over_current_close --
    def _section_2_test_fwd_ret_1d_uses_next_close_over_current_close():
        df = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
                "ts_code": ["000001.SZ"] * 3,
                "close": [100.0, 110.0, 121.0],
            }
        ).with_columns((pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret"))

        out = compute_fwd_returns(df, horizons=[1], ret_col="ret")

        assert out["fwd_ret_1d"].to_list() == pytest.approx([0.10, 0.10, None])

    _section_2_test_fwd_ret_1d_uses_next_close_over_current_close()

    # -- 原 test_fwd_ret_5d_is_cumulative_holding_period_return --
    def _section_3_test_fwd_ret_5d_is_cumulative_holding_period_return():
        closes = [100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0]
        df = pl.DataFrame(
            {
                "trade_date": [
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                    date(2024, 1, 8),
                    date(2024, 1, 9),
                    date(2024, 1, 10),
                ],
                "ts_code": ["000001.SZ"] * len(closes),
                "close": closes,
            }
        ).with_columns((pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret"))

        out = compute_fwd_returns(df, horizons=[5], ret_col="ret")

        assert out["fwd_ret_5d"][0] == pytest.approx(115.0 / 100.0 - 1.0)
        assert out["fwd_ret_5d"][1] == pytest.approx(121.0 / 101.0 - 1.0)
        assert out["fwd_ret_5d"].to_list()[-5:] == [None, None, None, None, None]

    _section_3_test_fwd_ret_5d_is_cumulative_holding_period_return()

    # -- 原 test_fwd_returns_compound_from_ret_when_close_is_absent --
    def _section_4_test_fwd_returns_compound_from_ret_when_close_is_absent():
        df = pl.DataFrame(
            {
                "trade_date": [
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                ],
                "ts_code": ["000001.SZ"] * 4,
                "ret": [0.0, 0.10, 0.20, -0.05],
            }
        )

        out = compute_fwd_returns(df, horizons=[2], ret_col="ret")

        assert out["fwd_ret_2d"][0] == pytest.approx((1.10 * 1.20) - 1.0)
        assert out["fwd_ret_2d"][1] == pytest.approx((1.20 * 0.95) - 1.0)
        assert out["fwd_ret_2d"].to_list()[-2:] == [None, None]

    _section_4_test_fwd_returns_compound_from_ret_when_close_is_absent()


def test_compute_rank_ic_factor_rank_reuse_bit_exact():
    """改动 B：factor_rank_col 开/关 + compute_rank_ic 与旧实现（每 horizon 独立算）逐位一致。

    随机小帧含 NaN / null / 并列值，覆盖 :204 NaN rank 陷阱路径。
    """
    import numpy as np

    from factorzen.daily.evaluation.ic_analysis import (
        ICAnalysisResult,
        _compute_walk_forward_ic,
        _ic_stats,
        _rank_ic_by_date,
        compute_rank_ic,
    )

    rng = np.random.default_rng(20260723)
    n_days, n_stocks = 15, 40
    horizons = [1, 5, 10, 20]
    d0 = date(2024, 3, 1)
    f_rows: list[dict] = []
    r_rows: list[dict] = []
    for di in range(n_days):
        d = d0 + timedelta(days=di)
        base = rng.integers(0, 8, size=n_stocks).astype(float)
        base[0] = base[1]  # 并列
        factor_vals = base.copy()
        if di % 3 == 0:
            factor_vals[2] = np.nan  # NaN 因子
        rets = {
            h: base * (0.3 / max(h, 1)) + rng.normal(0, 0.2, n_stocks) for h in horizons
        }
        for h in horizons:
            if di % 4 == (h % 4):
                rets[h][5] = np.nan  # 各 horizon 不同 NaN 掩码
            if di % 7 == 0:
                rets[h][6] = np.inf
        for si in range(n_stocks):
            code = f"{si:06d}.SZ"
            if di % 5 == 0 and si == 3:
                fv: float | None = None  # null 因子
            else:
                fv = float(factor_vals[si])
            f_rows.append({"trade_date": d, "ts_code": code, "factor_clean": fv})
            r_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    **{f"fwd_ret_{h}d": float(rets[h][si]) for h in horizons},
                }
            )

    factor_df = pl.DataFrame(f_rows)
    ret_df = pl.DataFrame(r_rows)

    # ---- 1) factor_rank_col 开/关：同一有效集上预计算秩应与自算逐位一致 ----
    merged = factor_df.join(ret_df, on=["trade_date", "ts_code"], how="inner")
    for ret_col in [f"fwd_ret_{h}d" for h in horizons]:
        self_ic = _rank_ic_by_date(merged, "factor_clean", ret_col)
        valid = merged.filter(
            pl.col("factor_clean").is_not_null()
            & pl.col("factor_clean").is_finite()
            & pl.col(ret_col).is_not_null()
            & pl.col(ret_col).is_finite()
        )
        if valid.is_empty():
            continue
        with_fr = valid.with_columns(
            pl.col("factor_clean")
            .rank(method="average")
            .over("trade_date")
            .alias("_factor_rank")
        )
        reuse_ic = _rank_ic_by_date(
            with_fr, "factor_clean", ret_col, factor_rank_col="_factor_rank"
        )
        assert reuse_ic.equals(self_ic), f"factor_rank_col 复用与自算不一致 ret={ret_col}"

    # ---- 2) compute_rank_ic 与内联旧实现（无缓存、无 factor_rank_col）逐位一致 ----
    def _legacy_compute_rank_ic(factor_df, daily_ret, factor_col="factor_clean", horizons=None, frequency="daily", oos_split=0.7):
        if horizons is None:
            horizons = [1, 5, 10, 20]
        merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")
        ic_series = _rank_ic_by_date(merged, factor_col, "fwd_ret_1d")
        ic_values = ic_series["ic"].drop_nulls().drop_nans().to_numpy()
        ic_mean, ic_std, ir, ic_pos, tstat, pvalue = _ic_stats(ic_values)
        decay = {}
        for h in horizons:
            ret_col = f"fwd_ret_{h}d"
            if ret_col not in merged.columns:
                continue
            h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
            h_vals = h_ic_df["ic"].drop_nulls().drop_nans().to_numpy()
            if len(h_vals) > 0:
                decay[h] = float(np.mean(h_vals))
        multi_period = {}
        for h in horizons:
            ret_col = f"fwd_ret_{h}d"
            if ret_col not in merged.columns:
                continue
            h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
            h_vals = h_ic_df["ic"].drop_nulls().drop_nans().to_numpy()
            if len(h_vals) > 0:
                h_mean, h_std, h_ir, h_pos, h_t, h_p = _ic_stats(h_vals)
                multi_period[h] = {
                    "ic_mean": h_mean,
                    "ic_std": h_std,
                    "ir": h_ir,
                    "ic_positive_ratio": h_pos,
                    "tstat": h_t,
                    "pvalue": h_p,
                }
        oos_ic = {}
        if len(ic_values) >= 4:
            n_train = max(2, int(len(ic_values) * oos_split))
            oos_ic["train"] = float(np.mean(ic_values[:n_train]))
            oos_ic["test"] = float(np.mean(ic_values[n_train:]))
        walk_forward_ic = _compute_walk_forward_ic(ic_values, n_folds=5, embargo=5)
        return ICAnalysisResult(
            factor_name=factor_col,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_positive_ratio=ic_pos,
            n_periods=len(ic_values),
            ic_series=ic_series,
            decay=decay,
            frequency=frequency,
            ic_tstat=tstat,
            ic_pvalue=pvalue,
            multi_period=multi_period,
            oos_ic=oos_ic,
            walk_forward_ic=walk_forward_ic,
        )

    new_res = compute_rank_ic(factor_df, ret_df, factor_col="factor_clean", horizons=horizons)
    old_res = _legacy_compute_rank_ic(factor_df, ret_df, factor_col="factor_clean", horizons=horizons)

    assert new_res.ic_mean == old_res.ic_mean
    assert new_res.ic_std == old_res.ic_std
    assert new_res.ir == old_res.ir
    assert new_res.ic_tstat == old_res.ic_tstat
    assert new_res.ic_pvalue == old_res.ic_pvalue
    assert new_res.n_periods == old_res.n_periods
    assert new_res.ic_positive_ratio == old_res.ic_positive_ratio
    assert new_res.decay == old_res.decay
    assert new_res.multi_period == old_res.multi_period
    assert new_res.oos_ic == old_res.oos_ic
    assert new_res.walk_forward_ic == old_res.walk_forward_ic
    assert new_res.ic_series.equals(old_res.ic_series)

