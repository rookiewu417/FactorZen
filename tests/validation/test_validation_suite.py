"""test_validation_bootstrap.py：block bootstrap IC 置信区间正/噪声/过短样本
test_validation_deflated_sharpe.py：deflated Sharpe 显著性与试验数收紧
test_validation_holdout.py：holdout 时间切分隔离与 holdout IC
test_validation_multiple_testing.py：TrialLedger 试验计数累计
test_validation_pbo.py：PBO 噪声≈0.5 / 主导因子偏低 / 过小返回 nan
"""

from datetime import date, timedelta

import numpy as np
import polars as pl


# ==== 来自 test_validation_bootstrap.py ====
def test_ic_ci_suite():
    """test_positive_ic_ci_above_zero；test_noise_ic_ci_straddles_zero；test_too_short_returns_nan"""
    # -- 原 test_positive_ic_ci_above_zero --
    def _section_0_test_positive_ic_ci_above_zero():
        from factorzen.validation.bootstrap import block_bootstrap_ic_ci
        rng = np.random.default_rng(0)
        ic = rng.normal(0.05, 0.02, 250)  # 明显正 IC
        lo, hi = block_bootstrap_ic_ci(ic, seed=1)
        assert lo > 0 and hi > lo

    _section_0_test_positive_ic_ci_above_zero()

    # -- 原 test_noise_ic_ci_straddles_zero --
    def _section_1_test_noise_ic_ci_straddles_zero():
        from factorzen.validation.bootstrap import block_bootstrap_ic_ci
        rng = np.random.default_rng(0)
        ic = rng.normal(0.0, 0.05, 250)  # 噪声 IC
        lo, hi = block_bootstrap_ic_ci(ic, seed=1)
        assert lo < 0 < hi

    _section_1_test_noise_ic_ci_straddles_zero()

    # -- 原 test_too_short_returns_nan --
    def _section_2_test_too_short_returns_nan():
        from factorzen.validation.bootstrap import block_bootstrap_ic_ci
        lo, hi = block_bootstrap_ic_ci(np.array([0.1, 0.2]), block_size=10)
        assert np.isnan(lo) and np.isnan(hi)

    _section_2_test_too_short_returns_nan()


# ==== 来自 test_validation_deflated_sharpe.py ====
def test_sharpe_significance_suite():
    """test_strong_sharpe_significant；test_noise_sharpe_not_significant；test_more_trials_tightens；test_expected_max_sharpe_grows_with_trials"""
    # -- 原 test_strong_sharpe_significant --
    def _section_0_test_strong_sharpe_significant():
        from factorzen.validation.deflated_sharpe import deflated_sharpe
        # 高 IR、长样本、少试验 → 应显著
        dsr, p = deflated_sharpe(sharpe=0.15, n_trials=5, n_obs=500, sharpe_variance=0.0025)
        assert dsr > 0.95 and p < 0.05

    _section_0_test_strong_sharpe_significant()

    # -- 原 test_noise_sharpe_not_significant --
    def _section_1_test_noise_sharpe_not_significant():
        from factorzen.validation.deflated_sharpe import deflated_sharpe
        # IR≈0 → 不显著
        _dsr, p = deflated_sharpe(sharpe=0.0, n_trials=100, n_obs=500, sharpe_variance=0.0025)
        assert p > 0.05

    _section_1_test_noise_sharpe_not_significant()

    # -- 原 test_more_trials_tightens --
    def _section_2_test_more_trials_tightens():
        from factorzen.validation.deflated_sharpe import deflated_sharpe
        # 同样观测 Sharpe，更多试验 → DSR 下降（多重检验收紧）
        dsr_few, _ = deflated_sharpe(0.12, n_trials=5, n_obs=500, sharpe_variance=0.0025)
        dsr_many, _ = deflated_sharpe(0.12, n_trials=1000, n_obs=500, sharpe_variance=0.0025)
        assert dsr_many < dsr_few

    _section_2_test_more_trials_tightens()

    # -- 原 test_expected_max_sharpe_grows_with_trials --
    def _section_3_test_expected_max_sharpe_grows_with_trials():
        from factorzen.validation.deflated_sharpe import expected_max_sharpe
        assert expected_max_sharpe(0.0025, 1000) > expected_max_sharpe(0.0025, 10)

    _section_3_test_expected_max_sharpe_grows_with_trials()


# ==== 来自 test_validation_holdout.py ====
# tests/test_validation_holdout.py

def _daily(n_stocks=40, n_days=200, seed=1):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)

def test_holdout_and_ledger_suite():
    """test_split_holdout_disjoint_and_isolated；test_holdout_ic_runs；test_trial_ledger_accumulates"""
    # -- 原 test_split_holdout_disjoint_and_isolated --
    def _section_0_test_split_holdout_disjoint_and_isolated():
        from factorzen.validation.holdout import split_holdout
        daily = _daily()
        mining, holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
        # 隔离：mining 全部 < holdout_start ≤ holdout 全部
        assert mining["trade_date"].max() < hstart
        assert holdout["trade_date"].min() >= hstart
        # holdout 约占 20%
        frac = holdout["trade_date"].n_unique() / daily["trade_date"].n_unique()
        assert 0.15 < frac < 0.25

    _section_0_test_split_holdout_disjoint_and_isolated()

    # -- 原 test_holdout_ic_runs --
    def _section_1_test_holdout_ic_runs():
        from factorzen.validation.holdout import holdout_ic, split_holdout
        daily = _daily()
        _mining, holdout, _ = split_holdout(daily, holdout_ratio=0.2)
        # 用「次日收益」当因子 → holdout IC 应为正
        fac = holdout.sort(["ts_code", "trade_date"]).with_columns(
            (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("factor_value")
        ).select(["trade_date", "ts_code", "factor_value"]).drop_nulls()
        ic_mean, _ir, (lo, hi) = holdout_ic(fac, holdout)
        assert ic_mean > 0.05 and lo <= hi

    _section_1_test_holdout_ic_runs()

    # -- 原 test_trial_ledger_accumulates --
    def _section_2_test_trial_ledger_accumulates():
        from factorzen.validation.multiple_testing import TrialLedger
        led = TrialLedger()
        assert led.n_trials == 0
        led.record()
        led.record(5)
        assert led.n_trials == 6

    _section_2_test_trial_ledger_accumulates()


# ==== 来自 test_validation_multiple_testing.py ====

# ==== 来自 test_validation_pbo.py ====
def test_pbo_suite():
    """纯噪声候选池：IS 最优在 OOS 无优势 → PBO ≈ 0.5。；一个候选全程显著最优 → IS 最优 = OOS 最优 → PBO 低。；test_pbo_too_small_returns_nan"""
    # -- 原 test_pbo_noise_near_half --
    def _section_0_test_pbo_noise_near_half():
        from factorzen.validation.pbo import compute_pbo
        rng = np.random.default_rng(0)
        perf = rng.normal(0, 1, (20, 200))
        pbo = compute_pbo(perf, n_splits=10)
        assert 0.3 < pbo < 0.7

    _section_0_test_pbo_noise_near_half()

    # -- 原 test_pbo_one_dominant_low --
    def _section_1_test_pbo_one_dominant_low():
        from factorzen.validation.pbo import compute_pbo
        rng = np.random.default_rng(0)
        perf = rng.normal(0, 1, (20, 200))
        perf[0] += 3.0  # 候选0 全程领先
        pbo = compute_pbo(perf, n_splits=10)
        assert pbo < 0.2

    _section_1_test_pbo_one_dominant_low()

    # -- 原 test_pbo_too_small_returns_nan --
    def _section_2_test_pbo_too_small_returns_nan():
        from factorzen.validation.pbo import compute_pbo
        assert np.isnan(compute_pbo(np.zeros((1, 100)), n_splits=10))

    _section_2_test_pbo_too_small_returns_nan()


