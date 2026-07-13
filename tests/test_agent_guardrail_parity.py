# tests/test_agent_guardrail_parity.py
"""Workstream B：护栏双路径对齐 + PBO。抽共享 guardrail_passed，M1/M5/M6 统一 p 值口径。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from factorzen.discovery.guardrails import guardrail_passed


def test_guardrail_passed_positive_ic_significant():
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                            ci_low=0.01, ci_high=0.08) is True


def test_guardrail_passed_rejects_insignificant_dsr():
    """DSR p 值不显著（0.3>0.05）→ 拒。旧 M5/M6 口径 dsr>0.5(即 pval<0.5)会误放行。"""
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.3,
                            ci_low=0.01, ci_high=0.08) is False


def test_guardrail_passed_rejects_sign_mismatch():
    assert guardrail_passed(ic_train=0.05, holdout_ic=-0.04, dsr_pvalue=0.01,
                            ci_low=0.01, ci_high=0.08) is False


def test_guardrail_passed_ci_crosses_zero_now_allowed():
    """2026-07 松一档：移除 holdout CI 单边门。CI 跨零不再单独否决——
    holdout 方向仅由点估计同号把关（DSR 显著 + holdout 与 train 同号即过）。"""
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                            ci_low=-0.01, ci_high=0.08) is True


def test_guardrail_passed_negative_ic_bidirectional():
    assert guardrail_passed(ic_train=-0.05, holdout_ic=-0.04, dsr_pvalue=0.01,
                            ci_low=-0.08, ci_high=-0.01) is True


def test_guardrail_passed_none_and_nan_conservative():
    assert guardrail_passed(ic_train=None, holdout_ic=0.04, dsr_pvalue=0.01, ci_low=0.01) is False
    nan = float("nan")
    assert guardrail_passed(ic_train=0.05, holdout_ic=nan, dsr_pvalue=0.01, ci_low=0.01) is False


def test_guardrail_negative_ic_without_ci_high_falls_back_to_ci_low():
    assert guardrail_passed(ic_train=-0.05, holdout_ic=-0.04, dsr_pvalue=0.01, ci_low=0.01) is True


def test_cross_path_parity_m1_delegates_to_shared():
    """M1 的 _guard_passed 委托共享入口 acceptance_reasons，逐样本无漂移：
    strict 口径 == guardrail_passed(DSR)，library 口径(默认) == not library_reasons。"""
    from factorzen.discovery.guardrails import library_reasons
    from factorzen.discovery.mining_session import _guard_passed
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = {"ic_train": float(rng.normal(0, 0.05)), "holdout_ic": float(rng.normal(0, 0.05)),
             "dsr_pvalue": float(rng.uniform(0, 1)), "ic_ci_low": float(rng.normal(0, 0.03))}
        strict = guardrail_passed(ic_train=c["ic_train"], holdout_ic=c["holdout_ic"],
                                  dsr_pvalue=c["dsr_pvalue"], ci_low=c["ic_ci_low"], dsr_alpha=0.05)
        assert _guard_passed(c, dsr_alpha=0.05, gate="strict") == strict
        library = not library_reasons(ic_train=c["ic_train"], holdout_ic=c["holdout_ic"])
        assert _guard_passed(c) == library, f"library drift: {c}"


def _mk_daily(n_days=260, n_stocks=30, seed=7):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for c in codes:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99,
                         "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7})
    return pl.DataFrame(rows)


def _run_guardrails_with(n_candidates: int):
    """跑 node_guardrails，用 stub 让指定数量的候选过护栏，返回 state。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily()
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.3)
    bundle = DataBundle.build(mining_df)
    state = AgentState(seed=1)
    # 用互不相关的表达式，避免被去相关剔除（corr>0.7）
    exprs = ["ts_mean(close, 5)", "rank(neg(vol))", "ts_std(close, 10)"][:n_candidates]
    for e in exprs:
        state.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=e, compile_ok=True,
            ic_train=0.05, passed_guardrails=False, critic_verdict=None, error=None,
            ir_train=1.2, n_train=100))

    import factorzen.discovery.guardrails as gmod
    import factorzen.validation.holdout as hmod
    from factorzen.validation.holdout import HoldoutICResult
    orig_hic, orig_pass = hmod.holdout_ic_result, gmod.acceptance_reasons
    # 覆盖充足 + 同号；acceptance_reasons 恒空 → 全过（与旧 mock guardrail_passed 等价）
    hmod.holdout_ic_result = lambda _f, _h: HoldoutICResult(
        0.05, 0.5, (0.01, 0.09), n_days=100)
    gmod.acceptance_reasons = lambda **_kw: []
    try:
        node_guardrails(state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=TrialLedger(), top_k=5, warmup_daily=daily)
    finally:
        hmod.holdout_ic_result, gmod.acceptance_reasons = orig_hic, orig_pass
    return state


def test_node_guardrails_pbo_is_a_probability_when_pool_is_big_enough():
    """PBO 是「过拟合概率」，必须落在 [0, 1]。

    旧断言 `state.pbo is None or isinstance(state.pbo, float)` 是**恒真**的——nan 也是 float，
    且单候选时 `pool_pbo` 本就返回 nan，所以它连「PBO 是不是概率」都没验证。
    """
    state = _run_guardrails_with(n_candidates=3)

    assert len(state.candidates) >= 2, "需要 ≥2 个候选，PBO(CSCV) 才有定义"
    assert state.pbo == state.pbo, "候选足够时 PBO 不该是 nan"
    assert 0.0 <= state.pbo <= 1.0, f"PBO 是概率，必须 ∈ [0,1]，实得 {state.pbo}"


def test_node_guardrails_pbo_is_nan_when_pool_too_small():
    """候选 < 2 时 CSCV 无从切分 → nan（而非 0.0 之类会被误读为「无过拟合」的值）。"""
    import math

    state = _run_guardrails_with(n_candidates=1)

    assert len(state.candidates) == 1
    assert math.isnan(state.pbo), f"单候选时 PBO 应为 nan，实得 {state.pbo}"
