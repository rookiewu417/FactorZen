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


def test_guardrail_passed_rejects_ci_crosses_zero():
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                            ci_low=-0.01, ci_high=0.08) is False


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
    from factorzen.discovery.mining_session import _guard_passed
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = {"ic_train": float(rng.normal(0, 0.05)), "holdout_ic": float(rng.normal(0, 0.05)),
             "dsr_pvalue": float(rng.uniform(0, 1)), "ic_ci_low": float(rng.normal(0, 0.03))}
        m1 = _guard_passed(c, dsr_alpha=0.05)
        shared = guardrail_passed(ic_train=c["ic_train"], holdout_ic=c["holdout_ic"],
                                  dsr_pvalue=c["dsr_pvalue"], ci_low=c["ic_ci_low"], dsr_alpha=0.05)
        assert m1 == shared, f"drift: {c} -> m1={m1} shared={shared}"


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


def test_node_guardrails_records_pbo():
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily()
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.3)
    bundle = DataBundle.build(mining_df)
    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="ts_mean(close, 5)", compile_ok=True,
        ic_train=0.05, passed_guardrails=False, critic_verdict=None, error=None, ir_train=1.2))
    node_guardrails(state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                    ledger=TrialLedger(), top_k=5)
    assert hasattr(state, "pbo")
    assert state.pbo is None or isinstance(state.pbo, float)
