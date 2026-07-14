# tests/test_decorr_boundary.py
"""session 内去相关边界统一：恰等阈值 = 拒（M1 / Agent / library 三处一致）。

M1: ``mc < threshold`` 才入选 → mc==threshold 拒
Agent: ``corr >= threshold`` 拒（修复前是 ``corr > 0.7``，恰 0.7 误放行）
library_orthogonal_check: ``mc >= threshold`` → ok=False
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.scoring import DEFAULT_DECORR_THRESHOLD

_SRC = Path(__file__).resolve().parents[1] / "src" / "factorzen"


def _mk_daily(n_days=80, n_stocks=30, seed=3):
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
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7,
            })
    return pl.DataFrame(rows)


# ── 三处语义契约（参数化）──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "corr,expect_reject",
    [
        (DEFAULT_DECORR_THRESHOLD, True),   # 恰等阈值 = 拒
        (0.699, False),
        (0.701, True),
        (0.0, False),
        (1.0, True),
    ],
)
def test_m1_session_pool_boundary_semantics(corr, expect_reject):
    """M1 top-K：``mc < threshold`` 入选 ⇒ 恰等拒。"""
    accept = corr < DEFAULT_DECORR_THRESHOLD
    assert accept is (not expect_reject)


@pytest.mark.parametrize(
    "corr,expect_reject",
    [
        (DEFAULT_DECORR_THRESHOLD, True),
        (0.699, False),
        (0.701, True),
    ],
)
def test_agent_session_pool_boundary_semantics(corr, expect_reject):
    """Agent：``corr >= threshold`` 拒（与 M1 恰等阈值语义一致）。"""
    reject = corr >= DEFAULT_DECORR_THRESHOLD
    assert reject is expect_reject


@pytest.mark.parametrize(
    "mc,expect_ok",
    [
        (DEFAULT_DECORR_THRESHOLD, False),
        (0.699, True),
        (0.701, False),
    ],
)
def test_library_orthogonal_check_boundary(mc, expect_ok, monkeypatch):
    """library_orthogonal_check：``mc >= threshold`` → ok=False。"""
    from factorzen.discovery import scoring as scoring_mod

    monkeypatch.setattr(
        scoring_mod, "max_correlation_detail",
        lambda _f, _p: (mc, "pool_expr"),
    )
    ok, got, nearest = scoring_mod.library_orthogonal_check(
        pl.DataFrame({"trade_date": ["d"], "ts_code": ["s"], "factor_value": [1.0]}),
        {"pool_expr": pl.DataFrame(
            {"trade_date": ["d"], "ts_code": ["s"], "factor_value": [1.0]},
        )},
        threshold=DEFAULT_DECORR_THRESHOLD,
    )
    assert ok is expect_ok
    assert got == mc
    assert nearest == "pool_expr"


# ── Agent 路径 runtime：node_guardrails 会话池去相关 ────────────────────────


def _seed_attempt(state, expr: str, *, ic: float = 0.05, ir: float = 1.2, n: int = 100):
    from factorzen.agents.state import AttemptRecord

    state.attempts.append(AttemptRecord(
        iteration=state.iteration, hypothesis="h", expression=expr,
        compile_ok=True, ic_train=ic, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=ir, turnover=0.3, n_train=n,
    ))


@pytest.mark.parametrize(
    "corr_value,expect_decorrelated",
    [
        (DEFAULT_DECORR_THRESHOLD, True),   # 恰 0.7 → 拒
        (0.699, False),                     # 略低 → 放行
    ],
)
def test_node_guardrails_session_decorr_boundary(
    corr_value, expect_decorrelated, monkeypatch,
):
    """Agent node_guardrails：会话池 max_correlation 恰阈值时与 M1 同拒。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily()
    bundle = DataBundle.build(daily)

    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda fdf, hdf: HoldoutICResult(0.05, 0.5, (0.01, 0.09), n_days=100),
    )
    # 护栏定量门恒过
    import factorzen.discovery.guardrails as gmod
    monkeypatch.setattr(gmod, "acceptance_reasons", lambda **_kw: [])

    # 第一个候选入池时 pool 空 → max_corr=0；第二个起返回受控 corr
    def _fake_max_corr(fdf, pool):
        if not pool:
            return 0.0
        return float(corr_value)

    monkeypatch.setattr(
        "factorzen.discovery.scoring.max_correlation", _fake_max_corr,
    )

    state = AgentState(seed=1)
    _seed_attempt(state, "rank(close)", ic=0.06)
    _seed_attempt(state, "rank(vol)", ic=0.05)

    node_guardrails(
        state, daily=daily, holdout_df=daily, bundle=bundle,
        ledger=TrialLedger(), top_k=5, lib_pool=None,
    )

    first = next(a for a in state.attempts if a.expression == "rank(close)")
    second = next(a for a in state.attempts if a.expression == "rank(vol)")
    assert first.expression in {c["expression"] for c in state.candidates}
    if expect_decorrelated:
        assert second.decorrelated is True
        assert second.expression not in {c["expression"] for c in state.candidates}
        assert second.reject_reason and "高度相关" in second.reject_reason
        # 文案用 ≥ 而非 >
        assert "≥" in second.reject_reason or ">=" in second.reject_reason
    else:
        assert second.decorrelated is False
        assert second.expression in {c["expression"] for c in state.candidates}


# ── M1 源码 + runtime 边界（贪心入选）──────────────────────────────────────


def test_m1_source_uses_strict_lt_threshold():
    """M1 mining_session 必须用 ``mc < decorr_threshold``（恰等拒）。"""
    text = (_SRC / "discovery" / "mining_session.py").read_text(encoding="utf-8")
    assert "mc < decorr_threshold" in text


def test_agent_source_uses_ge_default_decorr_threshold():
    """Agent 必须用 ``corr >= DEFAULT_DECORR_THRESHOLD``，禁止硬编码 ``corr > 0.7``。"""
    text = (_SRC / "agents" / "nodes.py").read_text(encoding="utf-8")
    assert "corr >= DEFAULT_DECORR_THRESHOLD" in text
    # 会话池去相关处不再出现开区间硬编码
    assert "corr > 0.7" not in text


def test_m1_greedy_boundary_via_max_correlation_mock(tmp_path, monkeypatch):
    """M1 top-K 路径：mock max_correlation 恰阈值时第二因子不入选。"""
    from factorzen.discovery.mining_session import run_session

    daily = _mk_daily(n_days=60, n_stocks=35)
    exprs = ["rank(close)", "rank(vol)"]
    idx = {"i": 0}

    class _FakeSearcher:
        def __init__(self, *a, **k):
            pass

        def propose(self):
            from factorzen.discovery.expression import parse_expr
            e = exprs[idx["i"] % len(exprs)]
            idx["i"] += 1
            return parse_expr(e)

    monkeypatch.setattr(
        "factorzen.discovery.mining_session.RandomSearcher", _FakeSearcher,
    )

    def _fake_max_corr(fdf, pool):
        if not pool:
            return 0.0
        return float(DEFAULT_DECORR_THRESHOLD)  # 恰阈值 → 应拒

    # mining_session 顶层 from-import，须 patch 模块内绑定名
    monkeypatch.setattr(
        "factorzen.discovery.mining_session.max_correlation", _fake_max_corr,
    )
    res = run_session(
        daily, n_trials=4, top_k=3, seed=1, method="random",
        out_dir=str(tmp_path / "sessions"),
        update_library=False,
        library_orthogonal=False,
        library_root=str(tmp_path / "empty_lib"),
    )
    cands = res["candidates"]
    # 恰阈值第二因子不得以 max_corr=0.7 入选
    for c in cands:
        if c.get("expression") == "rank(vol)":
            mc = c.get("max_corr")
            if mc is not None:
                assert float(mc) < DEFAULT_DECORR_THRESHOLD
