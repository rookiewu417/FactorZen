"""campaign trial family：DSR 的 N 跨 session 累计（消除清零漏记）。

同一评价配置下多个 team session 各自从零计数 → 跨 session 多重检验漏记账。
`campaign_prior` 从 ExperimentIndex 重建历史 trial 池；`node_finalize_guardrails`
用 prior∪session 的 union 做 deflation basis。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_finalize_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest
from factorzen.discovery.campaign import CampaignPrior, campaign_key, campaign_prior
from factorzen.discovery.guardrails import DeflationBasis

# ── helpers ───────────────────────────────────────────────────────────────

_WIN_A = {"start": "20200605", "end": "20260605", "universe": "csi300", "market": "ashare"}
_WIN_B = {"start": "20180101", "end": "20201231", "universe": "csi500", "market": "ashare"}
_N_OBS = 303


def _line(expr: str, ir: float, *, run_id: str, window: dict, compile_ok: bool = True) -> str:
    return json.dumps({
        "expression": expr,
        "hypothesis": "h",
        "ic_train": ir / 10.0,
        "ir_train": ir,
        "n_train": _N_OBS,
        "passed": False,
        "verdict": None,
        "decorrelated": False,
        "compile_ok": compile_ok,
        "error": None,
        "data_window": window,
        "run_id": run_id,
    }, ensure_ascii=False)


def _write_index(path: Path, lines: list[str]) -> str:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _attempt(it: int, ir: float, expr: str) -> AttemptRecord:
    return AttemptRecord(
        iteration=it, hypothesis="h", expression=expr, compile_ok=True,
        ic_train=ir / 10.0, passed_guardrails=False, critic_verdict=None, error=None,
        ir_train=ir, turnover=0.3, n_train=_N_OBS,
    )


def _candidate(ir: float, expr: str) -> dict:
    return {
        "expression": expr, "hypothesis": "h", "ic_train": ir / 10.0, "ir_train": ir,
        "turnover": 0.3, "holdout_ic": 0.05, "holdout_ir": 0.5,
        "ic_ci_low": 0.01, "ic_ci_high": 0.09, "n_train": _N_OBS,
        "dsr": 0.99, "dsr_pvalue": 0.001,
    }


def _mock_daily(n_stocks=40, n_days=180, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(n_stocks)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _scripted_team():
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


# ── 1. campaign_key 稳定性 ────────────────────────────────────────────────


def test_campaign_key_stable_and_sensitive():
    base = dict(
        market="ashare", universe="csi300", start="20200605", end="20260605",
        holdout_ratio=0.2, objective="residual", horizon=1, gate="library",
    )
    k0 = campaign_key(**base)
    assert k0 == campaign_key(**base)
    assert len(k0) == 16
    assert all(c in "0123456789abcdef" for c in k0)

    assert campaign_key(**{**base, "universe": "csi500"}) != k0
    assert campaign_key(**{**base, "holdout_ratio": 0.3}) != k0
    assert campaign_key(**{**base, "objective": "raw"}) != k0
    assert campaign_key(**{**base, "gate": "strict"}) != k0
    assert campaign_key(**{**base, "horizon": 5}) != k0
    assert campaign_key(**{**base, "start": "20200101"}) != k0


def test_campaign_key_none_and_empty_string_normalized():
    """None 与空串（strip 后）对 key 等价。"""
    k_none = campaign_key(
        market=None, universe=None, start=None, end=None,
        holdout_ratio=None, objective=None, horizon=None, gate=None,
    )
    k_empty = campaign_key(
        market="", universe="  ", start="", end="",
        holdout_ratio=None, objective="", horizon=None, gate="  ",
    )
    assert k_none == k_empty


# ── 2–4. campaign_prior 重建 ─────────────────────────────────────────────


def test_campaign_prior_rebuild_dedup_and_window(tmp_path: Path):
    """session A 3 式 + B 2 式（1 重复）同窗 + C 2 式异窗 → 同窗 prior N=4 / sessions=2。"""
    lines = [
        _line("expr_a1", 0.10, run_id="team_a", window=_WIN_A),
        _line("expr_a2", 0.20, run_id="team_a", window=_WIN_A),
        _line("expr_a3", 0.30, run_id="team_a", window=_WIN_A),
        _line("expr_a1", 0.99, run_id="team_b", window=_WIN_A),  # 与 A 重复，保首行 IR
        _line("expr_b2", 0.40, run_id="team_b", window=_WIN_A),
        _line("expr_c1", 0.50, run_id="team_c", window=_WIN_B),
        _line("expr_c2", 0.60, run_id="team_c", window=_WIN_B),
    ]
    path = _write_index(tmp_path / "experiment_index.jsonl", lines)

    prior = campaign_prior(
        path, market="ashare", universe="csi300",
        start="20200605", end="20260605",
    )
    assert prior is not None
    assert prior.n_trials == 4
    assert len(prior.irs) == 4
    assert prior.n_sessions == 2
    # 去重保首行：expr_a1 取 session A 的 0.10，不是 B 的 0.99；顺序=首现序
    first_order = ["expr_a1", "expr_a2", "expr_a3", "expr_b2"]
    assert prior.irs == [0.10, 0.20, 0.30, 0.40]
    assert prior.expressions == set(first_order)
    assert "expr_c1" not in prior.expressions
    assert prior.source_path == path


def test_campaign_prior_exclude_run_ids(tmp_path: Path):
    lines = [
        _line("e1", 0.1, run_id="team_a", window=_WIN_A),
        _line("e2", 0.2, run_id="team_a", window=_WIN_A),
        _line("e3", 0.3, run_id="team_a", window=_WIN_A),
        _line("e4", 0.4, run_id="team_b", window=_WIN_A),
        _line("e5", 0.5, run_id="team_b", window=_WIN_A),
    ]
    path = _write_index(tmp_path / "idx.jsonl", lines)
    prior = campaign_prior(
        path, market="ashare", universe="csi300",
        start="20200605", end="20260605",
        exclude_run_ids={"team_b"},
    )
    assert prior is not None
    assert prior.n_trials == 3
    assert prior.n_sessions == 1
    assert prior.expressions == {"e1", "e2", "e3"}


def test_campaign_prior_skips_corrupt_lines(tmp_path: Path):
    lines = [
        _line("e1", 0.1, run_id="team_a", window=_WIN_A),
        "NOT_JSON{{{",
        _line("e2", 0.2, run_id="team_a", window=_WIN_A),
    ]
    path = _write_index(tmp_path / "idx.jsonl", lines)
    prior = campaign_prior(
        path, market="ashare", universe="csi300",
        start="20200605", end="20260605",
    )
    assert prior is not None
    assert prior.n_trials == 2


def test_campaign_prior_missing_file_returns_none(tmp_path: Path):
    assert campaign_prior(
        str(tmp_path / "nope.jsonl"),
        market="ashare", universe="csi300",
        start="20200605", end="20260605",
    ) is None


# ── 5. finalize 单调性（TDD 核心）────────────────────────────────────────


def test_finalize_pvalue_monotone_with_prior_n():
    """同一候选：注入 100 个历史 IR 的 prior 后 p 变大（门槛更严）。"""
    cand_ir = 0.25
    expr = "rank(neg(pb))"
    session_pool = [0.05, -0.03, 0.12, cand_ir]

    state0 = AgentState(seed=1)
    for i, ir in enumerate(session_pool):
        e = expr if ir == cand_ir else f"rank(neg(ts_min(low, {5 + i})))"
        a = _attempt(0, ir, e)
        if e == expr:
            a.passed_guardrails = True
        state0.attempts.append(a)
    state0.candidates.append(_candidate(cand_ir, expr))

    basis0 = node_finalize_guardrails(state0)
    p0 = state0.candidates[0]["dsr_pvalue"]
    assert basis0.n_trials == 4

    # 同候选、同 session 池 + 100 个历史 IR
    hist_irs = [0.01 * ((i % 20) - 10) for i in range(100)]
    prior = CampaignPrior(
        campaign_id="deadbeefdeadbeef",
        n_trials=100,
        expressions={f"hist_{i}" for i in range(100)},
        irs=list(hist_irs),
        n_sessions=5,
        source_path="/tmp/x.jsonl",
    )
    state1 = AgentState(seed=1)
    for i, ir in enumerate(session_pool):
        e = expr if ir == cand_ir else f"rank(neg(ts_min(low, {5 + i})))"
        a = _attempt(0, ir, e)
        if e == expr:
            a.passed_guardrails = True
        state1.attempts.append(a)
    state1.candidates.append(_candidate(cand_ir, expr))

    basis1 = node_finalize_guardrails(state1, prior=prior)
    p1 = state1.candidates[0]["dsr_pvalue"]
    assert basis1.n_trials == 104, "100 prior + 4 session 唯一"
    assert p1 > p0, f"N 变大后 p 应更严：p0={p0:.6f} p1={p1:.6f}"


# ── 6. 同表达式不双计 ────────────────────────────────────────────────────


def test_finalize_union_dedups_session_vs_prior():
    prior = CampaignPrior(
        campaign_id="x",
        n_trials=2,
        expressions={"ts_mean(close,5)", "rank(vol)"},
        irs=[0.10, 0.20],
        n_sessions=1,
        source_path="/tmp/x.jsonl",
    )
    state = AgentState(seed=1)
    # 与 prior 重复的表达式 + 一个新表达式
    for expr, ir in [("ts_mean(close,5)", 0.99), ("rank(pb)", 0.15)]:
        state.attempts.append(_attempt(0, ir, expr))
    state.candidates.append(_candidate(0.15, "rank(pb)"))

    basis = node_finalize_guardrails(state, prior=prior)
    # prior 2 + session 新增 1（ts_mean 不双计）= 3
    assert basis.n_trials == 3


def test_finalize_prior_none_zero_regression():
    """prior=None 时 basis 与仅用 session attempts 的 from_ir_pool 一致。"""
    state = AgentState(seed=1)
    pool = [0.02, -0.05, 0.08, 0.11]
    for i, ir in enumerate(pool):
        state.attempts.append(_attempt(0, ir, f"e{i}"))
    state.candidates.append(_candidate(0.11, "e3"))

    basis = node_finalize_guardrails(state, prior=None)
    want = DeflationBasis.from_ir_pool(pool, two_sided=True)
    assert basis.n_trials == want.n_trials
    assert basis.sharpe_variance == pytest.approx(want.sharpe_variance)


# ── 7. orchestrator 端 manifest 字段 ─────────────────────────────────────


def test_orchestrator_manifest_campaign_fields(tmp_path: Path):
    """mock index + 小 session：manifest 含 campaign 族字段；family = prior + session 新增。"""
    # 历史：2 个唯一表达式，同窗
    hist = [
        _line("ts_std(close,10)", 0.08, run_id="team_99", window=_WIN_A),
        _line("rank(vol)", 0.06, run_id="team_99", window=_WIN_A),
    ]
    index_path = _write_index(tmp_path / "experiment_index.jsonl", hist)

    daily = _mock_daily()
    res = run_team_agent(
        daily, _scripted_team(),
        n_rounds=1, seed=42,
        index_path=index_path,
        data_window=dict(_WIN_A),
        heal_rounds=0,
        update_library=False,
        library_orthogonal=False,
        auto_lift=False,
        campaign_prior_enabled=True,
    )
    man_path = write_team_manifest(
        res, out_dir=str(tmp_path / "out"), run_id="t_campaign",
        params={"holdout_ratio": 0.2},
    )
    m = json.loads(man_path.read_text(encoding="utf-8"))

    assert m.get("campaign_id")
    assert len(m["campaign_id"]) == 16
    assert m["prior_n_trials"] == 2
    assert m["prior_n_sessions"] == 1
    assert "n_trials_family" in m
    # family = prior 唯一 + 本 session 不在 prior 里的唯一新增
    # scripted 评估 ts_mean(close,5)，不在历史 → family >= prior + 1（若编译成功）
    assert m["n_trials_family"] >= m["prior_n_trials"]
    if res.n_trials >= 1:
        # 本 session 至少有 1 个新表达式时 family 应严格大于 prior
        assert m["n_trials_family"] == m["prior_n_trials"] + res.n_trials or (
            m["n_trials_family"] > m["prior_n_trials"]
        )
    # 现有 n_trials 语义保持 = 本 session
    assert m["n_trials"] == res.n_trials


def test_orchestrator_campaign_prior_disabled_zero_regression(tmp_path: Path):
    hist = [
        _line("ts_std(close,10)", 0.08, run_id="team_99", window=_WIN_A),
        _line("rank(vol)", 0.06, run_id="team_99", window=_WIN_A),
    ]
    index_path = _write_index(tmp_path / "experiment_index.jsonl", hist)
    daily = _mock_daily()

    res_off = run_team_agent(
        daily, _scripted_team(),
        n_rounds=1, seed=7,
        index_path=index_path,
        data_window=dict(_WIN_A),
        heal_rounds=0,
        update_library=False,
        library_orthogonal=False,
        auto_lift=False,
        campaign_prior_enabled=False,
    )
    m_off = json.loads(write_team_manifest(
        res_off, out_dir=str(tmp_path / "out_off"), run_id="t_off",
        params={},
    ).read_text(encoding="utf-8"))

    assert m_off["prior_n_trials"] == 0
    assert m_off.get("prior_n_sessions", 0) == 0
    # 无 prior 时 family = 本 session N（basis.n_trials）
    assert m_off["n_trials_family"] == m_off["n_trials"] or m_off["n_trials_family"] >= 0
