# tests/test_agent_patience.py
"""Workstream G：自适应终止（连续 patience 轮无新 passed 候选则早停）。"""
from __future__ import annotations

import datetime as dt
import inspect
import json

import numpy as np
import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent


def _mock_daily(n_stocks=20, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _fn_invalid():
    """始终产非法表达式 → 永无候选过护栏 → 触发 patience 早停。"""
    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["not_a_func("]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def test_team_patience_early_stops(tmp_path):
    res = run_team_agent(_mock_daily(), _fn_invalid(), n_rounds=8, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), patience=2)
    assert res.state.iteration < 8, f"patience 未早停, iteration={res.state.iteration}"


def test_team_patience_none_runs_all_rounds(tmp_path):
    """patience=None（默认）→ 跑满 n_rounds（零回归）。"""
    res = run_team_agent(_mock_daily(), _fn_invalid(), n_rounds=3, seed=1,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 3


def test_m5_orchestrator_accepts_patience():
    from factorzen.agents.orchestrator import run_llm_agent
    assert "patience" in inspect.signature(run_llm_agent).parameters
