# tests/test_agent_structured.py
"""Workstream C：结构化假设（RD-Agent 步1）+ 任务分解（步2）。"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from factorzen.agents.roles.coder import decompose_tasks
from factorzen.agents.roles.hypothesis import format_structured, propose_structured


def test_propose_structured_returns_four_fields():
    def fake(_m):
        return json.dumps({"hypotheses": [{"direction": "高动量", "mechanism": "趋势延续",
                                           "expected_sign": 1, "falsification": "IC<0则证伪"}]})
    out = propose_structured(fake, known_invalid=[], known_valid=[])
    assert len(out) == 1
    for k in ["direction", "mechanism", "expected_sign", "falsification"]:
        assert k in out[0]
    assert out[0]["expected_sign"] == 1


def test_propose_structured_skips_malformed():
    def fake(_m):
        return json.dumps({"hypotheses": [{"no_direction": "x"}, {"direction": "ok"}]})
    out = propose_structured(fake, known_invalid=[], known_valid=[])
    assert len(out) == 1 and out[0]["direction"] == "ok"


def test_format_structured_renders_fields():
    h = {"direction": "高动量", "mechanism": "趋势延续", "expected_sign": 1, "falsification": "IC<0"}
    txt = format_structured(h)
    assert "高动量" in txt and "机制" in txt and "证伪" in txt


def test_decompose_tasks_returns_tasks():
    def fake(_m):
        return json.dumps({"tasks": [{"name": "mom20", "description": "20日动量", "rationale": "趋势"}]})
    tasks = decompose_tasks("高动量", fake)
    assert len(tasks) == 1
    assert tasks[0]["name"] == "mom20" and tasks[0]["rationale"] == "趋势"


def _mock_daily():
    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(20)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def test_team_structured_opt_in_closes_loop(tmp_path):
    """structured=True 走结构化假设路径，闭环完成不崩。"""
    from factorzen.agents.team_orchestrator import run_team_agent
    seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                       "expected_sign": 1, "falsification": "f"}]}),
           json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})] * 20
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    res = run_team_agent(_mock_daily(), fn, n_rounds=1, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), structured=True)
    assert res.state.iteration == 1
