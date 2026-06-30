# tests/test_team_orchestrator.py
import datetime as dt
import json
from pathlib import Path

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


def _scripted_team():
    """Hypothesis→Coder→Critic(keep) 一轮脚本，循环复用。"""
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


def test_run_team_closes_loop(tmp_path: Path):
    daily = _mock_daily()
    res = run_team_agent(daily, _scripted_team(), n_rounds=2, seed=42,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 2
    assert res.n_trials >= 1
    assert len(res.rounds_log) >= 1     # 角色决策可审计


def test_run_team_revise_loop_counts_n(tmp_path: Path):
    """轮1 Critic revise_expr → 轮2 Coder 改写（跨轮 feedback），两表达式都评估、都计入 N。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code1 = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit_revise = json.dumps({"verdict": "revise_expr", "reason": "窗口太短"})
    code2 = json.dumps({"expressions": ["ts_mean(close,20)"]})  # 下一轮 revise 产物
    crit_keep = json.dumps({"verdict": "keep", "reason": "ok"})
    # 轮1: propose,write,critic(revise) ; 轮2: revise(不再 propose),critic(keep)
    seq = [hyp, code1, crit_revise, code2, crit_keep]
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"]] if i["k"] < len(seq) else crit_keep
        i["k"] += 1
        return v

    daily = _mock_daily()
    res = run_team_agent(daily, fn, n_rounds=2, seed=1, index_path=str(tmp_path / "e.jsonl"))
    assert res.n_trials >= 2     # 两轮各评估一个表达式(原始 + 改写)，都计入 N
    assert any("ts_mean(close, 20)" in r["expressions"] for r in res.rounds_log)  # 轮2 是改写产物


def test_cross_session_dedup(tmp_path: Path):
    """共享 experiment_index：第二次 run 重复表达式被跳过（seen 去重）。"""
    daily = _mock_daily()
    idx_path = str(tmp_path / "shared.jsonl")
    run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
    res2 = run_team_agent(daily, _scripted_team(), n_rounds=1, seed=1, index_path=idx_path)
    # 第二次 run 产同样的 ts_mean(close,5)，已在 index → 本轮无新评估（n_trials 可能为 0）
    assert res2.n_trials == 0 or all(
        a.expression != "ts_mean(close, 5)" for a in res2.state.attempts)
