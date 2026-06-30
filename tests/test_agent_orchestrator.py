# tests/test_agent_orchestrator.py
import datetime as dt
import json

import numpy as np
import polars as pl

from factorzen.agents.orchestrator import run_llm_agent


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


def _scripted_llm():
    """每轮：1 个 proposal + semantic(pass) + critic(keep)。无限循环复用。"""
    prop = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [prop, sem, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def test_run_llm_agent_closes_loop():
    daily = _mock_daily()
    res = run_llm_agent(daily, _scripted_llm(), n_rounds=3, seed=42)
    assert res.state.iteration == 3
    assert res.n_trials >= 1            # N 累加了
    assert len(res.state.attempts) >= 1


def test_run_llm_agent_reproducible():
    daily = _mock_daily()
    r1 = run_llm_agent(daily, _scripted_llm(), n_rounds=2, seed=7)
    r2 = run_llm_agent(daily, _scripted_llm(), n_rounds=2, seed=7)
    # 同 seed + 同 scripted LLM → 尝试序列逐字节一致
    assert [a.expression for a in r1.state.attempts] == [a.expression for a in r2.state.attempts]
    assert r1.n_trials == r2.n_trials
