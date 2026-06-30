# tests/test_team_pipeline.py
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.pipelines.factor_mine_team import run_team_mine


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
    seq = [json.dumps({"hypotheses": ["动量"]}), json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})] * 50
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


def test_run_team_mine_writes_team_manifest(tmp_path: Path):
    daily = _mock_daily()
    res = run_team_mine(daily, n_rounds=2, seed=42, out_dir=str(tmp_path),
                        index_path=str(tmp_path / "e.jsonl"), llm_fn=_scripted_team(),
                        run_id="t1", export=False)
    run_dir = Path(res["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidates.csv").exists()
    m = json.loads((run_dir / "manifest.json").read_text())
    assert "rounds_log" in m and "roles" in m       # team manifest 角色决策可审计
    assert res["n_trials"] == m["n_trials"]
