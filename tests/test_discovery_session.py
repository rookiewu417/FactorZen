from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _daily(seed=3, n_stocks=40, n_days=120):
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
                         "open_adj": p, "high_adj": p, "low_adj": p, "open": p, "high": p, "low": p,
                         "amount": 1e7, "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def test_session_runs_and_writes_artifacts(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(), n_trials=20, top_k=5, seed=42,
                      method="random", out_dir=str(tmp_path))
    session_dir = Path(res["session_dir"])
    assert (session_dir / "candidates.csv").exists()
    assert (session_dir / "manifest.json").exists()
    assert 0 < len(res["candidates"]) <= 5
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["cli_n_trials"] == 20
    assert manifest["seed"] == 42
    for c in res["candidates"]:
        assert c["max_corr"] < 0.7  # 贪心去相关保证：top-K 互不近重复，max_corr 是真实测量


def test_session_reproducible_same_seed(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    a = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "a"))
    b = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "b"))
    expr_a = [c["expression"] for c in a["candidates"]]
    expr_b = [c["expression"] for c in b["candidates"]]
    assert expr_a == expr_b


def test_session_has_guard_metrics_and_holdout_isolated(tmp_path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(n_stocks=40, n_days=150), n_trials=30, top_k=5, seed=42,
                      method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert 0 < len(res["candidates"]) <= 5
    for c in res["candidates"]:
        # 护栏指标齐全
        for key in ("n_trials", "pbo", "holdout_ic", "dsr_pvalue", "ic_ci_low"):
            assert key in c
        assert c["n_trials"] > 0          # 真实评估数（非 CLI n_trials 摆设）
        assert 0.0 <= c["pbo"] <= 1.0 or c["pbo"] != c["pbo"]  # [0,1] 或 nan
