"""遗传搜索并行评分:同 seed 下 workers=1 与 workers=N 结果必须逐项等价。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from factorzen.discovery.mining_session import run_session
from factorzen.discovery.operators import BASIC_FEATURES


def _synthetic_daily(n_stocks=40, n_days=160, seed=3):
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    prices = {s: 10.0 + s for s in range(n_stocks)}
    rows = []
    for d in range(n_days):
        dt = start + timedelta(days=d)
        for s in range(n_stocks):
            prev = prices[s]
            price = max(1.0, prev * (1 + rng.normal(0, 0.02)))
            prices[s] = price
            vol = float(rng.uniform(1e5, 1e6))
            rows.append(
                {
                    "ts_code": f"{s:04d}.SZ",
                    "trade_date": dt,
                    "pre_close": prev,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "open_adj": price,
                    "high_adj": price * 1.01,
                    "low_adj": price * 0.99,
                    "close_adj": price,
                    "vol": vol,
                    "amount": price * vol,
                }
            )
    daily = pl.DataFrame(rows)
    return daily.with_columns(
        [pl.lit(1.0).alias(c) for c in sorted(BASIC_FEATURES) if c not in daily.columns]
    )


def test_genetic_parallel_deterministic(tmp_path):
    daily = _synthetic_daily()
    r1 = run_session(
        daily, n_trials=40, top_k=3, seed=7, method="genetic",
        out_dir=str(tmp_path / "w1"), workers=1,
    )
    r4 = run_session(
        daily, n_trials=40, top_k=3, seed=7, method="genetic",
        out_dir=str(tmp_path / "w4"), workers=4,
    )
    e1 = [c["expression"] for c in r1["candidates"]]
    e4 = [c["expression"] for c in r4["candidates"]]
    assert e1 == e4, "并行与串行的 leaderboard 表达式序列必须一致"
    f1 = [round(float(c["ir_train"]), 6) for c in r1["candidates"]]
    f4 = [round(float(c["ir_train"]), 6) for c in r4["candidates"]]
    assert f1 == f4, "并行与串行的候选分数必须一致"
