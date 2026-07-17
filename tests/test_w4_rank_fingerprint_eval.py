# tests/test_w4_rank_fingerprint_eval.py
"""W4：rank fingerprint 移植到 evaluate_expressions + agent 路径语义去重。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from factorzen.discovery.evaluation import evaluate_expressions
from factorzen.discovery.scoring import DataBundle


def _mock_daily(n_stocks=40, n_days=80, seed=7):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            amt = float(abs(rng.standard_normal()) * 1e7 + 1e6) + i * 1e3
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": amt,
            })
    return pl.DataFrame(rows)


def test_rank_fingerprint_import_path_m1_reexport():
    """M1 旧路径 re-export；与 evaluation 公共实现同一对象语义。"""
    from factorzen.discovery.evaluation import _rank_fingerprint as ev_fp
    from factorzen.discovery.mining_session import _rank_fingerprint as m1_fp

    # 同一函数（import re-export）
    assert m1_fp is ev_fp


def test_evaluate_fingerprint_dup_monotone_equivalent():
    """同截面秩序：rank(amount) 与 rank(mul(amount,2)) 第二记 duplicate_fingerprint、不计 N。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    seen: set[str] = set()
    out = evaluate_expressions(
        ["rank(amount)", "rank(mul(amount, 2.0))"],
        daily, bundle, seen_fingerprints=seen,
    )
    assert len(out) == 2
    assert out[0]["error"] is None and out[0]["ic_train"] is not None
    assert out[0]["n_train"] > 0
    assert out[1]["error"] == "duplicate_fingerprint"
    assert out[1]["ic_train"] is None
    assert out[1]["n_train"] == 0
    assert out[1]["compile_ok"] is True
    assert len(seen) == 1  # 只登记首个指纹


def test_evaluate_fingerprint_none_gating_zero_regression():
    """seen_fingerprints=None（默认）→ 不算指纹，两等价表达式都出 IC。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(
        ["rank(amount)", "rank(mul(amount, 2.0))"],
        daily, bundle,
    )
    assert all(r["error"] != "duplicate_fingerprint" for r in out)
    assert out[0]["ic_train"] is not None
    assert out[1]["ic_train"] is not None


def test_evaluate_fingerprint_persists_across_batches():
    """调用方持有跨批 set：第二批同源表达式被去重。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    seen: set[str] = set()
    out1 = evaluate_expressions(["rank(amount)"], daily, bundle, seen_fingerprints=seen)
    assert out1[0]["error"] is None
    out2 = evaluate_expressions(
        ["rank(mul(amount, 2.0))"], daily, bundle, seen_fingerprints=seen,
    )
    assert out2[0]["error"] == "duplicate_fingerprint"
    assert out2[0]["n_train"] == 0


def test_m1_rank_fingerprint_behaviour_unchanged():
    """定向：mining_session 路径 import 的指纹对单调变换仍合并（零回归）。"""
    from factorzen.discovery.mining_session import _rank_fingerprint

    def _mk(vals, n_days=5):
        rows = []
        for d in range(n_days):
            day = dt.date(2024, 1, 2) + dt.timedelta(days=d)
            for i, v in enumerate(vals):
                rows.append({
                    "trade_date": day, "ts_code": f"{i:06d}.SH",
                    "factor_value": float(v),
                })
        return pl.DataFrame(rows)

    base = [((i * 37) % 40) + 0.5 for i in range(40)]
    f_inc = _mk(base)
    f_inc2 = _mk([x * 3.0 + 7.0 for x in base])
    f_dec = _mk([-x for x in base])
    assert _rank_fingerprint(f_inc) == _rank_fingerprint(f_inc2)
    assert _rank_fingerprint(f_inc) != _rank_fingerprint(f_dec)
