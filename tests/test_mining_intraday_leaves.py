"""挖掘链：i_* 可评估 + 无面板时 leaf_health 摘叶 + 同 seed 自身一致性。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from factorzen.core.feature_schema import INTRADAY_FEATURES


def _mk_daily(
    n_days: int = 60,
    n_stocks: int = 35,
    seed: int = 7,
    *,
    with_intraday: bool = False,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            row = {
                "trade_date": dd,
                "ts_code": c,
                "close": px,
                "open": px,
                "high": px * 1.01,
                "low": px * 0.99,
                "close_adj": px,
                "open_adj": px,
                "high_adj": px * 1.01,
                "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": float(1e6 + rng.normal(0, 1e4)),
                "amount": float(1e7 + rng.normal(0, 1e5)),
            }
            if with_intraday:
                for leaf in sorted(INTRADAY_FEATURES):
                    row[leaf] = float(abs(rng.normal(0.02, 0.01)))
            rows.append(row)
    return pl.DataFrame(rows)


def test_run_session_with_i_rv_evaluable(tmp_path, monkeypatch):
    """合成帧含 i_* 列：含 i_rv 的表达式可评估且非全 null。"""
    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.mining_session import run_session
    from factorzen.discovery.scoring import DataBundle

    daily = _mk_daily(with_intraday=True)
    bundle = DataBundle.build(daily)
    res = evaluate_expressions(["rank(i_rv)", "ts_mean(i_rv, 5)"], daily, bundle)
    assert all(r["compile_ok"] for r in res), res
    assert any(r.get("ic_train") is not None for r in res)

    # 强制搜索产出 i_rv 表达式
    exprs = ["rank(i_rv)", "rank(close)"]
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
    out = run_session(
        daily, n_trials=4, top_k=2, seed=1, method="random",
        out_dir=str(tmp_path / "sess_i"),
        update_library=False,
        library_orthogonal=False,
    )
    # 至少跑完；i_rv 不应因缺列被 compile 拒绝
    assert "candidates" in out


def test_zero_regression_excluded_intraday_and_seed_consistency(tmp_path):
    """不带 i_* 列：excluded 恰含全部 INTRADAY_FEATURES；同 seed 候选序列自身一致。"""
    from factorzen.discovery.mining_session import run_session

    daily = _mk_daily(with_intraday=False)
    r1 = run_session(
        daily, n_trials=8, top_k=3, seed=42, method="random",
        out_dir=str(tmp_path / "a"),
        update_library=False,
        library_orthogonal=False,
    )
    r2 = run_session(
        daily, n_trials=8, top_k=3, seed=42, method="random",
        out_dir=str(tmp_path / "b"),
        update_library=False,
        library_orthogonal=False,
    )
    excl1 = set(r1.get("excluded_leaves") or {})
    excl2 = set(r2.get("excluded_leaves") or {})
    assert excl1 >= INTRADAY_FEATURES
    assert excl1 == excl2
    e1 = [c["expression"] for c in r1["candidates"]]
    e2 = [c["expression"] for c in r2["candidates"]]
    assert e1 == e2
