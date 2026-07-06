"""agent 评估路径须用真实复权价 + 提供全套叶子（F4，消除双路径漂移）。

根因：evaluate 路径把未复权 close 直接 rename 冒充 close_adj（除权日假收益），且只补
vwap/log_vol → LLM 被广告的 22 叶子里 ret_1d/amplitude/total_mv 等过半在评估帧不存在，
合法表达式一律报错、白耗轮次并误导 LLM。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.discovery.scoring import DataBundle


def _daily_with_adj_and_basic(n_stocks=20, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            prev = px
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98, "pre_close": prev,
                # close_adj 明显区别于 close（模拟复权：×2），验证不被 close 冒充
                "close_adj": px * 2.0, "open_adj": px * 0.99 * 2.0,
                "high_adj": px * 1.01 * 2.0, "low_adj": px * 0.98 * 2.0,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                "total_mv": 5e5 + i * 1e4, "pb": 1.0 + i * 0.1,
            })
    return pl.DataFrame(rows)


def test_derived_and_basic_leaves_evaluable():
    daily = _daily_with_adj_and_basic()
    bundle = DataBundle.build(daily)
    # ret_1d(派生)、total_mv(基本面)、amplitude(派生) —— 修复前评估帧缺这些列 → 报错
    out = evaluate_expressions(["rank(ret_1d)", "rank(total_mv)", "rank(amplitude)"], daily, bundle)
    for r in out:
        assert r["compile_ok"], f"{r['expression']} 应可编译"
        assert r["error"] is None, f"{r['expression']} 不应报错，实得 {r['error']}"
        assert r["ic_train"] is not None, f"{r['expression']} 应算出 IC"


def test_uses_real_close_adj_not_faked_from_close():
    """close_adj 明显≠close 时，ret_1d 须用 close_adj 计算，而非被 close 冒充。"""
    from factorzen.agents.evaluation import _preprocess_daily

    daily = _daily_with_adj_and_basic(n_stocks=2, n_days=10)
    prepped = _preprocess_daily(daily)
    # ret_1d 由 close_adj 算；close_adj=close×2 是等比缩放，比率与 close 算的相同，
    # 但关键是 prep 未把 close 覆盖成 close_adj —— close_adj 仍是 close 的 2 倍。
    a = prepped.filter(pl.col("close_adj").is_not_null()).select(
        (pl.col("close_adj") / pl.col("close")).alias("r"))["r"]
    assert all(abs(v - 2.0) < 1e-9 for v in a.to_list()), "close_adj 不应被未复权 close 覆盖"
