# tests/test_discovery_scoring.py
from __future__ import annotations
import numpy as np
import polars as pl
from datetime import date, timedelta


def _daily(seed=1, n_stocks=40, n_days=120):
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
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def _signal_factor_df(daily: pl.DataFrame) -> pl.DataFrame:
    """构造与次日收益正相关的因子（用于验证 IC 为正）。"""
    df = daily.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd"))
    return df.select(["trade_date", "ts_code", pl.col("fwd").alias("factor_value")]).drop_nulls()


def test_databundle_split():
    from factorzen.discovery.scoring import DataBundle
    b = DataBundle.build(_daily(), train_ratio=0.7)
    assert b.train_end is not None
    assert "fwd_ret_1d" in b.fwd_returns.columns


def test_quick_fitness_positive_for_signal():
    from factorzen.discovery.scoring import DataBundle, quick_fitness
    daily = _daily()
    b = DataBundle.build(daily, train_ratio=0.7)
    fac = _signal_factor_df(daily)
    res = quick_fitness(fac, b, segment="train")
    assert res["ic_mean"] > 0.05
    assert res["n"] > 0


def test_max_correlation_self_is_one():
    from factorzen.discovery.scoring import max_correlation
    daily = _daily()
    fac = _signal_factor_df(daily).rename({"factor_value": "factor_clean"})
    corr = max_correlation(fac.rename({"factor_clean": "factor_value"}),
                           {"self": fac})
    assert corr > 0.99


def test_score_penalizes_complexity():
    from factorzen.discovery.scoring import DataBundle, score_candidate
    from factorzen.discovery.expression import parse_expr
    daily = _daily()
    b = DataBundle.build(daily)
    fac = _signal_factor_df(daily)
    simple = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.01)
    # 复杂表达式（节点更多）在相同 IC 下 fitness 更低
    assert simple["complexity"] == 1
    # 相同因子值(IC 相同) + 更复杂的 node → complexity 更大 → fitness 更低（纯复杂度惩罚）
    complex_score = score_candidate(fac, parse_expr("ts_mean(close, 5)"), b, pool={}, gamma=0.01)
    assert complex_score["complexity"] > simple["complexity"]
    assert complex_score["fitness"] < simple["fitness"]
