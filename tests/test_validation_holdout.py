# tests/test_validation_holdout.py
from datetime import date, timedelta

import numpy as np
import polars as pl


def _daily(n_stocks=20, n_days=200, seed=1):
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


def test_split_holdout_disjoint_and_isolated():
    from factorzen.validation.holdout import split_holdout
    daily = _daily()
    mining, holdout, hstart = split_holdout(daily, holdout_ratio=0.2)
    # 隔离：mining 全部 < holdout_start ≤ holdout 全部
    assert mining["trade_date"].max() < hstart
    assert holdout["trade_date"].min() >= hstart
    # holdout 约占 20%
    frac = holdout["trade_date"].n_unique() / daily["trade_date"].n_unique()
    assert 0.15 < frac < 0.25


def test_holdout_ic_runs():
    from factorzen.validation.holdout import holdout_ic, split_holdout
    daily = _daily()
    _mining, holdout, _ = split_holdout(daily, holdout_ratio=0.2)
    # 用「次日收益」当因子 → holdout IC 应为正
    fac = holdout.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"]).drop_nulls()
    ic_mean, _ir, (lo, hi) = holdout_ic(fac, holdout)
    assert ic_mean > 0.05 and lo <= hi
