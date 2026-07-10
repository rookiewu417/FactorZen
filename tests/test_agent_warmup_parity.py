"""agent 挖掘路径的预热口径：train 与 holdout 必须走同一条裁剪路径。"""
from __future__ import annotations

import datetime as dt

import polars as pl

from factorzen.agents.evaluation import _node_to_factor_df
from factorzen.discovery.expression import parse_expr


def _synthetic_daily(n_days: int = 120, n_codes: int = 40, start=dt.date(2020, 1, 1)) -> pl.DataFrame:
    """确定性合成帧：close 单调可预测，无随机性，便于 ground-truth 断言。"""
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for d_i, d in enumerate(dates):
        for c_i in range(n_codes):
            rows.append({
                "trade_date": d,
                "ts_code": f"{c_i:06d}.SZ",
                "close": 10.0 + d_i * 0.1 + c_i,
                "open": 10.0 + d_i * 0.1 + c_i,
                "high": 11.0 + d_i * 0.1 + c_i,
                "low": 9.0 + d_i * 0.1 + c_i,
                "vol": 1000.0 + c_i,
                "amount": 5000.0 + c_i,
            })
    return pl.DataFrame(rows)


def test_node_to_factor_df_clips_both_bounds():
    daily = _synthetic_daily(n_days=100)
    node = parse_expr("ts_mean(close, 5)")
    lo, hi = dt.date(2020, 2, 1), dt.date(2020, 3, 1)

    out = _node_to_factor_df(node, daily, eval_start=lo, eval_end=hi)

    assert out["trade_date"].min() == lo
    assert out["trade_date"].max() == hi
