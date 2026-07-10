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


def test_warmup_bars_counts_nonnull_history_per_leaf():
    """预热段有交易日、但该叶子全 null → 可用预热必须是 0，不能按交易日数报充足。

    复现 daily_basic 缺 2019 的真实情况：帧里有 60 个预热交易日，dv_ttm 全 null。
    """
    from factorzen.agents.evaluation import _preprocess_daily, warmup_bars

    daily = _synthetic_daily(n_days=100)
    cutoff = dt.date(2020, 2, 1)
    # dv_ttm：cutoff 之前全 null，之后有值
    daily = daily.with_columns(
        pl.when(pl.col("trade_date") >= cutoff).then(pl.lit(2.0)).otherwise(None).alias("dv_ttm")
    )
    prepped = _preprocess_daily(daily)

    assert warmup_bars(parse_expr("dv_ttm"), prepped, cutoff) == 0
    # close 在预热段有值 → 预热 bar 数 = cutoff 之前的交易日数
    n_before = daily.filter(pl.col("trade_date") < cutoff)["trade_date"].n_unique()
    assert warmup_bars(parse_expr("close"), prepped, cutoff) == n_before
    # 混合表达式取各叶子最小值 → 被 dv_ttm 拉到 0
    assert warmup_bars(parse_expr("add(close, dv_ttm)"), prepped, cutoff) == 0
