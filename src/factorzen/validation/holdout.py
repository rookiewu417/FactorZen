"""OOS holdout 时间切分（软隔离）+ holdout 段 IC 验收。"""
from __future__ import annotations

import polars as pl

from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
from factorzen.validation.bootstrap import block_bootstrap_ic_ci


def split_holdout(daily: pl.DataFrame, holdout_ratio: float = 0.2):
    """按交易日时间序，最后 holdout_ratio 比例为 holdout；其余为 mining 段。"""
    dates = sorted(daily["trade_date"].unique().to_list())
    cut = int(len(dates) * (1.0 - holdout_ratio))
    cut = min(max(cut, 1), len(dates) - 1)
    holdout_start = dates[cut]
    mining_df = daily.filter(pl.col("trade_date") < holdout_start)
    holdout_df = daily.filter(pl.col("trade_date") >= holdout_start)
    return mining_df, holdout_df, holdout_start


def holdout_ic(factor_df: pl.DataFrame, holdout_df: pl.DataFrame):
    """top-K 候选因子值在 holdout 段算 (ic_mean, ir, bootstrap_ci)。

    Args:
        factor_df: [trade_date, ts_code, factor_value]，仅含 holdout 段数据。
        holdout_df: 原始日频价格数据（holdout 段），用于计算前向收益。

    Returns:
        (ic_mean, ir, (ci_lo, ci_hi))
    """
    price_col = "close_adj" if "close_adj" in holdout_df.columns else "close"
    fwd = compute_fwd_returns(holdout_df.sort(["ts_code", "trade_date"]), price_col=price_col)
    clean = cross_sectional_zscore(factor_df, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )
    res = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        fwd,
        factor_col="factor_clean",
        frequency="daily",
    )
    ic_vals = res.ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    ci = block_bootstrap_ic_ci(ic_vals)
    return (res.ic_mean, res.ir, ci)
