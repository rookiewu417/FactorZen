"""OOS holdout 时间切分（软隔离）+ holdout 段 IC 验收。"""
from __future__ import annotations

import polars as pl

from factorzen.daily.evaluation.ic_analysis import (
    _ic_stats,
    _rank_ic_by_date,
    compute_fwd_returns,
)
from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
from factorzen.validation.bootstrap import block_bootstrap_ic_ci

# holdout IC 验收时每日截面最小样本数；生产宇宙远大于此值
_HOLDOUT_MIN_CROSS_SAMPLES = 10


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
    # 计算 holdout 段前向收益
    daily_sorted = holdout_df.sort(["ts_code", "trade_date"])
    price_col = "close_adj" if "close_adj" in holdout_df.columns else "close"
    fwd = compute_fwd_returns(daily_sorted, price_col=price_col)

    # 截面 Z-score 标准化
    clean = cross_sectional_zscore(factor_df, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )

    # 合并因子与前向收益，计算每日截面 Rank IC
    # 注：使用 _HOLDOUT_MIN_CROSS_SAMPLES 而非全局默认值 30，
    #     以兼容宇宙较小的测试及早期挖矿场景
    merged = clean.select(["trade_date", "ts_code", "factor_clean"]).join(
        fwd.select(["trade_date", "ts_code", "fwd_ret_1d"]),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    ic_df = _rank_ic_by_date(
        merged, "factor_clean", "fwd_ret_1d", min_samples=_HOLDOUT_MIN_CROSS_SAMPLES
    )
    ic_vals = ic_df["ic"].drop_nulls().drop_nans().to_numpy()

    ic_mean, _ic_std, ir, *_ = _ic_stats(ic_vals)
    ci = block_bootstrap_ic_ci(ic_vals)
    return (ic_mean, ir, ci)
