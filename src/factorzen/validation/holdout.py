"""OOS holdout 时间切分（软隔离）+ holdout 段 IC 验收。"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class HoldoutICResult:
    """holdout IC 点估计 + 有效 IC 天数（覆盖守卫用）。

    ``n_days`` = 截面样本≥门槛后、有限 IC 值的交易日数。因子全 null / 截面过薄 → 0。
    不把空序列伪装成「IC=0 无预测力」——调用方应用 n_days 判 coverage，而非只看 ic_mean。
    """

    ic_mean: float
    ir: float
    ci: tuple[float, float]
    n_days: int


def holdout_ic_result(factor_df: pl.DataFrame, holdout_df: pl.DataFrame) -> HoldoutICResult:
    """top-K 候选在 holdout 段算 IC，并返回有效 IC 天数。

    空因子帧 / 无有效截面 → ``n_days=0``，``ic_mean/ir`` 为 nan（避免 0.0 哨兵被同号门误读）。
    """
    if factor_df is None or factor_df.is_empty() or "factor_value" not in factor_df.columns:
        return HoldoutICResult(float("nan"), float("nan"), (float("nan"), float("nan")), 0)
    finite = factor_df.filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    if finite.is_empty():
        return HoldoutICResult(float("nan"), float("nan"), (float("nan"), float("nan")), 0)

    price_col = "close_adj" if "close_adj" in holdout_df.columns else "close"
    fwd = compute_fwd_returns(holdout_df.sort(["ts_code", "trade_date"]), price_col=price_col)
    clean = cross_sectional_zscore(finite, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )
    res = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        fwd,
        factor_col="factor_clean",
        frequency="daily",
    )
    ic_vals = res.ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    n_days = len(ic_vals)
    if n_days == 0:
        return HoldoutICResult(float("nan"), float("nan"), (float("nan"), float("nan")), 0)
    ci = block_bootstrap_ic_ci(ic_vals)
    return HoldoutICResult(float(res.ic_mean), float(res.ir), ci, n_days)


def holdout_ic(factor_df: pl.DataFrame, holdout_df: pl.DataFrame):
    """top-K 候选因子值在 holdout 段算 (ic_mean, ir, bootstrap_ci)。

    向后兼容 3-tuple；需要 ``n_days`` 时请用 `holdout_ic_result`。
    空/稀疏时 ic_mean 可能为 nan（不再静默 0.0）——与 coverage 守卫配套。
    """
    r = holdout_ic_result(factor_df, holdout_df)
    return (r.ic_mean, r.ir, r.ci)
