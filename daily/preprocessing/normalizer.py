"""截面 Z-score、rank、quantile 标准化。"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import norm as _norm


def cross_sectional_zscore(
    df: pl.DataFrame,
    col: str = "factor_value_clip_fill",
) -> pl.DataFrame:
    out_col = f"{col}_z"
    mean = pl.col(col).mean().over("trade_date")
    std = pl.col(col).std().over("trade_date")
    safe_z = pl.when(std > 0).then((pl.col(col) - mean) / std).otherwise(0.0)
    return df.with_columns(safe_z.alias(out_col))


# ---------------------------------------------------------------------------
# Cross-sectional rank normalisation
# ---------------------------------------------------------------------------


def cross_sectional_rank(
    df: pl.DataFrame,
    factor_col: str,
    method: Literal["uniform", "normal"] = "uniform",
) -> pl.DataFrame:
    """截面 rank 标准化：uniform → (0, 1)，normal → 标准正态。

    Parameters
    ----------
    df : pl.DataFrame
        必须包含 trade_date 和 factor_col 两列。
    factor_col : str
        待处理的因子列名（原地替换）。
    method : {"uniform", "normal"}, default "uniform"
        "uniform" 返回 rank / (n + 1) ∈ (0, 1)；
        "normal" 进一步做 Φ⁻¹ 变换使其近似标准正态。

    Returns
    -------
    pl.DataFrame
        原地替换 factor_col 列的 rank 标准化结果。
    """
    # rank / (n + 1): average ties, per trade_date cross-section
    result = df.with_columns(
        (
            pl.col(factor_col).rank("average").over("trade_date")
            / (pl.col(factor_col).count().over("trade_date") + 1)
        ).alias(factor_col)
    )
    if method == "normal":
        result = result.with_columns(
            pl.col(factor_col)
            .map_batches(
                lambda s: pl.Series(
                    _norm.ppf(s.fill_null(0.5).to_numpy().clip(1e-7, 1 - 1e-7))
                )
            )
            .alias(factor_col)
        )
    return result


# ---------------------------------------------------------------------------
# Quantile transform (sklearn-compatible, per-date)
# ---------------------------------------------------------------------------


def quantile_transform(
    df: pl.DataFrame,
    factor_col: str,
    n_quantiles: int = 1000,
    output: Literal["normal", "uniform"] = "normal",
) -> pl.DataFrame:
    """仿 sklearn QuantileTransformer，按日期分组变换。

    对每个交易日截面独立做分位数变换：
      1. 对有效值排序并计算分位数位置 (rank - 0.5) / n ∈ (0, 1)。
      2. 若 output="normal" 则再做 Φ⁻¹ 变换；否则保留 (0, 1) 均匀分布。

    Parameters
    ----------
    df : pl.DataFrame
        必须包含 trade_date 和 factor_col 两列。
    factor_col : str
        待处理的因子列名（原地替换）。
    n_quantiles : int, default 1000
        保留参数，与 sklearn 接口对齐（当前实现不分桶，直接用连续 rank）。
    output : {"normal", "uniform"}, default "normal"
        输出分布类型。

    Returns
    -------
    pl.DataFrame
        原地替换 factor_col 列的分位数变换结果，schema 与输入相同。
    """
    from scipy.stats import rankdata

    rows: list[pl.DataFrame] = []
    for _date_key, group in df.group_by("trade_date"):
        vals = group[factor_col].to_numpy().copy().astype(float)
        valid_mask = np.isfinite(vals)

        if valid_mask.sum() == 0:
            # All NaN — return as-is
            rows.append(group)
            continue

        valid_vals = vals[valid_mask]
        n = len(valid_vals)

        if n == 1 or np.all(valid_vals == valid_vals[0]):
            # Constant column — use midpoint of target distribution
            if output == "normal":
                transformed_const = np.zeros(n, dtype=float)
            else:
                transformed_const = np.full(n, 0.5, dtype=float)
            result_vals = vals.copy()
            result_vals[valid_mask] = transformed_const
            rows.append(group.with_columns(pl.Series(factor_col, result_vals)))
            continue

        ranks = rankdata(valid_vals, method="average")
        quantiles = (ranks - 0.5) / n  # centre of each quantile bin ∈ (0, 1)
        quantiles = np.clip(quantiles, 1e-7, 1 - 1e-7)

        if output == "normal":
            transformed = _norm.ppf(quantiles)
        else:
            transformed = quantiles

        result_vals = vals.copy()
        result_vals[valid_mask] = transformed
        rows.append(group.with_columns(pl.Series(factor_col, result_vals)))

    if not rows:
        return df.clone()

    return pl.concat(rows)
