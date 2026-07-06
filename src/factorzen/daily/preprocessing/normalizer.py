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
    # NaN → null，使 mean/std 聚合跳过它而非被传染成 NaN（否则截面内任一 NaN 会让
    # 整个交易日 z 值全变 NaN、整日被静默丢弃）。
    clean = pl.col(col).fill_nan(None)
    mean = clean.mean().over("trade_date")
    std = clean.std().over("trade_date")
    safe_z = (
        pl.when(clean.is_null()).then(pl.col(col))   # 本行是 NaN/缺失 → 保留原值，不置 0
        .when(std > 0).then((clean - mean) / std)
        .otherwise(0.0)                              # 有效行但截面无离散度(n=1 或常数) → 0
    )
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
    # NaN → null，否则 polars rank 把 NaN 排为最大 → NaN 股票获最高分位，且 count 把
    # NaN 计入分母。fill_nan(None) 后 NaN 得 null 秩、被 count 排除。
    clean = pl.col(factor_col).fill_nan(None)
    result = df.with_columns(
        (
            clean.rank("average").over("trade_date")
            / (clean.count().over("trade_date") + 1)
        ).alias(factor_col)
    )
    if method == "normal":
        result = result.with_columns(
            pl.when(pl.col(factor_col).is_not_null())
            .then(
                pl.col(factor_col).map_batches(
                    lambda s: pl.Series(
                        _norm.ppf(s.to_numpy().clip(1e-7, 1 - 1e-7)),
                        dtype=pl.Float64,
                    )
                )
            )
            .otherwise(None)
            .alias(factor_col)
        )
    return result


# ---------------------------------------------------------------------------
# Quantile transform (sklearn-compatible, per-date)
# ---------------------------------------------------------------------------


def quantile_transform(
    df: pl.DataFrame,
    factor_col: str,
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
    output : {"normal", "uniform"}, default "normal"
        输出分布类型。

    Returns
    -------
    pl.DataFrame
        原地替换 factor_col 列的分位数变换结果，schema 与输入相同。
    """
    from scipy.stats import rankdata

    rows: list[pl.DataFrame] = []
    for _date_key, group in df.group_by("trade_date", maintain_order=True):
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
