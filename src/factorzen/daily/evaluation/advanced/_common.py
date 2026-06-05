"""高级评估共享工具：分组 IC 计算。"""

from __future__ import annotations

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def _grouped_ic(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    group_col: str,
    min_per_cell: int = 2,
) -> pl.DataFrame:
    """在分组标签上计算截面 Rank IC，返回 (group_col_renamed_to_group → ic) DataFrame。

    原理：rank within (group, date) → pearson_corr grouped by (group, date) → mean by group。
    """
    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )
    if valid_df.is_empty():
        return pl.DataFrame({"regime": [], "ic": []})

    ranked = valid_df.with_columns(
        [
            pl.col(factor_col)
            .rank(method="average")
            .over([group_col, "trade_date"])
            .alias("_factor_rank"),
            pl.col(ret_col)
            .rank(method="average")
            .over([group_col, "trade_date"])
            .alias("_ret_rank"),
        ]
    )
    out_col = "regime" if group_col != "regime" else group_col
    return (
        ranked.group_by([group_col, "trade_date"])
        .agg(
            [
                pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                pl.len().alias("_n"),
            ]
        )
        .filter(pl.col("_n") >= min_per_cell)
        .drop("_n")
        .group_by(group_col)
        .agg(pl.col("ic").mean())
        .rename({group_col: out_col} if group_col != out_col else {})
        .sort(out_col)
    )
