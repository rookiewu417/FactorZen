"""截面分位分组：单一实现，供 signal_backtest / turnover / monotonicity 共用。"""

from __future__ import annotations

import polars as pl


def assign_quantile_groups(
    df: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    *,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """逐日截面按因子值分位分组，返回加 ``group`` 列（Int32，0..n_groups-1，0=因子最小）。

    先过滤 ``factor_col`` 的 null 与 NaN（polars 中 NaN 非 null，``rank`` 会把 NaN
    排最大污染最高组）。

    分组公式（与历史 turnover / monotonicity 逐位一致）::

        rank = col.rank("ordinal", descending=False).over(date_col)
        group = (rank - 1) * n_groups // rank.max().over(date_col)  → Int32

    ordinal rank 天然打散并列。
    """
    out = df.filter(pl.col(factor_col).is_not_null() & pl.col(factor_col).is_not_nan())
    if out.is_empty():
        if "group" not in out.columns:
            return out.with_columns(pl.lit(None, dtype=pl.Int32).alias("group"))
        return out
    return (
        out.with_columns(
            pl.col(factor_col)
            .rank("ordinal", descending=False)
            .over(date_col)
            .alias("_rank")
        )
        .with_columns(
            ((pl.col("_rank") - 1) * n_groups // pl.col("_rank").max().over(date_col))
            .cast(pl.Int32)
            .alias("group")
        )
        .drop("_rank")
    )
