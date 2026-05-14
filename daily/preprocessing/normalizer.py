"""截面 Z-score 标准化。"""

import polars as pl


def cross_sectional_zscore(
    df: pl.DataFrame,
    col: str = "factor_value_clip_fill",
) -> pl.DataFrame:
    out_col = f"{col}_z"
    mean = pl.col(col).mean().over("trade_date")
    std = pl.col(col).std().over("trade_date")
    safe_z = pl.when(std > 0).then((pl.col(col) - mean) / std).otherwise(0.0)
    return df.with_columns(safe_z.alias(out_col))
