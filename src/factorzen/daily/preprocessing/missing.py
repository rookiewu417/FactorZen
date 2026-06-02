"""缺失值处理。按截面中位数填充。"""

import polars as pl


def fill_cross_sectional_median(
    df: pl.DataFrame,
    col: str = "factor_value_clip",
) -> pl.DataFrame:
    out_col = f"{col}_fill"
    return df.with_columns(
        pl.col(col).fill_null(pl.col(col).median().over("trade_date")).alias(out_col)
    )
