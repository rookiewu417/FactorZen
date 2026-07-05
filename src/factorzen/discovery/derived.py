"""factor.py 与 mining_session.py 共用的派生列逻辑，杜绝双路径漂移。

约定：输入 df 已按 (ts_code, trade_date) 排序且已做停牌掩码
（vol==0 行价量列置 null）。pre_close 不参与掩码，作分母时用 when 保护。
"""
from __future__ import annotations

import polars as pl


def add_derived_columns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            (pl.col("amount") / pl.col("vol")).alias("vwap"),
            (pl.col("vol") + 1.0).log().alias("log_vol"),
        ])
        .with_columns(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias("ret_1d")
        )
        .with_columns([
            pl.when(pl.col("pre_close") > 1e-12)
              .then((pl.col("high") - pl.col("low")) / pl.col("pre_close"))
              .otherwise(None).alias("amplitude"),
            pl.when(pl.col("open") > 1e-12)
              .then(pl.col("close") / pl.col("open") - 1.0)
              .otherwise(None).alias("intraday_ret"),
            pl.when(pl.col("pre_close") > 1e-12)
              .then(pl.col("open") / pl.col("pre_close") - 1.0)
              .otherwise(None).alias("overnight_ret"),
        ])
    )
