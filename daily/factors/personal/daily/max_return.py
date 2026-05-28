"""5-day max return factor (lottery effect anomaly)."""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class MaxReturn5D(DailyFactor):
    name = "max_return_5d"
    category = "daily"
    description = "5-day MAX factor (Bali et al. 2011): negatively predicts future returns"
    lookback_days = 10

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias(
                    "_ret"
                )
            )
            .with_columns(
                pl.col("_ret").rolling_max(5, min_samples=3).over("ts_code").alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


MaxReturn5D()
