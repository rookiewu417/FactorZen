"""20-day return skewness factor."""

import polars as pl
from daily.factors.base import LFTFactor
from daily.data.context import FactorDataContext


class Skewness20D(LFTFactor):
    name = "skewness_20d"
    category = "daily"
    description = "20-day return skewness; right-skewed (positive) stocks earn lower future returns"
    lookback_days = 25

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0)
                .alias("_ret")
            )
            .with_columns(
                pl.col("_ret").rolling_skew(20, bias=True).over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


Skewness20D()
