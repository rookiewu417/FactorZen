"""Monthly book-to-market ratio (B/M) factor."""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class BmRatioMonthly(DailyFactor):
    name = "bm_ratio"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "Monthly B/M = 1/PB; value factor, high B/M predicts higher returns"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        monthly_basic = ctx.monthly_basic
        result = (
            monthly_basic.filter(pl.col("pb").is_not_null() & (pl.col("pb") > 0))
            .select(
                [
                    pl.col("trade_date"),
                    pl.col("ts_code"),
                    (1.0 / (pl.col("pb") + 1e-8)).alias("factor_value"),
                ]
            )
            .collect()
        )
        return result


BmRatioMonthly()
