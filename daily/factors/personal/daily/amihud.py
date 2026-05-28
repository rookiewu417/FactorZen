"""Amihud non-liquidity factor."""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class AmihudIlliquidity(DailyFactor):
    name = "amihud_illiquidity"
    category = "daily"
    description = "Amihud (2002) illiquidity: 20-day mean of |ret|/amount, higher = less liquid"
    lookback_days = 25

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                [
                    (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0)
                    .abs()
                    .alias("_abs_ret"),
                    (pl.col("amount") + 1e-6).alias("_amount"),
                ]
            )
            .with_columns(
                (pl.col("_abs_ret") / pl.col("_amount"))
                .rolling_mean(20, min_samples=10)
                .over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


AmihudIlliquidity()
