"""60-day CAPM Beta factor."""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class Beta60D(DailyFactor):
    name = "beta_60d"
    category = "daily"
    description = "60-day rolling CAPM beta vs equal-weight market portfolio"
    lookback_days = 65

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily.collect()

        daily = daily.sort(["ts_code", "trade_date"]).with_columns(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias("ret")
        )

        market_ret = daily.group_by("trade_date").agg(pl.col("ret").mean().alias("market_ret"))

        df = daily.join(market_ret, on="trade_date", how="inner")

        # rolling OLS: beta = cov(ret_i, ret_m) / var(ret_m)
        df = (
            df.sort(["ts_code", "trade_date"])
            .with_columns(
                [
                    pl.col("ret")
                    .rolling_mean(60, min_samples=30)
                    .over("ts_code")
                    .alias("_ret_mean"),
                    pl.col("market_ret")
                    .rolling_mean(60, min_samples=30)
                    .over("ts_code")
                    .alias("_mkt_mean"),
                ]
            )
            .with_columns(
                [
                    (
                        (pl.col("ret") - pl.col("_ret_mean"))
                        * (pl.col("market_ret") - pl.col("_mkt_mean"))
                    )
                    .rolling_mean(60, min_samples=30)
                    .over("ts_code")
                    .alias("_cov"),
                    ((pl.col("market_ret") - pl.col("_mkt_mean")) ** 2)
                    .rolling_mean(60, min_samples=30)
                    .over("ts_code")
                    .alias("_var_m"),
                ]
            )
            .with_columns((pl.col("_cov") / (pl.col("_var_m") + 1e-12)).alias("factor_value"))
        )

        result = df.filter(
            pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d")
        ).select(["trade_date", "ts_code", "factor_value"])
        return result


Beta60D()
