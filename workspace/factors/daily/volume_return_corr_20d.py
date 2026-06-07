"""20-day volume-return correlation factor."""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class VolumeReturnCorr20D(DailyFactor):
    name = "volume_return_corr_20d"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 30
    description = "20-day rolling Pearson correlation between 1-day return and log volume"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        return (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                [
                    (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0)
                    .alias("_ret"),
                    pl.col("vol").log1p().alias("_log_vol"),
                ]
            )
            .with_columns((pl.col("_ret") * pl.col("_log_vol")).alias("_ret_x_log_vol"))
            .with_columns(
                [
                    pl.col("_ret").rolling_mean(20, min_samples=10).over("ts_code").alias("_ret_mean"),
                    pl.col("_log_vol")
                    .rolling_mean(20, min_samples=10)
                    .over("ts_code")
                    .alias("_log_vol_mean"),
                    pl.col("_ret_x_log_vol")
                    .rolling_mean(20, min_samples=10)
                    .over("ts_code")
                    .alias("_ret_x_log_vol_mean"),
                    (pl.col("_ret") ** 2)
                    .rolling_mean(20, min_samples=10)
                    .over("ts_code")
                    .alias("_ret_sq_mean"),
                    (pl.col("_log_vol") ** 2)
                    .rolling_mean(20, min_samples=10)
                    .over("ts_code")
                    .alias("_log_vol_sq_mean"),
                ]
            )
            .with_columns(
                [
                    (pl.col("_ret_x_log_vol_mean") - pl.col("_ret_mean") * pl.col("_log_vol_mean"))
                    .alias("_cov"),
                    (pl.col("_ret_sq_mean") - pl.col("_ret_mean") ** 2).alias("_ret_var"),
                    (pl.col("_log_vol_sq_mean") - pl.col("_log_vol_mean") ** 2).alias("_log_vol_var"),
                ]
            )
            .with_columns(
                pl.when((pl.col("_ret_var") > 0) & (pl.col("_log_vol_var") > 0))
                .then(pl.col("_cov") / (pl.col("_ret_var") * pl.col("_log_vol_var")).sqrt())
                .otherwise(None)
                .clip(-1.0, 1.0)
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            .collect()
        )


VolumeReturnCorr20D()
