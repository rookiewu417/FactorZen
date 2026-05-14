"""20-day idiosyncratic volatility factor (residual after market beta)."""

import polars as pl
from daily.factors.base import LFTFactor
from daily.data.context import FactorDataContext


class IdiosyncraticVol20D(LFTFactor):
    name = "idiosyncratic_vol_20d"
    category = "daily"
    description = "20-day idiosyncratic volatility: std of residuals after removing market beta"
    lookback_days = 25

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily.collect()

        daily = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0)
                .alias("ret")
            )
        )

        market_ret = (
            daily.group_by("trade_date")
            .agg(pl.col("ret").mean().alias("market_ret"))
        )

        df = daily.join(market_ret, on="trade_date", how="inner").sort(["ts_code", "trade_date"])

        # rolling 20-day beta, then residual std
        df = df.with_columns([
            pl.col("ret").rolling_mean(20, min_samples=10).over("ts_code").alias("_ret_mean"),
            pl.col("market_ret").rolling_mean(20, min_samples=10).over("ts_code").alias("_mkt_mean"),
        ]).with_columns([
            ((pl.col("ret") - pl.col("_ret_mean")) * (pl.col("market_ret") - pl.col("_mkt_mean")))
            .rolling_mean(20, min_samples=10).over("ts_code").alias("_cov"),
            ((pl.col("market_ret") - pl.col("_mkt_mean")) ** 2)
            .rolling_mean(20, min_samples=10).over("ts_code").alias("_var_m"),
        ]).with_columns(
            (pl.col("_cov") / (pl.col("_var_m") + 1e-12)).alias("_beta")
        ).with_columns(
            (pl.col("ret") - pl.col("_beta") * pl.col("market_ret")).alias("_resid")
        ).with_columns(
            pl.col("_resid").rolling_std(20, min_samples=10).over("ts_code").alias("factor_value")
        )

        result = (
            df.filter(
                pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d")
            )
            .select(["trade_date", "ts_code", "factor_value"])
        )
        return result


IdiosyncraticVol20D()
