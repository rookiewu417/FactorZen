"""20日波动率因子。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import LFTFactor


class Volatility20D(LFTFactor):
    name = "volatility_20d"
    category = "daily"
    description = "20日波动率：std(log_return) over 20 days"
    lookback_days = 25

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code"))
                .log()
                .alias("log_ret")
            )
            .with_columns(
                pl.col("log_ret")
                .rolling_std(20, min_samples=15)
                .over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


# 模块级实例化，供 registry 自动发现
Volatility20D()
