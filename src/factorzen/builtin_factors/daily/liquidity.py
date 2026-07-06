"""流动性风格因子：21 日换手率均值。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class LiquidityStyle(DailyFactor):
    name = "liquidity_style"
    category = "daily"
    description = "流动性因子：21 日换手率均值，Barra LIQUIDITY"
    lookback_days = 30
    required_data = ["daily_basic", "daily"]

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        result = (
            ctx.daily_basic.sort(["ts_code", "trade_date"])
            .with_columns(
                pl.col("turnover_rate")
                .rolling_mean(21, min_samples=10)
                .over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


LiquidityStyle()
