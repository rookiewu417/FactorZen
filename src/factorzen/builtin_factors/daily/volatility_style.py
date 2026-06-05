"""波动率风格因子：21 日对数收益标准差。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class VolatilityStyle(DailyFactor):
    name = "volatility_style"
    category = "daily"
    description = "波动率因子：21 日 std(log_return)，Barra VOLATILITY"
    lookback_days = 30

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        result = (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code"))
                .log()
                .alias("_log_ret")
            )
            .with_columns(
                pl.col("_log_ret")
                .rolling_std(21, min_samples=10)
                .over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


VolatilityStyle()
