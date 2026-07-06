"""规模因子：log(总市值)。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class SizeStyle(DailyFactor):
    name = "size_style"
    category = "daily"
    description = "规模因子：log(total_mv)，Barra SIZE"
    lookback_days = 5
    required_data = ["daily_basic", "daily"]

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        result = (
            ctx.daily_basic.filter(pl.col("total_mv") > 0)
            .with_columns(pl.col("total_mv").log().alias("factor_value"))
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


SizeStyle()
