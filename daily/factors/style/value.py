"""价值风格因子：log(B/P) = -log(PB)。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class ValueStyle(DailyFactor):
    name = "value_style"
    category = "daily"
    description = "价值因子：-log(PB)，即 log(B/P)，Barra VALUE"
    lookback_days = 5
    required_data = ["daily_basic"]

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        result = (
            ctx.daily_basic.filter(pl.col("pb") > 0)
            .with_columns((-pl.col("pb").log()).alias("factor_value"))
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


ValueStyle()
