"""周频换手率因子。日频公式 + 周频采样。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class TurnoverWeekly(DailyFactor):
    name = "turnover_weekly"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 15
    description = "周频 5 日平均成交量（换手率 proxy）"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                pl.col("vol")
                .rolling_mean(5, min_samples=3)
                .over("ts_code")
                .log1p()
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        result = result.filter(pl.col("trade_date").is_in(ctx.snapshot_dates))
        return result


TurnoverWeekly()
