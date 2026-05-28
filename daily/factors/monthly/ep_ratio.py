"""Monthly earnings-to-price ratio (E/P) factor."""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class EpRatioMonthly(DailyFactor):
    name = "ep_ratio"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "Monthly E/P = 1/PE_TTM; earnings value factor, high E/P predicts higher returns"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        monthly_basic = ctx.monthly_basic
        result = (
            monthly_basic.filter(pl.col("pe_ttm").is_not_null() & (pl.col("pe_ttm") > 0))
            .select(
                [
                    pl.col("trade_date"),
                    pl.col("ts_code"),
                    (1.0 / (pl.col("pe_ttm") + 1e-8)).alias("factor_value"),
                ]
            )
            .collect()
        )
        return result


EpRatioMonthly()
