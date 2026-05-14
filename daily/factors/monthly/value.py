"""月频估值因子：pe_ttm 和 pb。直接从 daily_basic 月末快照提取。"""

import polars as pl
from daily.factors.base import LFTFactor
from daily.data.context import FactorDataContext


class PeTtmMonthly(LFTFactor):
    name = "pe_ttm"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "月频滚动市盈率（PE-TTM），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        monthly_basic = ctx.monthly_basic
        result = (
            monthly_basic
            .select([
                pl.col("trade_date"),
                pl.col("ts_code"),
                pl.col("pe_ttm").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
            .collect()
        )
        return result


class PbMonthly(LFTFactor):
    name = "pb"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "月频市净率（PB），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        monthly_basic = ctx.monthly_basic
        result = (
            monthly_basic
            .select([
                pl.col("trade_date"),
                pl.col("ts_code"),
                pl.col("pb").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
            .collect()
        )
        return result


PeTtmMonthly()
PbMonthly()
