"""周频波动率因子。日频公式 + 周频采样。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import LFTFactor


class VolatilityWeekly(LFTFactor):
    name = "volatility_weekly"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 30
    description = "周频 20 日波动率（日频公式 + 周频采样）"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        """使用日频 20 日滚动 std(log_return)，最终仅输出周频快照日期的因子值。"""
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code")).log().alias("log_ret")
            )
            .with_columns(
                pl.col("log_ret")
                .rolling_std(20, min_samples=10)
                .over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        result = result.filter(pl.col("trade_date").is_in(ctx.snapshot_dates))
        return result


# 模块级实例化，供 registry 自动发现
VolatilityWeekly()
