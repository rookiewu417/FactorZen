"""周频动量因子。复用日频 20 日动量公式，下采样到周频快照。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class MomentumWeekly(DailyFactor):
    name = "momentum_weekly"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 30
    description = "周频 20 日动量（日频公式 + 下采样到周频快照日期）"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(20).over("ts_code") - 1.0).alias(
                    "factor_value"
                )
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        # 下采样到周频快照日期
        snapshots = ctx.snapshot_dates
        result = result.filter(pl.col("trade_date").is_in(snapshots))
        return result


# 模块级实例化，供 registry 自动发现
MomentumWeekly()
