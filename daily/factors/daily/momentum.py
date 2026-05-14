"""20日动量因子。"""

import polars as pl
from daily.factors.base import LFTFactor
from daily.data.context import FactorDataContext


class Momentum20D(LFTFactor):
    name = "momentum_20d"
    category = "daily"
    description = "20 日动量：(close(t) / close(t-20) - 1)"
    lookback_days = 25

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(20).over("ts_code") - 1.0)
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


# 模块级实例化，供 registry 自动发现
Momentum20D()
