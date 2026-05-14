"""5日反转因子。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import LFTFactor


class Reversal5D(LFTFactor):
    name = "reversal_5d"
    category = "daily"
    description = "5日反转因子：-(close(t) / close(t-5) - 1)"
    lookback_days = 10

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (-(pl.col("close") / pl.col("close").shift(5).over("ts_code") - 1.0))
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


# 模块级实例化，供 registry 自动发现
Reversal5D()
