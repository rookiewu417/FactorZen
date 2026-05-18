"""5日平均换手率因子。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import LFTFactor


class Turnover5D(LFTFactor):
    name = "turnover_5d"
    category = "daily"
    description = "5日平均成交量（换手率 proxy）"
    lookback_days = 10

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
        return result


# 模块级实例化，供 registry 自动发现
Turnover5D()
