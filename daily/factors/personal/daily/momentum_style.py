"""动量风格因子：Jegadeesh-Titman 12-1 动量。"""

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class MomentumStyle(DailyFactor):
    name = "momentum_style"
    category = "daily"
    description = "JT 12-1 动量：close_adj[t-21] / close_adj[t-252] - 1，剔除最近1个月反转效应"
    lookback_days = 265

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        result = (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                [
                    pl.col("close_adj").shift(21).over("ts_code").alias("_close_1m_ago"),
                    pl.col("close_adj").shift(252).over("ts_code").alias("_close_12m_ago"),
                ]
            )
            .with_columns(
                (pl.col("_close_1m_ago") / pl.col("_close_12m_ago") - 1.0).alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


MomentumStyle()
