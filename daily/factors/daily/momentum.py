"""动量因子。"""

import warnings

import polars as pl

from daily.data.context import FactorDataContext
from daily.factors.base import DailyFactor


class Momentum20D(DailyFactor):
    name = "momentum_20d"
    category = "daily"
    description = "20 日动量：(close(t) / close(t-20) - 1)（已弃用，请用 Momentum12_1）"
    lookback_days = 25

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        warnings.warn(
            "Momentum20D 混杂短期反转效应，建议改用 Momentum12_1（JT 12-1）",
            DeprecationWarning,
            stacklevel=2,
        )
        daily = ctx.daily
        result = (
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(20).over("ts_code") - 1.0).alias(
                    "factor_value"
                )
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


class Momentum12_1(DailyFactor):
    """Jegadeesh-Titman 12-1 动量因子。

    用 t-21 至 t-252 的收益率（剔除最近 1 个月，避免短期反转污染）。
    """

    name = "momentum_12_1"
    category = "daily"
    description = "JT 12-1 动量：close_adj[t-21] / close_adj[t-252] - 1"
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
            .filter(pl.col("_close_12m_ago") > 0)
            .with_columns(
                (pl.col("_close_1m_ago") / pl.col("_close_12m_ago") - 1.0).alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


# 模块级实例化，供 registry 自动发现
Momentum20D()
Momentum12_1()
