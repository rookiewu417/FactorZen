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
            daily.sort(["ts_code", "trade_date"])
            .with_columns(
                [
                    (
                        -(pl.col("close_adj") / pl.col("close_adj").shift(5).over("ts_code") - 1.0)
                    ).alias("factor_value"),
                    # 过去 5 日有效交易天数（vol > 0 表示未停牌）
                    (pl.col("vol") > 0)
                    .cast(pl.Int8)
                    .rolling_sum(5, min_samples=1)
                    .over("ts_code")
                    .alias("_active_days"),
                ]
            )
            # 过去 5 日至少 3 天正常交易，否则认为存在停牌污染，置 null
            .with_columns(
                pl.when(pl.col("_active_days") >= 3)
                .then(pl.col("factor_value"))
                .otherwise(None)
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


# 模块级实例化，供 registry 自动发现
Reversal5D()
