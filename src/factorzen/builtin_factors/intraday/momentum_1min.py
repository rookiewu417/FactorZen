"""intraday Demo: 1分钟 5-bar 动量因子。

Momentum1Min = close(t) / close(t-5) - 1
基于 ctx.minute 提供的 1 分钟 K 线数据计算。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from factorzen.intraday.factors.base import IntradayFactor

if TYPE_CHECKING:
    from factorzen.intraday.data.context import IntradayDataContext


@dataclass
class Momentum1Min(IntradayFactor):
    """1分钟 5-bar 动量因子。

    计算: factor_value = close(t) / close(t-5) - 1
    即当前收盘价相对于 5 根 bar 前收盘价的涨跌幅。

    Attributes:
        name: 因子名称。
        bar_size: K 线精度（1min）。
        lookback_bars: 计算所需的最小 bar 数（5 + 1 = 6）。
        description: 因子描述文本。
    """

    name: str = "momentum_1min"
    bar_size: str = "1min"
    lookback_bars: int = 6
    description: str = "5-bar momentum: close(t) / close(t-5) - 1"

    def compute(self, ctx: "IntradayDataContext") -> pl.DataFrame:
        """计算 5-bar 动量因子。

        公式: factor_value = close(t) / close(t-5) - 1

        在 ts_code 分组内进行 shift，ctx.minute 已按 ts_code + trade_time 排序。

        Args:
            ctx: intraday 数据上下文，提供 ctx.minute LazyFrame。

        Returns:
            pl.DataFrame，包含列: trade_time, ts_code, factor_value。
            前 5 根 bar 的 factor_value 为 null，已过滤。
        """
        lf = ctx.minute
        # 按 (ts_code, 交易日) 分区做 shift：否则 shift(5).over("ts_code") 会让次日开盘首根
        # bar 用前一日尾盘价算动量，把隔夜跳空（及跨日缺口）当成日内动量污染因子。
        # 日期从 trade_time 前 10 位取，兼容 Datetime 与字符串两种 dtype。
        result = (
            lf.with_columns(
                pl.col("trade_time").cast(pl.Utf8).str.slice(0, 10).alias("_mom_date")
            )
            .with_columns(
                (
                    pl.col("close")
                    / pl.col("close").shift(5).over(["ts_code", "_mom_date"])
                    - 1.0
                ).alias("factor_value")
            )
            .select(["trade_time", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null())
            .collect()
        )
        return result
