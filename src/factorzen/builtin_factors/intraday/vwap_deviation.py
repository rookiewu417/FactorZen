"""intraday/factors/technical/vwap_deviation.py — VWAP 偏离度因子。

factor_value = (close - vwap) / vwap
vwap = cumsum(amount) / cumsum(vol)，当日内累计。

直觉：factor > 0 表示当前价高于日内均价（反转信号）；factor < 0 表示低于均价（做多候选）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from factorzen.intraday.data.context import IntradayDataContext
from factorzen.intraday.factors.base import IntradayFactor


@dataclass
class VwapDeviation(IntradayFactor):
    name: str = "vwap_deviation"
    description: str = "当前价格与日内 VWAP 的偏离度"
    bar_size: str = "1min"
    lookback_bars: int = 0
    required_data: list[str] = field(default_factory=lambda: ["minute"])

    def compute(self, ctx: IntradayDataContext) -> pl.DataFrame:
        df = ctx.minute.collect()
        if df.is_empty():
            return pl.DataFrame(
                schema={"trade_time": pl.Datetime, "ts_code": pl.Utf8, "factor_value": pl.Float64}
            )

        df = df.sort(["ts_code", "trade_time"])

        df = (
            df.with_columns(pl.col("trade_time").dt.date().alias("_trade_date"))
            .with_columns(
                [
                    pl.col("amount")
                    .cum_sum()
                    .over(["ts_code", "_trade_date"])
                    .alias("_cum_amount"),
                    pl.col("vol").cum_sum().over(["ts_code", "_trade_date"]).alias("_cum_vol"),
                ]
            )
            .with_columns((pl.col("_cum_amount") / pl.col("_cum_vol")).alias("_vwap"))
            .with_columns(
                ((pl.col("close") - pl.col("_vwap")) / pl.col("_vwap")).alias("factor_value")
            )
            .select(["trade_time", "ts_code", "factor_value"])
        )

        return df.filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
