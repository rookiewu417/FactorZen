"""Intraday 分钟频数据上下文。为分钟频因子提供 lazy 数据加载。

通过 common.storage.load_parquet("minute", ...) 惰性加载分钟线数据，
支持 universe 过滤和 max_bars 安全上限。
"""

from dataclasses import dataclass, field

import polars as pl

from common.calendar import prev_trade_date
from common.storage import load_parquet


@dataclass
class IntradayDataContext:
    """分钟频因子计算的数据上下文。

    Attributes:
        start: 起始日期 "YYYYMMDD"。
        end: 结束日期 "YYYYMMDD"。
        bar_size: 分钟粒度，默认 "1min"。
        required_data: 需要加载的数据类型列表，默认 ["minute"]。
        universe: 股票 ts_code 列表，None 表示全部。
        max_bars: 每个标的加载的最大 bar 数，防止 OOM，默认 10_000。
    """

    start: str  # "YYYYMMDD"
    end: str  # "YYYYMMDD"
    bar_size: str = "1min"
    required_data: list[str] = field(default_factory=lambda: ["minute"])
    universe: list[str] | None = None
    max_bars: int = 10_000  # 安全上限，防止 OOM
    _minute: pl.LazyFrame | None = field(default=None, repr=False)

    @property
    def expanded_start(self) -> str:
        """返回考虑 lookback_bars 后扩展的起始日期。

        向前扩展 5 个交易日作为安全边界。
        """
        return prev_trade_date(self.start, n=5).strftime("%Y%m%d")

    @property
    def minute(self) -> pl.LazyFrame:
        """惰性加载分钟线数据（含扩展区间），按需过滤 universe 和 max_bars。"""
        if "minute" not in self.required_data:
            raise ValueError("minute data not declared in required_data")
        if self._minute is None:
            lf = load_parquet(
                "minute",
                start=self.expanded_start,
                end=self.end,
                date_col="trade_time",
            )
            if self.universe:
                lf = lf.filter(pl.col("ts_code").is_in(self.universe))
            # 应用 max_bars 安全限制（按 ts_code 分组取最近 max_bars 条）
            lf = (
                lf.sort(["ts_code", "trade_time"])
                .with_row_index("_mft_row_idx")
                .with_columns(pl.int_range(0, pl.len()).over("ts_code").alias("_mft_bar_idx"))
                .filter(
                    pl.col("_mft_bar_idx")
                    >= pl.col("_mft_bar_idx").max().over("ts_code") - self.max_bars
                )
                .drop(["_mft_row_idx", "_mft_bar_idx"])
            )
            self._minute = lf
        return self._minute

    def load_all(self) -> None:
        """强制加载所有声明的数据（触发惰性求值）。"""
        for data_type in self.required_data:
            if data_type == "minute":
                _ = self.minute

