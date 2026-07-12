"""国内商品期货标的池：近 N 交易日主力连续成交额 Top-K，逐日 PIT 快照。

- 成交额 = 主力连续帧的 ``amount``（万元；见 factors.py 单位注释）。
- Top-N 品种（默认 40）——商品品种总数约 60-70，取活跃头部；小截面 IC 噪声大是已知特征
  （护栏不放松，smoke 如实记录 IC 分布，见计划 2.2）。
- 基准：MVP 用单个高流动性品种（默认螺纹钢 ``RB.SHF``）连续 close 作代理；缺失则回退池内首个。
  商品无单一「市场」基准，此为诚实标注的 MVP（挖掘链路不依赖基准）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from factorzen.markets.base import DataProvider, Universe


def _to_date(d: date | str) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y%m%d").date()
    return d


class FuturesUniverse(Universe):
    def __init__(
        self,
        provider: DataProvider,
        top_n: int = 40,
        lookback_days: int = 20,
        benchmark_code: str = "RB.SHF",
    ) -> None:
        self.provider = provider
        self.top_n = top_n
        self.lookback_days = lookback_days  # 交易日近似（用自然日窗口拉主力连续帧）
        self.benchmark_code = benchmark_code

    def snapshot(self, d: date | str) -> list[str]:
        d = _to_date(d)
        # lookback_days 交易日的自然日近似（*1.6 覆盖周末/节假日），主力连续帧只含交易日
        window_start = d - timedelta(days=int(self.lookback_days * 1.6) + 5)
        bars = self.provider.fetch_bars(
            None, window_start.strftime("%Y%m%d"), d.strftime("%Y%m%d")
        )
        if bars.is_empty():
            return []
        turnover = (
            bars.group_by("ts_code")
            .agg(pl.col("amount").sum().alias("tot_amount"))
            .filter(pl.col("tot_amount") > 0)
            .sort("tot_amount", descending=True)
            .head(self.top_n)
        )
        return turnover["ts_code"].to_list()

    def benchmark(self, start: str, end: str) -> pl.DataFrame:
        bars = self.provider.fetch_bars([self.benchmark_code], start, end)
        if bars.is_empty():
            bars = self.provider.fetch_bars(None, start, end)
            if bars.is_empty():
                return pl.DataFrame(schema={"trade_date": pl.Date, "close": pl.Float64})
            first = bars["ts_code"].to_list()[0]
            bars = bars.filter(pl.col("ts_code") == first)
        return bars.select("trade_date", "close").sort("trade_date")
