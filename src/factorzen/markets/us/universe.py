"""美股标的池：S&P 500 **静态成分固定池**（MVP）+ SPY 基准。

**幸存者偏差警示（MVP 已知限制，manifest/调用方 docstring 须转述）：**
``snapshot(d)`` 对任意 ``d`` 返回**同一份当前 S&P500 静态快照**（见 sp500_snapshot.py），
**不做 PIT 历史成分**——当年被剔除/退市的成分缺席、当年未纳入的却已在池内。真正的逐日
历史成分（Wikipedia 变更表回放）留二期，本 Phase 不做。

S&P 500 成分本身即「大型股 + 已隐含流动性过滤」，故 snapshot 不再额外做成交额/次新过滤
（避免为排序而预拉全 500 标的行情的双拉；成分即池）。``top_n`` 截断（字母序，MVP）供快 smoke。
基准：SPY（标普 500 ETF）后复权 close。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.markets.base import DataProvider, Universe
from factorzen.markets.us.sp500_snapshot import sp500_symbols


class USUniverse(Universe):
    def __init__(
        self,
        provider: DataProvider,
        top_n: int | None = None,
        benchmark_symbol: str = "SPY",
        symbols: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.top_n = top_n
        self.benchmark_symbol = benchmark_symbol
        self._symbols = symbols  # 注入（测试小池）；None → 静态 S&P500 快照

    def snapshot(self, d: date | str) -> list[str]:
        # 静态成分固定池（幸存者偏差，见模块 docstring）；忽略 d，不做 PIT 历史成分。
        base = self._symbols if self._symbols is not None else sp500_symbols()
        if self.top_n is not None and self.top_n < len(base):
            return list(base[: self.top_n])
        return list(base)

    def benchmark(self, start: str, end: str) -> pl.DataFrame:
        bars = self.provider.fetch_bars([self.benchmark_symbol], start, end)
        if bars.is_empty():
            return pl.DataFrame(schema={"trade_date": pl.Date, "close": pl.Float64})
        return bars.select("trade_date", "close").sort("trade_date")
