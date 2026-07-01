"""A 股 FactorSet port —— 叶子字典复用 discovery.operators（单一来源）。

派生列公式与 discovery/factor.py 的内联预计算一致（vwap=amount/vol,
log_vol=ln(vol+1), ret_1d 用 close_adj）。MC1 引擎 rewire 时把 factor.py 的内联
块替换为对本方法的调用，统一到单路径（避免双份漂移）。
"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.operators import BASIC_FEATURES, LEAF_FEATURES
from factorzen.markets.base import FactorSet


class AShareFactorSet(FactorSet):
    def leaf_features(self) -> dict[str, str]:
        return dict(LEAF_FEATURES)

    def basic_features(self) -> set[str]:
        return set(BASIC_FEATURES)

    def derived_columns(self, bars: pl.DataFrame) -> pl.DataFrame:
        # 与 discovery/factor.py:50-55 一致：A 股用复权价 close_adj、log(vol+1)。
        out = bars.sort(["ts_code", "trade_date"])
        return out.with_columns(
            (pl.col("amount") / pl.col("vol")).alias("vwap"),
            (pl.col("vol") + 1.0).log().alias("log_vol"),
        ).with_columns(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias(
                "ret_1d"
            )
        )
