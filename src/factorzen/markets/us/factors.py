"""美股因子叶子集与派生列（价量族 MVP）。

与 A 股/crypto/期货不同：
- ``open/high/low/close`` 已由 provider **后复权**（``× adj_factor``，adj_factor=adjclose/close_raw），
  故 ``ret_1d`` 直接用 ``close`` pct_change 即真实收益、无拆股跳变。
- 叶子只有**价量族**：OHLCV + vwap/log_vol/ret_1d。**市值/基本面 PIT 留二期，不做**——
  prompt 也不广告不存在的叶子（能力层↔接线层漂移教训）。
- 无 funding/OI/财报，``basic_features`` 为空（全部叶子来自 Yahoo 日线帧，无需额外 join）。

**单位（见 provider.py 复权契约）：**
- ``open/high/low/close``：美元/股，**已后复权**。
- ``vol``：成交量，原始股数（**未复权**——拆股日该股自身股数会跳变，截面 level 因子里属
  个股偶发伪影，MVP 诚实标注不复权，同期货量列口径）。
- ``amount`` = ``close_raw × vol_raw`` = 美元成交额（**拆股不变量**，跨拆股连续的流动性度量）。
- ``vwap`` = 后复权典型价 ``(high+low+close)/3``（美股日线无盘中 vwap，用典型价作日频代理；
  H/L/C 已后复权 → 跨拆股连续）。``log_vol`` = ln(vol)。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import FactorSet

# 叶子名 → 求值表列名。vwap/log_vol/ret_1d 为派生列（derived_columns 预计算）。
_LEAF_FEATURES: dict[str, str] = {
    "close": "close", "open": "open", "high": "high", "low": "low",
    "vol": "vol", "amount": "amount",
    "vwap": "vwap", "log_vol": "log_vol", "ret_1d": "ret_1d",
}


class USFactorSet(FactorSet):
    def leaf_features(self) -> dict[str, str]:
        return dict(_LEAF_FEATURES)

    def basic_features(self) -> set[str]:
        return set()  # 全部叶子来自 Yahoo 日线后复权帧，无需额外 join

    def derived_columns(self, bars: pl.DataFrame) -> pl.DataFrame:
        out = bars.sort(["ts_code", "trade_date"])
        return out.with_columns(
            # vwap：后复权典型价 (high+low+close)/3（无盘中数据的日频 vwap 代理，跨拆股连续）
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap"),
            # log_vol：量列不复权；vol<=0 → null 防 log 负/零
            pl.when(pl.col("vol") > 0).then(pl.col("vol").log()).otherwise(None).alias("log_vol"),
            # ret_1d：close 已后复权 → pct_change 即真实收益、无拆股跳变
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret_1d"),
        )
