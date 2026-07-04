"""crypto 因子叶子集与派生列。

与 A 股不同：
- 无复权（无除权除息），``close`` 直接用（A 股用 ``close_adj``）。
- 无 ``pre_close``/隔夜缺口（24/7 连续），``ret_1d`` 用连续 close pct_change。
- 新增 crypto 特有叶子 ``funding_rate`` / ``open_interest``（需 join，类比 A 股 daily_basic）。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import FactorSet

# 叶子名 → 求值表列名。vwap/log_vol/ret_1d/taker_buy_ratio 为派生列(derived_columns 预计算)。
# ret_1d = 1 bar 收益(日频=日收益,intraday 随 freq 变);taker_buy_ratio = taker 买量/总量。
_LEAF_FEATURES: dict[str, str] = {
    "close": "close", "open": "open", "high": "high", "low": "low",
    "vol": "vol", "amount": "amount", "vwap": "vwap", "log_vol": "log_vol", "ret_1d": "ret_1d",
    "funding_rate": "funding_rate", "open_interest": "open_interest",
    "taker_buy_ratio": "taker_buy_ratio",
}
# 需额外 join 的非价量叶子（触发 funding/OI 拉取）
_BASIC_FEATURES: set[str] = {"funding_rate", "open_interest"}


class CryptoFactorSet(FactorSet):
    def leaf_features(self) -> dict[str, str]:
        return dict(_LEAF_FEATURES)

    def basic_features(self) -> set[str]:
        return set(_BASIC_FEATURES)

    def derived_columns(self, bars: pl.DataFrame) -> pl.DataFrame:
        out = bars.sort(["ts_code", "trade_date"])
        out2 = out.with_columns(
            pl.when(pl.col("vol").abs() > 1e-12)
            .then(pl.col("amount") / pl.col("vol"))
            .otherwise(None)
            .alias("vwap"),
            pl.when(pl.col("vol") > 0)
            .then(pl.col("vol").log())
            .otherwise(None)
            .alias("log_vol"),
            (pl.col("close") / pl.col("close").shift(1) - 1)
            .over("ts_code")
            .alias("ret_1d"),
        )
        # taker_buy_ratio:订单流失衡(taker 买量/总量);vol=0 → null 防除零;
        # 缺 taker_buy_volume 源列(ccxt 旧路径)→ 全 null,表达式安全退化
        tbr = (
            pl.when(pl.col("vol") > 0)
            .then(pl.col("taker_buy_volume") / pl.col("vol"))
            .otherwise(None)
            if "taker_buy_volume" in out.columns
            else pl.lit(None, dtype=pl.Float64)
        )
        return out2.with_columns(tbr.alias("taker_buy_ratio"))
