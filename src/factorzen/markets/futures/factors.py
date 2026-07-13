"""国内商品期货因子叶子集与派生列。

与 A 股/crypto 不同：
- 价量来自主力连续序列，``close``/``open``/``high``/``low`` **已后复权**（continuous.py 拼接时
  乘 ``adj_factor``），故 ``ret_1d`` 直接用 ``close`` pct_change 即为真实连续收益、跨展期无跳变。
- 新增期货特有叶子 ``oi``（持仓量）+ 派生 ``oi_chg``（持仓变化率）。
- 无复权因子表/无财报/无 funding，``basic_features`` 为空（全部叶子来自 fut_daily，无需额外 join）。

**单位（实测确认，见计划 2.1 探测）：**
- ``open/high/low/close``：元/合约报价单位（如铜 元/吨），已后复权。
- ``vol``：成交量，手（lots）。``amount``：成交额，**万元**。``oi``：持仓量，手（lots）。
- ``vwap`` = amount/vol = 万元/手（品种内价格×合约乘数的活跃度代理，跨品种不可比），随
  ``adj_factor`` 复权以跨展期连续。``log_vol`` = ln(vol)（量列不复权）。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.base import FactorSet

# 叶子名 → 求值表列名。vwap/log_vol/ret_1d/oi_chg 为派生列（derived_columns 预计算）。
_LEAF_FEATURES: dict[str, str] = {
    "close": "close", "open": "open", "high": "high", "low": "low",
    "vol": "vol", "amount": "amount", "oi": "oi",
    "vwap": "vwap", "log_vol": "log_vol", "ret_1d": "ret_1d", "oi_chg": "oi_chg",
}


class FuturesFactorSet(FactorSet):
    def leaf_features(self) -> dict[str, str]:
        return dict(_LEAF_FEATURES)

    def basic_features(self) -> set[str]:
        return set()  # 全部叶子来自 fut_daily 主力连续帧，无需额外 join

    def derived_columns(self, bars: pl.DataFrame) -> pl.DataFrame:
        out = bars.sort(["ts_code", "trade_date"])
        # adj_factor 缺失（如裸测试帧）→ 视作 1.0（不复权），保证纯 derived 也能跑
        adj = pl.col("adj_factor") if "adj_factor" in out.columns else pl.lit(1.0)
        out = out.with_columns(
            # vwap：价格代理，随 adj_factor 复权 → 跨展期连续；vol=0 → null 防除零
            pl.when(pl.col("vol").abs() > 1e-12)
            .then(pl.col("amount") / pl.col("vol") * adj)
            .otherwise(None)
            .alias("vwap"),
            # log_vol：量列不复权；vol<=0 → null
            pl.when(pl.col("vol") > 0).then(pl.col("vol").log()).otherwise(None).alias("log_vol"),
            # ret_1d：close 已后复权 → pct_change 即真实连续收益、跨展期无跳变
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret_1d"),
        )
        # oi_chg：持仓变化率；展期日（换合约）置 null，避免换合约的机械虚假跳变
        oi_chg = pl.col("oi") / pl.col("oi").shift(1).over("ts_code") - 1.0
        if "mapping_ts_code" in out.columns:
            is_roll = (
                pl.col("mapping_ts_code") != pl.col("mapping_ts_code").shift(1).over("ts_code")
            ).fill_null(False)
            oi_chg = pl.when(is_roll).then(None).otherwise(oi_chg)
        return out.with_columns(oi_chg.alias("oi_chg"))
