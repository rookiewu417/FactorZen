"""Market Regime IC — 市场状态分层 IC。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)
from factorzen.daily.evaluation.advanced._common import _grouped_ic  # noqa: E402


@dataclass
class MarketRegimeICResult:
    """市场状态分层 IC 结果。

    Attributes:
        factor_name: 因子名称
        regime_ic: 各状态 IC DataFrame (regime, ic)
        regime_type: 状态类型 ("direction" / "volatility")
    """

    factor_name: str = ""
    regime_ic: pl.DataFrame = field(default_factory=pl.DataFrame)
    regime_type: str = ""

    def summary(self) -> str:
        lines = [f"Market Regime IC [{self.regime_type}]: {self.factor_name}"]
        if not self.regime_ic.is_empty():
            for row in self.regime_ic.iter_rows(named=True):
                lines.append(f"  {row['regime']}: IC={row['ic']:.4f}")
        return "\n".join(lines)


def compute_market_regime_ic(
    factor_df: pl.DataFrame,
    market_df: pl.DataFrame | None = None,
    factor_col: str = "factor_clean",
    ret_col: str = "fwd_ret",
    regime_type: str = "direction",
    n_regimes: int = 2,
    return_object: bool = False,
) -> pl.DataFrame | MarketRegimeICResult:
    """按市场状态分组计算 Rank IC。

    支持两种状态划分方式：
    - "direction": 按市场收益率 > 0 (up/bull) 或 <= 0 (down/bear)
    - "volatility": 按市场波动率分位分组

    Args:
        factor_df: 因子值 DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}
        market_df: 市场状态 DataFrame，列: trade_date, market_return, [market_volatility]
        factor_col: 因子列名
        ret_col: 收益列名
        regime_type: "direction" 或 "volatility"
        n_regimes: volatility 模式下的状态数
        return_object: True 时返回 MarketRegimeICResult 对象

    Returns:
        pl.DataFrame (regime, ic) 或 MarketRegimeICResult
    """
    # 如果没有市场状态数据，从因子数据中计算等权市场收益
    if market_df is None:
        market_ret = factor_df.group_by("trade_date").agg(
            pl.col(ret_col).mean().alias("market_return")
        )
    else:
        if "market_return" in market_df.columns:
            market_ret = market_df
        else:
            # 从因子数据计算
            market_ret = factor_df.group_by("trade_date").agg(
                pl.col(ret_col).mean().alias("market_return")
            )

    # 合并市场状态和因子数据
    merged = factor_df.join(market_ret, on="trade_date", how="inner")

    if regime_type == "direction":
        # 标记每个交易日的涨跌方向
        date_regime = market_ret.with_columns(
            pl.when(pl.col("market_return") > 0)
            .then(pl.lit("up"))
            .otherwise(pl.lit("down"))
            .alias("regime")
        ).select(["trade_date", "regime"])
        merged_r = merged.join(date_regime, on="trade_date", how="inner")
        result_df = _grouped_ic(merged_r, factor_col, ret_col, group_col="regime")

    elif regime_type == "volatility":
        if "market_volatility" not in merged.columns:
            merged = merged.with_columns(pl.col("market_return").abs().alias("market_volatility"))

        vol_values = merged["market_volatility"].unique().drop_nulls().sort().to_numpy()
        if len(vol_values) < n_regimes:
            n_regimes = max(1, len(vol_values))

        quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
        thresholds = np.quantile(vol_values, quantiles) if len(vol_values) > 0 else np.array([])

        date_vol = merged.group_by("trade_date").agg(pl.col("market_volatility").first())

        base_labels = ["low_vol", "mid_vol", "high_vol"]
        regime_labels = (
            base_labels[:n_regimes] if n_regimes <= 3 else [f"vol_{i}" for i in range(n_regimes)]
        )

        # 用 polars 条件表达式分配 regime
        expr = pl.lit(regime_labels[-1])
        for ri in range(n_regimes - 2, -1, -1):
            threshold = thresholds[ri] if ri < len(thresholds) else float("inf")
            expr = (
                pl.when(pl.col("market_volatility") <= threshold)
                .then(pl.lit(regime_labels[ri]))
                .otherwise(expr)
            )

        date_regime = date_vol.with_columns(expr.alias("regime")).select(["trade_date", "regime"])
        merged_r = merged.join(date_regime, on="trade_date", how="inner")
        result_df = _grouped_ic(merged_r, factor_col, ret_col, group_col="regime")
    else:
        result_df = pl.DataFrame()

    if return_object:
        return MarketRegimeICResult(
            factor_name=factor_col,
            regime_ic=result_df,
            regime_type=regime_type,
        )
    return result_df
