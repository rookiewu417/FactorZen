"""Rank Autocorrelation — 因子排名自相关。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RankAutocorrResult:
    """因子排名自相关结果。

    Attributes:
        factor_name: 因子名称
        autocorr_values: 各滞后期自相关系数列表
        mean_autocorr: 平均自相关
        half_life_est: 估计半衰期（期数）
        _lag_to_autocorr: 内部映射 {lag: autocorr}
    """

    factor_name: str = ""
    autocorr_values: list[float] = field(default_factory=list)
    mean_autocorr: float = 0.0
    half_life_est: float = 0.0
    _lag_to_autocorr: dict[int, float] = field(default_factory=dict)

    def get_lag(self, lag: int) -> float:
        """获取指定滞后期的自相关系数。

        Args:
            lag: 滞后期（1-based）

        Returns:
            自相关系数；0.0 如果 lag 不存在
        """
        return self._lag_to_autocorr.get(lag, 0.0)

    def summary(self) -> str:
        lines = [
            f"Rank Autocorr: {self.factor_name}",
            f"  Mean autocorr: {self.mean_autocorr:.4f}",
            f"  Half-life est: {self.half_life_est:.1f} periods",
        ]
        if self._lag_to_autocorr:
            for lag, ac in sorted(self._lag_to_autocorr.items()):
                lines.append(f"  Lag {lag}: {ac:.4f}")
        return "\n".join(lines)


def compute_rank_autocorr(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    lags: list[int] | None = None,
) -> RankAutocorrResult:
    """计算因子排名自相关：相邻期因子排名的 Spearman 相关系数。

    衡量因子信号的时序稳定性：高自相关 = 信号持久；低自相关 = 信号快速变化。

    Args:
        factor_df: DataFrame，列: trade_date, ts_code, {factor_col}
        factor_col: 因子列名
        lags: 滞后期列表，默认 [1]

    Returns:
        RankAutocorrResult
    """
    if lags is None:
        lags = [1]

    # 只保留有效因子值参与截面 rank（NaN 非 null，rank 排最大会污染秩自相关）
    factor_df = factor_df.filter(
        pl.col(factor_col).is_not_null() & pl.col(factor_col).is_not_nan()
    )

    # 按日期排序，计算每天的排名
    df = factor_df.sort(["ts_code", "trade_date"]).with_columns(
        pl.col(factor_col).rank("ordinal", descending=False).over("trade_date").alias("_rank")
    )

    lag_to_autocorr: dict[int, float] = {}
    autocorr_values: list[float] = []

    for lag in lags:
        lag_col = f"_rank_lag{lag}"
        df_lag = df.with_columns(pl.col("_rank").shift(lag).over("ts_code").alias(lag_col))
        # _rank 已经是截面内排名，直接对两列排名求 pearson_corr = Spearman 自相关
        ac_df = (
            df_lag.filter(pl.col("_rank").is_not_null() & pl.col(lag_col).is_not_null())
            .group_by("trade_date")
            .agg(
                [
                    pl.corr("_rank", lag_col).alias("ac"),
                    pl.len().alias("_n"),
                ]
            )
            .filter(pl.col("_n") >= 2)
            .drop("_n")
        )
        ac_arr = ac_df["ac"].drop_nulls().to_numpy()
        ac_mean = float(np.mean(ac_arr)) if len(ac_arr) > 0 else 0.0
        lag_to_autocorr[lag] = ac_mean
        autocorr_values.append(ac_mean)

    # 平均自相关（所有 lag 的均值）
    mean_autocorr = float(np.mean(autocorr_values)) if autocorr_values else 0.0

    # 半衰期估计 = -ln(2) / ln(mean_autocorr)
    # cap at reasonable max
    if mean_autocorr <= 0:
        half_life_est = 0.0
    elif mean_autocorr >= 1.0:
        half_life_est = 1000.0
    else:
        half_life_est = float(-np.log(2) / np.log(mean_autocorr))

    return RankAutocorrResult(
        factor_name=factor_col,
        autocorr_values=autocorr_values,
        mean_autocorr=mean_autocorr,
        half_life_est=half_life_est,
        _lag_to_autocorr=lag_to_autocorr,
    )
