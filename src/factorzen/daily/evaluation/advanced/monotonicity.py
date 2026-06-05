"""Monotonicity — 分位收益单调性。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonotonicityResult:
    """因子单调性分析结果。

    Attributes:
        factor_name: 因子名称
        monotonicity_score: 单调性得分 (0.0-1.0)，连续分位间收益方向一致的占比
        group_means: 各分组的平均收益
        direction: 方向 ("positive" / "negative")
    """

    factor_name: str = ""
    monotonicity_score: float = 0.0
    group_means: list[float] = field(default_factory=list)
    direction: str = "neutral"
    ols_slope: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Monotonicity: {self.factor_name}",
            f"  Score: {self.monotonicity_score:.4f}  Direction: {self.direction}",
            f"  OLS slope: {self.ols_slope:.6f}",
            f"  Group means: {[f'{m:.4f}' for m in self.group_means]}",
        ]
        return "\n".join(lines)


def compute_monotonicity(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    ret_col: str = "fwd_ret",
    n_groups: int = 10,
) -> MonotonicityResult:
    """计算因子单调性：按因子大小分组，检验各组收益是否单调。

    Args:
        factor_df: DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}
        factor_col: 因子列名
        ret_col: 收益列名
        n_groups: 分组数

    Returns:
        MonotonicityResult
    """
    df = (
        factor_df.with_columns(
            pl.col(factor_col).rank("ordinal", descending=False).over("trade_date").alias("_rank")
        )
        .with_columns(
            ((pl.col("_rank") - 1) * n_groups // pl.col("_rank").max().over("trade_date"))
            .cast(pl.Int32)
            .alias("group")
        )
        .drop("_rank")
    )

    # 每组平均收益
    group_ret = df.group_by(["trade_date", "group"]).agg(pl.col(ret_col).mean().alias("mean_ret"))

    # 各分组全局平均收益
    means_df = group_ret.group_by("group").agg(pl.col("mean_ret").mean()).sort("group")
    group_means = means_df["mean_ret"].to_list()

    # 单调性得分：连续分位间方向一致的比例
    if len(group_means) < 2:
        return MonotonicityResult(
            factor_name=factor_col,
            monotonicity_score=0.0,
            group_means=group_means,
            direction="neutral",
        )

    same_direction = 0
    for i in range(len(group_means) - 1):
        if (group_means[i + 1] - group_means[i]) * (group_means[-1] - group_means[0]) >= 0:
            same_direction += 1

    monotonicity_score = same_direction / (len(group_means) - 1)

    # 方向
    direction = "positive" if group_means[-1] > group_means[0] else "negative"

    # OLS slope: 线性拟合 group index → mean ret
    x_vals = np.arange(len(group_means))
    y_vals = np.array(group_means)
    if len(x_vals) >= 2 and np.std(x_vals) > 0:
        ols_slope = float(np.polyfit(x_vals, y_vals, 1)[0])
    else:
        ols_slope = 0.0

    return MonotonicityResult(
        factor_name=factor_col,
        monotonicity_score=monotonicity_score,
        group_means=group_means,
        direction=direction,
        ols_slope=ols_slope,
    )
