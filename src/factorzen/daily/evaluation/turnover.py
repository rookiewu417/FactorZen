"""换手率分析。计算分组迁移矩阵与平均换手率。"""

from dataclasses import dataclass

import polars as pl

from factorzen.core.validation import require_columns


@dataclass
class TurnoverResult:
    factor_name: str
    avg_turnover: float  # 平均单边换手率
    migration_matrix: pl.DataFrame  # from_group, to_group, prob
    daily_turnover: pl.DataFrame  # trade_date, turnover
    frequency: str = "daily"

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(
            self.frequency, self.frequency
        )
        if self.frequency == "monthly":
            n = len(self.daily_turnover) if not self.daily_turnover.is_empty() else 0
            return f"Turnover [{freq_label}]: avg={self.avg_turnover:.4f} (N={n}) ⚠️ 月频样本极少"
        return f"Turnover [{freq_label}]: avg={self.avg_turnover:.4f}"


def compute_turnover(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    frequency: str = "daily",
) -> TurnoverResult:
    """计算因子分组换手率。

    Args:
        factor_df: 含 trade_date, ts_code, {factor_col}
        factor_col: 因子列名
        n_groups: 分组数
    """
    require_columns(factor_df, ["trade_date", "ts_code", factor_col], context="compute_turnover")
    # 每日分组
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

    # 按股票排序，计算上一期分组
    df = df.sort(["ts_code", "trade_date"]).with_columns(
        pl.col("group").shift(1).over("ts_code").alias("prev_group")
    )

    # 每日换手率 = 分组变更的股票比例
    daily_turnover = (
        df.filter(pl.col("prev_group").is_not_null())
        .with_columns((pl.col("group") != pl.col("prev_group")).cast(pl.Float64).alias("changed"))
        .group_by("trade_date")
        .agg(pl.col("changed").mean().alias("turnover"))
        .sort("trade_date")
    )

    avg_turnover = 0.0
    if not daily_turnover.is_empty():
        mean_turnover = daily_turnover["turnover"].mean()
        if isinstance(mean_turnover, (int, float)):
            avg_turnover = float(mean_turnover)

    # 迁移矩阵
    migration = (
        df.filter(pl.col("prev_group").is_not_null())
        .group_by(["prev_group", "group"])
        .len(name="count")
    )
    total_per_from = migration.group_by("prev_group").agg(pl.col("count").sum().alias("total"))
    migration = (
        migration.join(total_per_from, on="prev_group")
        .with_columns((pl.col("count") / pl.col("total")).alias("prob"))
        .select(["prev_group", "group", "prob"])
        .sort(["prev_group", "group"])
    )

    return TurnoverResult(
        factor_name="",
        avg_turnover=avg_turnover,
        migration_matrix=migration,
        daily_turnover=daily_turnover,
        frequency=frequency,
    )
