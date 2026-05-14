"""Rank IC 分析。计算因子值与未来收益的截面 Spearman 相关系数。

性能说明：
    使用 polars group_by + pearson_corr(ranks) 替代逐日 Python for 循环，
    Spearman 相关系数 = 对排名后序列求 Pearson 相关系数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# 最少截面样本数（低于此值的交易日跳过）
_MIN_CROSS_SAMPLES = 30


def compute_fwd_returns(
    price_df: pl.DataFrame,
    horizons: list[int] | None = None,
    ret_col: str = "ret_1d",
) -> pl.DataFrame:
    """预计算各时间窗口的前向收益。

    Args:
        price_df: 含 trade_date, ts_code, {ret_col} 的日收益 DataFrame。
        horizons: 前向窗口（交易日），默认 [1, 5, 10, 20]。
        ret_col: 单日收益列名。

    Returns:
        含 fwd_ret_{h}d 列的 DataFrame。
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    df = price_df.sort(["ts_code", "trade_date"])
    for h in horizons:
        df = df.with_columns(
            pl.col(ret_col).shift(-h).over("ts_code").alias(f"fwd_ret_{h}d")
        )
    return df


@dataclass
class ICAnalysisResult:
    factor_name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_ratio: float
    n_periods: int
    ic_series: pl.DataFrame  # trade_date, ic
    decay: dict[int, float] = field(default_factory=dict)  # {horizon_days: ic_mean}
    frequency: str = "daily"

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(
            self.frequency, self.frequency
        )
        lines = [
            f"Factor: {self.factor_name} [{freq_label}]",
            f"  IC Mean: {self.ic_mean:.4f}  |  IC Std: {self.ic_std:.4f}  |  IR: {self.ir:.2f}",
            f"  IC > 0 Ratio: {self.ic_positive_ratio:.1%}  |  Periods: {self.n_periods}",
        ]
        if self.decay:
            decay_parts = [f"{h}d={v:.4f}" for h, v in sorted(self.decay.items())]
            lines.append(f"  IC Decay: {', '.join(decay_parts)}")
        return "\n".join(lines)


def _rank_ic_by_date(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    min_samples: int = _MIN_CROSS_SAMPLES,
) -> pl.DataFrame:
    """计算每日截面 Rank IC（polars vectorized 实现）。

    原理：Spearman 相关系数 = Pearson(rank(x), rank(y))
    用 polars 的 `.rank().over("trade_date")` + `pearson_corr` group_by 实现，
    避免 Python-level for 循环。

    Returns:
        pl.DataFrame with columns [trade_date, ic]，已按 trade_date 排序。
    """
    # 联合有效掩码（因子 + 前向收益都不为 null/inf）
    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    if valid_df.is_empty():
        return pl.DataFrame({"trade_date": [], "ic": []}).cast(
            {"trade_date": pl.Date, "ic": pl.Float64}
        )

    # 截面内排名（average 方法，与 scipy.stats.spearmanr 默认一致）
    ranked = valid_df.with_columns([
        pl.col(factor_col).rank(method="average").over("trade_date").alias("_factor_rank"),
        pl.col(ret_col).rank(method="average").over("trade_date").alias("_ret_rank"),
    ])

    # group_by 日期，计算排名 Pearson 相关（= Spearman），过滤样本不足的日期
    ic_df = (
        ranked.group_by("trade_date")
        .agg([
            pl.corr("_factor_rank", "_ret_rank").alias("ic"),
            pl.len().alias("_n"),
        ])
        .filter(pl.col("_n") >= min_samples)
        .drop("_n")
        .sort("trade_date")
    )
    return ic_df


def compute_rank_ic(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    horizons: list[int] | None = None,
    frequency: str = "daily",
) -> ICAnalysisResult:
    """计算 Rank IC（polars vectorized，消除逐日 Python for 循环）。

    Args:
        factor_df: 含 trade_date, ts_code, {factor_col} 的 DataFrame。
        daily_ret: 含 trade_date, ts_code, fwd_ret_{h}d 的 DataFrame
                   （通过 compute_fwd_returns() 预计算）。
        factor_col: 因子列名。
        horizons: IC decay 的时间窗口，默认 [1, 5, 10, 20]。
        frequency: 频率标签，用于 summary 显示。

    Returns:
        ICAnalysisResult。
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")

    # ---------- 主 IC（horizon=1d）----------
    ic_series = _rank_ic_by_date(merged, factor_col, "fwd_ret_1d")
    ic_values = ic_series["ic"].drop_nulls().to_numpy()

    ic_mean = float(np.mean(ic_values)) if len(ic_values) > 0 else 0.0
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_values > 0)) if len(ic_values) > 0 else 0.0

    # ---------- IC Decay（所有 horizon）----------
    decay: dict[int, float] = {}
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
        h_vals = h_ic_df["ic"].drop_nulls().to_numpy()
        if len(h_vals) > 0:
            decay[h] = float(np.mean(h_vals))

    return ICAnalysisResult(
        factor_name=factor_col,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_positive_ratio=ic_pos,
        n_periods=len(ic_values),
        ic_series=ic_series,
        decay=decay,
        frequency=frequency,
    )
