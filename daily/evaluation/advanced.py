"""高级单因子评估指标。

包含：
1. IC Decay 增强分析 — 多持有期 IC 衰减
2. Monotonicity — 分位收益单调性
3. Sector-stratified IC — 行业分层 IC
4. Size-stratified IC — 市值分层 IC
5. Factor Crowding — 因子拥挤度检测（实验性）
6. Market Regime IC — 市场状态分层 IC
7. Rank Autocorrelation — 因子排名自相关
8. Neutralized IC — 行业/市值中性化后的 Rank IC
9. Event Study — 事件前后窗口累计收益分析
10. Factor Correlation — 多因子截面 Rank 相关性矩阵
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from common.logger import get_logger
from daily.evaluation.ic_analysis import IcStats, _build_ic_stats, _rank_ic_by_date

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


def _grouped_ic(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    group_col: str,
    min_per_cell: int = 2,
) -> pl.DataFrame:
    """在分组标签上计算截面 Rank IC，返回 (group_col_renamed_to_group → ic) DataFrame。

    原理：rank within (group, date) → pearson_corr grouped by (group, date) → mean by group。
    """
    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )
    if valid_df.is_empty():
        return pl.DataFrame({"regime": [], "ic": []})

    ranked = valid_df.with_columns(
        [
            pl.col(factor_col)
            .rank(method="average")
            .over([group_col, "trade_date"])
            .alias("_factor_rank"),
            pl.col(ret_col)
            .rank(method="average")
            .over([group_col, "trade_date"])
            .alias("_ret_rank"),
        ]
    )
    out_col = "regime" if group_col != "regime" else group_col
    return (
        ranked.group_by([group_col, "trade_date"])
        .agg(
            [
                pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                pl.len().alias("_n"),
            ]
        )
        .filter(pl.col("_n") >= min_per_cell)
        .drop("_n")
        .group_by(group_col)
        .agg(pl.col("ic").mean())
        .rename({group_col: out_col} if group_col != out_col else {})
        .sort(out_col)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IC Decay 增强分析
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ICDecayResult:
    """单个持有期的 IC 衰减结果。"""

    horizon: int
    ic_mean: float
    ic_std: float
    ic_series: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return f"Horizon {self.horizon}d: IC_mean={self.ic_mean:.4f}, IC_std={self.ic_std:.4f}"


def compute_ic_decay(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    horizons: list[int] | None = None,
) -> list[ICDecayResult]:
    """计算因子 IC 随持有期的衰减。

    对每个持有期 h，计算因子值与 fwd_ret_{h}d 的 Rank IC。

    Args:
        factor_df: 因子值 DataFrame，列: trade_date, ts_code, {factor_col}
        daily_ret: 前向收益 DataFrame，列: trade_date, ts_code, fwd_ret_{h}d
        factor_col: 因子列名
        horizons: 持有期列表；None 时从 daily_ret 列名自动检测

    Returns:
        list[ICDecayResult]: 每个 holding period 一个结果
    """
    # 自动检测 horizons
    if horizons is None:
        detected: list[int] = []
        for c in daily_ret.columns:
            match = re.fullmatch(r"fwd_ret_(\d+)d", c)
            if match is not None:
                detected.append(int(match.group(1)))
        horizons = sorted(detected)

    if not horizons:
        return []

    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")

    results: list[ICDecayResult] = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
        ic_arr = h_ic_df["ic"].drop_nulls().to_numpy()
        results.append(
            ICDecayResult(
                horizon=h,
                ic_mean=float(np.mean(ic_arr)) if len(ic_arr) > 0 else float("nan"),
                ic_std=float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else float("nan"),
                ic_series=ic_arr.tolist(),
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Monotonicity 分析
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sector-stratified IC
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SectorICResult:
    """行业分层 IC 结果。

    Attributes:
        factor_name: 因子名称
        sector_ic_df: 行业 IC DataFrame (sector, ic)
        low_sample_warnings: 低样本量警告列表
    """

    factor_name: str = ""
    sector_ic_df: pl.DataFrame = field(default_factory=pl.DataFrame)
    low_sample_warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Sector IC: {self.factor_name}"]
        if not self.sector_ic_df.is_empty():
            for row in self.sector_ic_df.iter_rows(named=True):
                lines.append(f"  {row['sector']}: IC={row['ic']:.4f}")
        if self.low_sample_warnings:
            lines.append("  Warnings:")
            for w in self.low_sample_warnings:
                lines.append(f"    {w}")
        return "\n".join(lines)


def compute_sector_ic(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    ret_col: str = "fwd_ret",
    sector_col: str = "sector",
    return_object: bool = False,
    min_samples: int = 30,
) -> pl.DataFrame | SectorICResult:
    """按行业分组计算 Rank IC。

    Args:
        factor_df: DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}, {sector_col}
        factor_col: 因子列名
        ret_col: 收益列名
        sector_col: 行业列名
        return_object: True 时返回 SectorICResult 对象
        min_samples: 触发低样本警告的阈值

    Returns:
        pl.DataFrame (sector, ic) 或 SectorICResult
    """
    valid_df = factor_df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    warnings: list[str] = []
    sector_counts = factor_df.group_by(sector_col).agg(pl.len().alias("_n"))
    for row in sector_counts.iter_rows(named=True):
        if row["_n"] < min_samples:
            warnings.append(
                f"Sector '{row[sector_col]}' has only {row['_n']} samples (< {min_samples})"
            )

    if valid_df.is_empty():
        result_df = pl.DataFrame({"sector": [], "ic": []})
    else:
        ranked = valid_df.with_columns(
            [
                pl.col(factor_col)
                .rank(method="average")
                .over([sector_col, "trade_date"])
                .alias("_factor_rank"),
                pl.col(ret_col)
                .rank(method="average")
                .over([sector_col, "trade_date"])
                .alias("_ret_rank"),
            ]
        )
        result_df = (
            ranked.group_by([sector_col, "trade_date"])
            .agg(
                [
                    pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                    pl.len().alias("_n"),
                ]
            )
            .filter(pl.col("_n") >= 2)
            .drop("_n")
            .group_by(sector_col)
            .agg(pl.col("ic").mean())
            .rename({sector_col: "sector"} if sector_col != "sector" else {})
            .sort("sector")
        )

    if return_object:
        return SectorICResult(
            factor_name=factor_col,
            sector_ic_df=result_df,
            low_sample_warnings=warnings,
        )
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Size-stratified IC
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SizeICResult:
    """市值分层 IC 结果。

    Attributes:
        factor_name: 因子名称
        buckets: 市值分桶关键词典 {bucket_name: ic_mean}
        summary: 文本摘要
    """

    factor_name: str = ""
    buckets: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def __str__(self) -> str:
        lines = [f"Size IC: {self.factor_name}"]
        for name, ic in self.buckets.items():
            lines.append(f"  {name}: IC={ic:.4f}")
        return "\n".join(lines)


def compute_size_ic(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    ret_col: str = "fwd_ret",
    cap_col: str = "market_cap",
    n_buckets: int = 3,
    return_object: bool = False,
) -> pl.DataFrame | SizeICResult:
    """按市值分组计算 Rank IC。

    Args:
        factor_df: DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}, {cap_col}
        factor_col: 因子列名
        ret_col: 收益列名
        cap_col: 市值列名
        n_buckets: 分桶数（默认 3: Large/Mid/Small）
        return_object: True 时返回 SizeICResult 对象

    Returns:
        pl.DataFrame (cap_bucket, ic) 或 SizeICResult
    """
    # 按市值排序分桶
    df = (
        factor_df.with_columns(
            pl.col(cap_col).rank("ordinal", descending=False).over("trade_date").alias("_cap_rank")
        )
        .with_columns(
            ((pl.col("_cap_rank") - 1) * n_buckets // pl.col("_cap_rank").max().over("trade_date"))
            .cast(pl.Int32)
            .alias("cap_bucket")
        )
        .drop("_cap_rank")
    )

    # bucket labels
    if n_buckets == 2:
        labels = {0: "Small", 1: "Large"}
    elif n_buckets == 3:
        labels = {0: "Small", 1: "Mid", 2: "Large"}
    else:
        labels = {i: f"Bucket{i}" for i in range(n_buckets)}

    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    ic_rows: list[dict] = []
    buckets_dict: dict[str, float] = {}

    if valid_df.is_empty():
        result_df = pl.DataFrame({"cap_bucket": [], "ic": []})
    else:
        ranked = valid_df.with_columns(
            [
                pl.col(factor_col)
                .rank(method="average")
                .over(["cap_bucket", "trade_date"])
                .alias("_factor_rank"),
                pl.col(ret_col)
                .rank(method="average")
                .over(["cap_bucket", "trade_date"])
                .alias("_ret_rank"),
            ]
        )
        bucket_ic_df = (
            ranked.group_by(["cap_bucket", "trade_date"])
            .agg(
                [
                    pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                    pl.len().alias("_n"),
                ]
            )
            .filter(pl.col("_n") >= 2)
            .drop("_n")
            .group_by("cap_bucket")
            .agg(pl.col("ic").mean())
            .sort("cap_bucket")
        )

        for row in bucket_ic_df.iter_rows(named=True):
            label = labels.get(row["cap_bucket"], f"Bucket{row['cap_bucket']}")
            ic_rows.append({"cap_bucket": label, "ic": row["ic"]})
            buckets_dict[label] = row["ic"]

        result_df = pl.DataFrame(ic_rows)

    if return_object:
        lines = [f"Size IC: {factor_col}"]
        for name, ic in buckets_dict.items():
            lines.append(f"  {name}: IC={ic:.4f}")
        return SizeICResult(
            factor_name=factor_col,
            buckets=buckets_dict,
            summary="\n".join(lines),
        )
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Factor Crowding（因子拥挤度 - 实验性）
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CrowdingResult:
    """因子拥挤度检测结果（**实验性指标**）。

    Attributes:
        factor_name: 因子名称（或 "multi-factor"）
        crowding_score: 拥挤度得分 (0.0-1.0)
        corr_matrix: 因子间截面相关性矩阵
        factor_names: 因子名称列表
        pairwise_corr: 因子对级相关性 DataFrame (factor_a, factor_b, corr)
        interpretation: 拥挤度解读 ("Low" / "Moderate" / "High")
        warnings: 警告列表
    """

    factor_name: str = ""
    crowding_score: float = 0.0
    corr_matrix: np.ndarray = field(default_factory=lambda: np.eye(1))
    factor_names: list[str] = field(default_factory=list)
    pairwise_corr: pl.DataFrame = field(default_factory=pl.DataFrame)
    interpretation: str = "Low"
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Factor Crowding [{self.factor_name}] ⚠️ EXPERIMENTAL",
            f"  Crowding Score: {self.crowding_score:.4f} ({self.interpretation})",
        ]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  Warning: {w}")
        return "\n".join(lines)


def compute_factor_crowding(
    factor_dict: dict[str, pl.DataFrame],
    factor_col: str = "factor_clean",
    n_groups: int = 10,
) -> CrowdingResult:
    """计算因子拥挤度（**实验性指标**）。

    通过计算多个因子间的截面相关性来检测因子过度拥挤风险。
    相关性越高，拥挤度越大。

    Args:
        factor_dict: {factor_name: DataFrame(trade_date, ts_code, {factor_col})}
        factor_col: 因子列名
        n_groups: 保留参数（暂未使用于简化计算）

    Returns:
        CrowdingResult
    """
    names = list(factor_dict.keys())
    n = len(names)

    if n < 2:
        return CrowdingResult(
            factor_name=names[0] if names else "",
            crowding_score=0.0,
            corr_matrix=np.eye(n),
            factor_names=names,
            pairwise_corr=pl.DataFrame(),
            interpretation="Low",
            warnings=["Need at least 2 factors for crowding analysis"],
        )

    # 合并所有因子到一个 DataFrame
    merged = None
    for name, df in factor_dict.items():
        renamed = df.select(["trade_date", "ts_code", pl.col(factor_col).alias(name)])
        if merged is None:
            merged = renamed
        else:
            merged = merged.join(renamed, on=["trade_date", "ts_code"], how="inner")

    if merged is None or merged.is_empty():
        return CrowdingResult(
            factor_name="multi-factor",
            crowding_score=0.0,
            corr_matrix=np.eye(n),
            factor_names=names,
            pairwise_corr=pl.DataFrame(),
            interpretation="Low",
        )

    # 对每个日期算截面相关性，然后平均
    dates = merged["trade_date"].unique().sort().to_list()
    cum_corr = np.zeros((n, n))
    count = 0

    for d in dates:
        cross = merged.filter(pl.col("trade_date") == d).drop_nulls()
        if len(cross) < 2:
            continue
        arr = np.column_stack([cross[name].to_numpy() for name in names])
        # 过滤常量列（std=0），避免 corrcoef 产生除零 warning
        stds = arr.std(axis=0)
        if np.any(stds == 0):
            continue
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(arr.T)
            if not np.any(np.isnan(corr)):
                cum_corr += corr
                count += 1
        except Exception:
            continue

    if count > 0:
        cum_corr /= count

    np.fill_diagonal(cum_corr, 1.0)

    # crowding score = 非对角线元素的平均绝对值
    non_diag_mask = ~np.eye(n, dtype=bool)
    non_diag_vals = cum_corr[non_diag_mask]
    if len(non_diag_vals) > 0:
        crowding_score = float(np.mean(np.abs(non_diag_vals)))
    else:
        crowding_score = 0.0

    # 解释
    if crowding_score > 0.7:
        interpretation = "High"
    elif crowding_score > 0.4:
        interpretation = "Moderate"
    else:
        interpretation = "Low"

    # 成对相关性 DataFrame
    pairwise_rows: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_rows.append(
                {
                    "factor_a": names[i],
                    "factor_b": names[j],
                    "corr": cum_corr[i][j],
                }
            )
    pairwise_df = pl.DataFrame(pairwise_rows)

    warnings: list[str] = []
    if crowding_score > 0.7:
        warnings.append(
            f"High crowding detected (score={crowding_score:.3f}). "
            "Factor signal uniqueness may be compromised."
        )
    warnings.append("⚠️ This metric is EXPERIMENTAL and not academically validated.")

    return CrowdingResult(
        factor_name="multi-factor",
        crowding_score=crowding_score,
        corr_matrix=cum_corr,
        factor_names=names,
        pairwise_corr=pairwise_df,
        interpretation=interpretation,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Market Regime IC
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Rank Autocorrelation
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Neutralized IC
# ═══════════════════════════════════════════════════════════════════════════════


def compute_neutralized_ic(
    factor_df: pl.DataFrame,
    ret_col: str = "ret_1d",
    neutralize_by: str = "industry+size",
    factor_col: str = "factor_clean",
) -> IcStats:
    """中性化因子后计算 Rank IC。

    Args:
        factor_df: DataFrame，必须含 trade_date, ts_code, {factor_col}, {ret_col}。
                   - 行业中性化需要 "industry" 列
                   - 市值中性化需要 "log_mktcap" 列（或 "total_mv" 列作为备选）
        ret_col: 收益列名（默认 "ret_1d"）
        neutralize_by: "industry" / "size" / "industry+size"（默认）
        factor_col: 因子列名（默认 "factor_clean"）

    Returns:
        IcStats — 中性化后的 Rank IC 统计结果
    """
    from daily.preprocessing.neutralizer import neutralize_ols

    # 根据 neutralize_by 决定传入 neutralize_ols 的参数
    stock_basic: pl.DataFrame | None = None
    daily_basic: pl.DataFrame | None = None

    do_industry = "industry" in neutralize_by
    do_size = "size" in neutralize_by

    if do_industry and "industry" in factor_df.columns:
        # 构造 stock_basic DataFrame（ts_code, industry）
        stock_basic = factor_df.select(["ts_code", "industry"]).unique(subset=["ts_code"])

    if do_size:
        # 支持 log_mktcap 或 total_mv 作为市值列
        if "log_mktcap" in factor_df.columns:
            # 将 log_mktcap 转为 total_mv（exp 反变换），供 neutralize_ols 使用
            daily_basic = factor_df.select(
                ["trade_date", "ts_code", pl.col("log_mktcap").exp().alias("total_mv")]
            )
        elif "total_mv" in factor_df.columns:
            daily_basic = factor_df.select(["trade_date", "ts_code", "total_mv"])

    if stock_basic is None and daily_basic is None:
        # 无法中性化，直接计算 Rank IC
        logger.warning(
            "compute_neutralized_ic: 缺少 %s 等中性化所需列，返回未中性化 IC", neutralize_by
        )
        ic_series = _rank_ic_by_date(factor_df, factor_col, ret_col)
        return _build_ic_stats(ic_series)

    # 调用 neutralize_ols，col 参数为 factor_col
    neutralized_df = neutralize_ols(
        factor_df,
        col=factor_col,
        stock_basic=stock_basic,
        daily_basic=daily_basic,
    )

    # 残差列名为 {factor_col}_neutral
    residual_col = f"{factor_col}_neutral"
    if residual_col not in neutralized_df.columns:
        # 回退：直接使用原因子
        ic_series = _rank_ic_by_date(factor_df, factor_col, ret_col)
        return _build_ic_stats(ic_series)

    # 用残差计算 Rank IC
    ic_series = _rank_ic_by_date(neutralized_df, residual_col, ret_col)
    return _build_ic_stats(ic_series)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Event Study
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EventStudyResult:
    """事件研究结果。

    Attributes:
        windows: 相对事件日的窗口列表，如 [-5, -4, ..., 0, ..., 20]
        avg_cumret: 各窗口日的平均累计收益（shape: len(windows)）
        ci_95: 95% 置信区间半宽（1.96 * std / sqrt(n_events)），shape: len(windows)
        n_events: 事件数量
    """

    windows: list[int]
    avg_cumret: np.ndarray
    ci_95: np.ndarray
    n_events: int


def compute_event_study(
    factor_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    event_threshold: float = 0.95,
    pre_window: int = 5,
    post_window: int = 20,
    factor_col: str = "factor_clean",
) -> EventStudyResult:
    """选 factor top event_threshold 分位作为事件，计算事件前后窗口平均累计收益。

    Args:
        factor_df: 含 trade_date, ts_code, {factor_col} 的因子 DataFrame
        ret_df: 含 trade_date, ts_code, ret_1d 的收益 DataFrame
        event_threshold: 事件阈值分位数（默认 0.95，即 top 5% 为事件）
        pre_window: 事件前窗口天数（默认 5）
        post_window: 事件后窗口天数（默认 20）
        factor_col: 因子列名

    Returns:
        EventStudyResult
    """
    windows = list(range(-pre_window, post_window + 1))
    n_windows = len(windows)

    # 过滤有效因子值
    valid_factor = factor_df.filter(pl.col(factor_col).is_not_null())
    if valid_factor.is_empty():
        return EventStudyResult(
            windows=windows,
            avg_cumret=np.zeros(n_windows),
            ci_95=np.zeros(n_windows),
            n_events=0,
        )

    # 按日期找 top event_threshold 分位的事件
    event_rows = (
        valid_factor.with_columns(
            pl.col(factor_col)
            .rank(method="average")
            .over("trade_date")
            .alias("_rank"),
            pl.len().over("trade_date").alias("_n"),
        )
        .filter(pl.col("_rank") / pl.col("_n") >= event_threshold)
        .select(["trade_date", "ts_code"])
    )

    if event_rows.is_empty():
        return EventStudyResult(
            windows=windows,
            avg_cumret=np.zeros(n_windows),
            ci_95=np.zeros(n_windows),
            n_events=0,
        )

    # 构建日期索引（用于窗口偏移）
    all_dates = sorted(ret_df["trade_date"].unique().to_list())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 对 ret_df 建立 (date, ts_code) -> ret_1d 的查找字典
    ret_lookup: dict[tuple, float] = {}
    for row in ret_df.iter_rows(named=True):
        ret_lookup[(row["trade_date"], row["ts_code"])] = float(row["ret_1d"])

    # 对每个事件，计算窗口内累计收益
    event_cumrets: list[np.ndarray] = []

    for row in event_rows.iter_rows(named=True):
        event_date = row["trade_date"]
        ts_code = row["ts_code"]

        if event_date not in date_to_idx:
            continue
        event_idx = date_to_idx[event_date]

        # 收集各窗口日的日收益
        daily_rets = []
        valid_event = True
        for w in windows:
            target_idx = event_idx + w
            if target_idx < 0 or target_idx >= len(all_dates):
                valid_event = False
                break
            target_date = all_dates[target_idx]
            ret = ret_lookup.get((target_date, ts_code), np.nan)
            daily_rets.append(ret)

        if not valid_event:
            continue

        # 计算以事件日（w=0）为基准的累计收益
        # cumret[i] = prod(1 + ret[event_day..i]) - 1
        daily_arr = np.array(daily_rets, dtype=float)

        # 如果缺失数据过多（超过 50%），跳过该事件
        nan_ratio = np.sum(np.isnan(daily_arr)) / len(daily_arr)
        if nan_ratio > 0.5:
            continue

        # w=0 对应 pre_window 索引
        base_idx = pre_window
        cumrets = np.zeros(n_windows)
        for i in range(n_windows):
            if i <= base_idx:
                # 事件前：反向累乘（忽略 NaN）
                segment = daily_arr[i : base_idx + 1]
                if len(segment) == 0:
                    cumrets[i] = 0.0
                else:
                    cumrets[i] = float(np.nanprod(1.0 + segment)) - 1.0
                    cumrets[i] = -cumrets[i]  # 负号：事件前为反向
            else:
                # 事件后：正向累乘（忽略 NaN）
                segment = daily_arr[base_idx : i + 1]
                cumrets[i] = float(np.nanprod(1.0 + segment)) - 1.0

        event_cumrets.append(cumrets)

    if len(event_cumrets) == 0:
        return EventStudyResult(
            windows=windows,
            avg_cumret=np.zeros(n_windows),
            ci_95=np.zeros(n_windows),
            n_events=0,
        )

    cumret_matrix = np.array(event_cumrets)  # shape: (n_events, n_windows)
    avg_cumret = np.mean(cumret_matrix, axis=0)
    n_events = len(event_cumrets)

    if n_events > 1:
        ci_95 = 1.96 * np.std(cumret_matrix, axis=0, ddof=1) / np.sqrt(n_events)
    else:
        ci_95 = np.zeros(n_windows)

    return EventStudyResult(
        windows=windows,
        avg_cumret=avg_cumret,
        ci_95=ci_95,
        n_events=n_events,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Factor Correlation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_factor_correlation(
    factor_dfs: dict[str, pl.DataFrame],
    factor_col: str = "factor_clean",
) -> pl.DataFrame:
    """计算多因子截面 Rank 相关性均值矩阵（Spearman 相关）。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, {factor_col})}
        factor_col: 因子列名

    Returns:
        pl.DataFrame — 含 "factor" 列及各因子列的方阵，值为平均 Spearman 相关系数。
    """
    names = list(factor_dfs.keys())
    n = len(names)

    if n == 0:
        return pl.DataFrame()

    if n == 1:
        return pl.DataFrame({"factor": names, names[0]: [1.0]})

    # 合并所有因子到宽表（每日截面）
    merged: pl.DataFrame | None = None
    for name, df in factor_dfs.items():
        col_renamed = df.select(["trade_date", "ts_code", pl.col(factor_col).alias(name)])
        if merged is None:
            merged = col_renamed
        else:
            merged = merged.join(col_renamed, on=["trade_date", "ts_code"], how="inner")

    if merged is None or merged.is_empty():
        # 返回单位矩阵
        data: dict[str, list] = {"factor": names}
        for n1 in names:
            data[n1] = [1.0 if n1 == n2 else 0.0 for n2 in names]
        return pl.DataFrame(data)

    # 对每个日期计算截面 Rank 相关，然后累加
    dates = merged["trade_date"].unique().sort().to_list()
    cum_corr = np.zeros((n, n))
    count = 0

    for d in dates:
        cross = merged.filter(pl.col("trade_date") == d).drop_nulls()
        if len(cross) < 2:
            continue
        arr = np.column_stack([cross[name].to_numpy() for name in names])
        stds = arr.std(axis=0)
        if np.any(stds == 0):
            continue
        # Spearman: rank then Pearson
        from scipy.stats import rankdata

        ranked_arr = np.column_stack([rankdata(arr[:, i]) for i in range(n)])
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(ranked_arr.T)
            if not np.any(np.isnan(corr)):
                cum_corr += corr
                count += 1
        except Exception:
            continue

    if count > 0:
        avg_corr = cum_corr / count
    else:
        avg_corr = np.eye(n)

    np.fill_diagonal(avg_corr, 1.0)

    # 构造 DataFrame（含 "factor" 列作为行标签）
    data2: dict[str, list] = {"factor": names}
    for j, col_name in enumerate(names):
        data2[col_name] = [float(avg_corr[i, j]) for i in range(n)]

    return pl.DataFrame(data2)


def apply_fdr_correction(
    p_values: dict[str, float],
    method: str = "fdr_bh",
) -> dict[str, float]:
    """对多因子批量评估的 p 值进行多重检验校正。

    Args:
        p_values: {因子名: p_value} 字典。
        method: statsmodels multipletests 支持的方法，如：
            "fdr_bh"（Benjamini-Hochberg，控制 FDR，默认）、
            "bonferroni"（Bonferroni，控制 FWER，更保守）、
            "fdr_by"（Benjamini-Yekutieli）。

    Returns:
        {因子名: 校正后 p 值} 字典，键顺序与输入一致。
    """
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return {}

    names = list(p_values.keys())
    raw_pvals = np.array([p_values[n] for n in names])

    _, pvals_corrected, _, _ = multipletests(raw_pvals, method=method)

    return {n: float(p) for n, p in zip(names, pvals_corrected, strict=True)}
