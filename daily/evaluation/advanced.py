"""高级单因子评估指标。

包含：
1. IC Decay 增强分析 — 多持有期 IC 衰减
2. Monotonicity — 分位收益单调性
3. Sector-stratified IC — 行业分层 IC
4. Size-stratified IC — 市值分层 IC
5. Factor Crowding — 因子拥挤度检测（实验性）
6. Market Regime IC — 市场状态分层 IC
7. Rank Autocorrelation — 因子排名自相关
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import spearmanr


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
        return (
            f"Horizon {self.horizon}d: IC_mean={self.ic_mean:.4f}, "
            f"IC_std={self.ic_std:.4f}"
        )


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
        ret_cols = [c for c in daily_ret.columns if re.match(r"fwd_ret_(\d+)d", c)]
        horizons = sorted(int(re.match(r"fwd_ret_(\d+)d", c).group(1)) for c in ret_cols)

    if not horizons:
        return []

    # 合并因子和收益
    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")
    trade_dates = merged["trade_date"].unique().sort().to_list()

    results: list[ICDecayResult] = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ics: list[float] = []
        for d in trade_dates:
            cross = merged.filter(pl.col("trade_date") == d)
            x = cross[factor_col].drop_nulls().to_numpy()
            mask = cross[factor_col].is_not_null()
            y_arr = cross.filter(mask)[ret_col].to_numpy()
            valid = ~np.isnan(x) & ~np.isnan(y_arr)
            if valid.sum() < 30:
                continue
            ic, _ = spearmanr(x[valid], y_arr[valid])
            h_ics.append(float(ic) if not np.isnan(ic) else 0.0)

        ic_arr = np.array(h_ics)
        results.append(ICDecayResult(
            horizon=h,
            ic_mean=float(np.mean(ic_arr)) if len(ic_arr) > 0 else float("nan"),
            ic_std=float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 0 else float("nan"),
            ic_series=h_ics,
        ))

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
    df = factor_df.with_columns(
        pl.col(factor_col).rank("ordinal", descending=False).over("trade_date").alias("_rank")
    ).with_columns(
        ((pl.col("_rank") - 1) * n_groups // pl.col("_rank").max().over("trade_date"))
        .cast(pl.Int32)
        .alias("group")
    ).drop("_rank")

    # 每组平均收益
    group_ret = (
        df.group_by(["trade_date", "group"])
        .agg(pl.col(ret_col).mean().alias("mean_ret"))
    )

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
    sectors = factor_df[sector_col].unique().to_list()
    ic_rows: list[dict] = []
    warnings: list[str] = []

    trade_dates = factor_df["trade_date"].unique().sort().to_list()
    for sector in sectors:
        sec_df = factor_df.filter(pl.col(sector_col) == sector)
        sec_ics: list[float] = []
        for d in trade_dates:
            cross = sec_df.filter(pl.col("trade_date") == d)
            if len(cross) < 2:
                continue
            x = cross[factor_col].drop_nulls().to_numpy()
            mask = cross[factor_col].is_not_null()
            y_arr = cross.filter(mask)[ret_col].to_numpy()
            valid = ~np.isnan(x) & ~np.isnan(y_arr)
            if valid.sum() < 2:
                continue
            ic, _ = spearmanr(x[valid], y_arr[valid])
            sec_ics.append(float(ic) if not np.isnan(ic) else 0.0)

        ic_mean = float(np.mean(sec_ics)) if sec_ics else float("nan")
        ic_rows.append({"sector": sector, "ic": ic_mean})

        if len(sec_df) < min_samples:
            warnings.append(f"Sector '{sector}' has only {len(sec_df)} samples (< {min_samples})")

    result_df = pl.DataFrame(ic_rows)

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
    df = factor_df.with_columns(
        pl.col(cap_col).rank("ordinal", descending=False).over("trade_date").alias("_cap_rank")
    ).with_columns(
        ((pl.col("_cap_rank") - 1) * n_buckets // pl.col("_cap_rank").max().over("trade_date"))
        .cast(pl.Int32)
        .alias("cap_bucket")
    ).drop("_cap_rank")

    # bucket labels
    if n_buckets == 2:
        labels = {0: "Small", 1: "Large"}
    elif n_buckets == 3:
        labels = {0: "Small", 1: "Mid", 2: "Large"}
    else:
        labels = {i: f"Bucket{i}" for i in range(n_buckets)}

    trade_dates = df["trade_date"].unique().sort().to_list()
    ic_rows: list[dict] = []
    buckets_dict: dict[str, float] = {}

    for bucket_id in range(n_buckets):
        bucket_df = df.filter(pl.col("cap_bucket") == bucket_id)
        bucket_ics: list[float] = []
        for d in trade_dates:
            cross = bucket_df.filter(pl.col("trade_date") == d)
            if len(cross) < 2:
                continue
            x = cross[factor_col].drop_nulls().to_numpy()
            mask = cross[factor_col].is_not_null()
            y_arr = cross.filter(mask)[ret_col].to_numpy()
            valid = ~np.isnan(x) & ~np.isnan(y_arr)
            if valid.sum() < 2:
                continue
            ic, _ = spearmanr(x[valid], y_arr[valid])
            bucket_ics.append(float(ic) if not np.isnan(ic) else 0.0)

        ic_mean = float(np.mean(bucket_ics)) if bucket_ics else float("nan")
        label = labels.get(bucket_id, f"Bucket{bucket_id}")
        ic_rows.append({"cap_bucket": label, "ic": ic_mean})
        buckets_dict[label] = ic_mean

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
        try:
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
            pairwise_rows.append({
                "factor_a": names[i],
                "factor_b": names[j],
                "corr": cum_corr[i][j],
            })
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
        market_ret = (
            factor_df.group_by("trade_date")
            .agg(pl.col(ret_col).mean().alias("market_return"))
        )
    else:
        if "market_return" in market_df.columns:
            market_ret = market_df
        else:
            # 从因子数据计算
            market_ret = (
                factor_df.group_by("trade_date")
                .agg(pl.col(ret_col).mean().alias("market_return"))
            )

    # 合并市场状态和因子数据
    merged = factor_df.join(market_ret, on="trade_date", how="inner")
    trade_dates = merged["trade_date"].unique().sort().to_list()

    if regime_type == "direction":
        # 按市场方向分组
        regimes: dict[str, list[str]] = {"up": [], "down": []}
        for d in trade_dates:
            mret = merged.filter(pl.col("trade_date") == d)["market_return"].mean()
            if mret is not None and mret > 0:
                regimes["up"].append(d)
            else:
                regimes["down"].append(d)

        ic_rows: list[dict] = []
        for regime_name in ["up", "down"]:
            regime_dates = regimes[regime_name]
            regime_ics: list[float] = []
            for d in regime_dates:
                cross = merged.filter(pl.col("trade_date") == d)
                x = cross[factor_col].drop_nulls().to_numpy()
                mask = cross[factor_col].is_not_null()
                y_arr = cross.filter(mask)[ret_col].to_numpy()
                valid = ~np.isnan(x) & ~np.isnan(y_arr)
                if valid.sum() < 2:
                    continue
                ic, _ = spearmanr(x[valid], y_arr[valid])
                regime_ics.append(float(ic) if not np.isnan(ic) else 0.0)

            ic_mean = float(np.mean(regime_ics)) if regime_ics else float("nan")
            ic_rows.append({"regime": regime_name, "ic": ic_mean})

        result_df = pl.DataFrame(ic_rows)

    elif regime_type == "volatility":
        # 按波动率分位分组
        if "market_volatility" not in merged.columns:
            # 如果没有波动率数据，用日收益绝对值作为代理
            merged = merged.with_columns(
                pl.col("market_return").abs().alias("market_volatility")
            )

        # 计算分位阈值
        vol_values = merged["market_volatility"].unique().drop_nulls().sort().to_numpy()
        if len(vol_values) < n_regimes:
            n_regimes = max(1, len(vol_values))

        quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
        thresholds = np.quantile(vol_values, quantiles) if len(vol_values) > 0 else np.array([])

        # 为每个日期分配波动率状态
        date_vol = merged.group_by("trade_date").agg(
            pl.col("market_volatility").first()
        ).sort("trade_date")

        regime_labels = ["low_vol", "mid_vol", "high_vol"][:n_regimes]
        if n_regimes > 3:
            regime_labels = [f"vol_{i}" for i in range(n_regimes)]

        ic_rows = []
        for ri in range(n_regimes):
            regime_name = regime_labels[ri] if ri < len(regime_labels) else f"vol_{ri}"
            if len(thresholds) == 0:
                regime_dates = trade_dates
            elif ri == 0:
                mask = date_vol["market_volatility"] <= thresholds[0]
                regime_dates = date_vol.filter(mask)["trade_date"].to_list()
            elif ri == n_regimes - 1:
                mask = date_vol["market_volatility"] > thresholds[-1]
                regime_dates = date_vol.filter(mask)["trade_date"].to_list()
            else:
                mask = (date_vol["market_volatility"] > thresholds[ri - 1]) & (
                    date_vol["market_volatility"] <= thresholds[ri]
                )
                regime_dates = date_vol.filter(mask)["trade_date"].to_list()

            regime_ics: list[float] = []
            for d in regime_dates:
                cross = merged.filter(pl.col("trade_date") == d)
                if len(cross) < 2:
                    continue
                x = cross[factor_col].drop_nulls().to_numpy()
                mask_f = cross[factor_col].is_not_null()
                y_arr = cross.filter(mask_f)[ret_col].to_numpy()
                valid = ~np.isnan(x) & ~np.isnan(y_arr)
                if valid.sum() < 2:
                    continue
                ic, _ = spearmanr(x[valid], y_arr[valid])
                regime_ics.append(float(ic) if not np.isnan(ic) else 0.0)

            ic_mean = float(np.mean(regime_ics)) if regime_ics else float("nan")
            ic_rows.append({"regime": regime_name, "ic": ic_mean})

        result_df = pl.DataFrame(ic_rows)
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
        # 对每个股票，计算当前排名与 lag 期前排名的相关性
        df_lag = df.with_columns(
            pl.col("_rank").shift(lag).over("ts_code").alias(f"_rank_lag{lag}")
        )

        # 每天计算截面 Spearman 相关
        trade_dates = df_lag["trade_date"].unique().sort().to_list()
        lag_ics: list[float] = []
        for d in trade_dates:
            cross = df_lag.filter(pl.col("trade_date") == d)
            rank_col = cross["_rank"].drop_nulls()
            lag_col = cross[f"_rank_lag{lag}"].drop_nulls()
            # 用内部 join 确保配对
            valid_cross = cross.filter(
                pl.col("_rank").is_not_null() & pl.col(f"_rank_lag{lag}").is_not_null()
            )
            if len(valid_cross) < 2:
                continue
            x = valid_cross["_rank"].to_numpy()
            y = valid_cross[f"_rank_lag{lag}"].to_numpy()
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() < 2:
                continue
            ic, _ = spearmanr(x[valid], y[valid])
            lag_ics.append(float(ic) if not np.isnan(ic) else 0.0)

        ac_mean = float(np.mean(lag_ics)) if lag_ics else 0.0
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
