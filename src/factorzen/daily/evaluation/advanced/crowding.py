"""Factor Crowding — 因子拥挤度检测（实验性）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


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
