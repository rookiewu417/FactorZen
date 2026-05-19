"""因子预处理管线。配置驱动的多步处理流程。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from daily.preprocessing.missing import fill_cross_sectional_median
from daily.preprocessing.neutralizer import neutralize_by_styles, neutralize_ols
from daily.preprocessing.normalizer import (
    cross_sectional_rank,
    cross_sectional_zscore,
    quantile_transform,
)
from daily.preprocessing.outlier import mad_clip, sigma_clip, winsorize_percentile

# Literal type aliases for IDE/mypy support
OutlierMethod = Literal["mad", "winsorize", "sigma"]
NormalizerMethod = Literal["zscore", "rank_uniform", "rank_normal", "quantile_normal"]


@dataclass
class PreprocessingPipeline:
    steps: list[str] = field(default_factory=lambda: ["outlier", "missing", "normalize"])
    outlier_sigma: float = 3.0
    neutralize: bool = False
    neutralize_style: bool = False
    # New in Phase 2: method selection
    outlier_method: OutlierMethod = "mad"
    normalizer_method: NormalizerMethod = "zscore"

    def run(
        self,
        df: pl.DataFrame,
        col: str = "factor_value",
        stock_basic: pl.DataFrame | None = None,
        daily_basic: pl.DataFrame | None = None,
        style_dfs: list[pl.DataFrame] | None = None,
        industry_map: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        result = df
        current_col = col

        for step in self.steps:
            if step == "outlier":
                result = _apply_outlier(
                    result, current_col, self.outlier_method, self.outlier_sigma
                )
                # mad_clip produces a new _clip column; winsorize/sigma overwrite in-place
                if self.outlier_method == "mad":
                    current_col = f"{current_col}_clip"
            elif step == "missing":
                result = fill_cross_sectional_median(result, col=current_col)
                current_col = f"{current_col}_fill"
            elif step == "normalize":
                result = _apply_normalizer(result, current_col, self.normalizer_method)
                if self.normalizer_method == "zscore":
                    current_col = f"{current_col}_z"
                # rank_* and quantile_* overwrite factor_col in-place (no new suffix)
            else:
                raise ValueError(f"未知预处理步骤: {step}")

        if self.neutralize:
            result = neutralize_ols(
                result, col=current_col, stock_basic=stock_basic, daily_basic=daily_basic
            )
            current_col = f"{current_col}_neutral"

        if self.neutralize_style and style_dfs:
            result = neutralize_by_styles(
                result, style_dfs=style_dfs, industry_map=industry_map, col=current_col
            )
            current_col = f"{current_col}_style_neutral"

        # 标记最终列
        result = result.with_columns(pl.col(current_col).alias("factor_clean"))
        return result


def _apply_outlier(
    df: pl.DataFrame,
    col: str,
    method: OutlierMethod,
    sigma: float,
) -> pl.DataFrame:
    """Dispatch to the appropriate outlier function."""
    if method == "mad":
        return mad_clip(df, col=col, n_sigma=sigma)
    if method == "winsorize":
        return winsorize_percentile(df, factor_col=col)
    if method == "sigma":
        return sigma_clip(df, factor_col=col, n_sigma=sigma)
    raise ValueError(f"未知 outlier_method: {method!r}")


def _apply_normalizer(
    df: pl.DataFrame,
    col: str,
    method: NormalizerMethod,
) -> pl.DataFrame:
    """Dispatch to the appropriate normalizer function."""
    if method == "zscore":
        return cross_sectional_zscore(df, col=col)
    if method == "rank_uniform":
        return cross_sectional_rank(df, factor_col=col, method="uniform")
    if method == "rank_normal":
        return cross_sectional_rank(df, factor_col=col, method="normal")
    if method == "quantile_normal":
        return quantile_transform(df, factor_col=col, output="normal")
    raise ValueError(f"未知 normalizer_method: {method!r}")


def quick_preprocess(
    df: pl.DataFrame,
    col: str = "factor_value",
    style_dfs: list[pl.DataFrame] | None = None,
    industry_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """快速预处理：去极值 + 填充 + 标准化，可选 Barra style 中性化。"""
    neutralize_style = bool(style_dfs)
    return PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        neutralize=False,
        neutralize_style=neutralize_style,
    ).run(df, col=col, style_dfs=style_dfs, industry_map=industry_map)


def full_preprocess(
    df: pl.DataFrame,
    col: str = "factor_value",
    stock_basic: pl.DataFrame | None = None,
    daily_basic: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """完整预处理：去极值 + 填充 + 标准化 + 行业/市值中性化"""
    return PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        neutralize=True,
    ).run(df, col=col, stock_basic=stock_basic, daily_basic=daily_basic)
