"""因子预处理管线。配置驱动的多步处理流程。"""

from dataclasses import dataclass, field

import polars as pl

from daily.preprocessing.missing import fill_cross_sectional_median
from daily.preprocessing.neutralizer import neutralize_by_styles, neutralize_ols
from daily.preprocessing.normalizer import cross_sectional_zscore
from daily.preprocessing.outlier import mad_clip


@dataclass
class PreprocessingPipeline:
    steps: list[str] = field(default_factory=lambda: ["outlier", "missing", "normalize"])
    outlier_sigma: float = 3.0
    neutralize: bool = False
    neutralize_style: bool = False

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
                result = mad_clip(result, col=current_col, n_sigma=self.outlier_sigma)
                current_col = f"{current_col}_clip"
            elif step == "missing":
                result = fill_cross_sectional_median(result, col=current_col)
                current_col = f"{current_col}_fill"
            elif step == "normalize":
                result = cross_sectional_zscore(result, col=current_col)
                current_col = f"{current_col}_z"
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
    stock_basic: "pl.DataFrame | None" = None,
    daily_basic: "pl.DataFrame | None" = None,
) -> pl.DataFrame:
    """完整预处理：去极值 + 填充 + 标准化 + 行业/市值中性化"""
    return PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        neutralize=True,
    ).run(df, col=col, stock_basic=stock_basic, daily_basic=daily_basic)
