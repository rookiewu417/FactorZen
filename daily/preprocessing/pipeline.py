"""因子预处理管线。配置驱动的多步处理流程。"""

from dataclasses import dataclass, field
import polars as pl
from daily.preprocessing.outlier import mad_clip
from daily.preprocessing.missing import fill_cross_sectional_median
from daily.preprocessing.normalizer import cross_sectional_zscore
from daily.preprocessing.neutralizer import neutralize_ols


@dataclass
class PreprocessingPipeline:
    steps: list[str] = field(default_factory=lambda: ["outlier", "missing", "normalize"])
    outlier_sigma: float = 3.0
    neutralize: bool = False

    def run(
        self,
        df: pl.DataFrame,
        col: str = "factor_value",
        stock_basic: pl.DataFrame | None = None,
        daily_basic: pl.DataFrame | None = None,
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
            result = neutralize_ols(result, col=current_col, stock_basic=stock_basic, daily_basic=daily_basic)
            current_col = f"{current_col}_neutral"

        # 标记最终列
        result = result.with_columns(pl.col(current_col).alias("factor_clean"))
        return result


def quick_preprocess(df: pl.DataFrame, col: str = "factor_value") -> pl.DataFrame:
    """快速预处理：去极值 + 填充 + 标准化（不做中性化）"""
    return PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        neutralize=False,
    ).run(df, col=col)


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
