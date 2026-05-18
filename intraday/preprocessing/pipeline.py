"""Intraday 因子预处理管线。分钟频因子的缺失值填充与异常值截尾。"""

from dataclasses import dataclass

import polars as pl


@dataclass
class IntradayPreprocessingPipeline:
    """Intraday 因子预处理管线。

    配置驱动的处理流程：缺失值填充 → 异常值截尾 → 产出 factor_clean 列。
    不包含 Z-score 标准化（待后续实现）。

    Attributes:
        do_fill_missing: 是否执行 forward-fill 缺失值。
        do_clip_outliers: 是否执行分位数截尾。
        clip_lower_pct: 截尾下界分位数，默认 1.0。
        clip_upper_pct: 截尾上界分位数，默认 99.0。
    """

    do_fill_missing: bool = True
    do_clip_outliers: bool = True
    clip_lower_pct: float = 1.0
    clip_upper_pct: float = 99.0

    def run(self, df: pl.DataFrame, col: str = "factor_value") -> pl.DataFrame:
        """按配置运行预处理，返回带 factor_clean 列的 DataFrame。

        Args:
            df: 输入 DataFrame，必须包含 col 列。
            col: 因子值列名，默认 "factor_value"。

        Returns:
            包含 factor_clean 列的结果 DataFrame。
        """
        result = df
        if self.do_fill_missing:
            result = fill_missing_bars(result)
        if self.do_clip_outliers:
            result = clip_outliers(
                result, col=col, lower_pct=self.clip_lower_pct, upper_pct=self.clip_upper_pct
            )
        result = result.with_columns(pl.col(col).alias("factor_clean"))
        return result


def fill_missing_bars(
    df: pl.DataFrame,
    time_col: str = "trade_time",
    group_col: str = "ts_code",
) -> pl.DataFrame:
    """Forward-fill 缺失的分钟 bar 因子值，但不跨交易日。

    按股票和交易日期分组、时间排序后，对 factor_value 做 forward-fill。

    Args:
        df: 输入 DataFrame，必须包含 factor_value, ts_code, trade_time 列。
        time_col: 时间列名，默认 "trade_time"。
        group_col: 分组列名，默认 "ts_code"。

    Returns:
        填充后的 DataFrame（factor_value 列已原地更新）。
    """
    helper_col = "_trade_date_for_fill"
    return (
        df.sort([group_col, time_col])
        .with_columns(pl.col(time_col).dt.date().alias(helper_col))
        .with_columns(pl.col("factor_value").forward_fill().over([group_col, helper_col]))
        .drop(helper_col)
    )


def clip_outliers(
    df: pl.DataFrame,
    col: str = "factor_value",
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> pl.DataFrame:
    """基于分位数的异常值截尾。

    计算 col 列的 lower_pct 和 upper_pct 分位数，
    将超出范围的值截断到边界。

    Args:
        df: 输入 DataFrame。
        col: 待处理的因子列名。
        lower_pct: 下界分位数（0-100）。
        upper_pct: 上界分位数（0-100）。

    Returns:
        截尾后的 DataFrame（col 列已原地更新）。
    """
    lo = df[col].quantile(lower_pct / 100.0)
    hi = df[col].quantile(upper_pct / 100.0)
    return df.with_columns(pl.col(col).clip(lo, hi))


# 向后兼容别名
MFTPreprocessingPipeline = IntradayPreprocessingPipeline
