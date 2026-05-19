"""MAD 去极值。每个截面做 median ± n_sigma * MAD 截尾。"""

import polars as pl

from config.constants import MAD_GAUSSIAN_CONST


def mad_clip(
    df: pl.DataFrame,
    col: str = "factor_value",
    n_sigma: float = 3.0,
) -> pl.DataFrame:
    """MAD 截尾去极值。

    在每个 trade_date 截面上，以中位数 ± n_sigma * 1.4826 * MAD 为界，
    超出范围的因子值替换为 None。

    Parameters
    ----------
    df : pl.DataFrame
        必须包含 trade_date 和 col 两列。
    col : str, default "factor_value"
        待处理的因子列名。
    n_sigma : float, default 3.0
        MAD 倍数，越大界越宽。

    Returns
    -------
    pl.DataFrame
        原始 DataFrame 加上新列 {col}_clip，值为截尾后的因子值。
    """
    out_col = f"{col}_clip"
    return df.with_columns(
        pl.when(
            (pl.col(col).is_not_null())
            & (
                pl.col(col)
                >= pl.col(col).median().over("trade_date")
                - n_sigma
                * MAD_GAUSSIAN_CONST
                * (pl.col(col) - pl.col(col).median().over("trade_date"))
                .abs()
                .median()
                .over("trade_date")
            )
            & (
                pl.col(col)
                <= pl.col(col).median().over("trade_date")
                + n_sigma
                * MAD_GAUSSIAN_CONST
                * (pl.col(col) - pl.col(col).median().over("trade_date"))
                .abs()
                .median()
                .over("trade_date")
            )
        )
        .then(pl.col(col))
        .otherwise(None)
        .alias(out_col)
    )


# ---------------------------------------------------------------------------
# Percentile winsorize
# ---------------------------------------------------------------------------


def winsorize_percentile(
    df: pl.DataFrame,
    factor_col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """按交易日截面 percentile 裁剪异常值。

    Parameters
    ----------
    df : pl.DataFrame
        必须包含 trade_date 和 factor_col 两列。
    factor_col : str
        待处理的因子列名。
    lower : float, default 0.01
        下分位数（0~1 之间）。
    upper : float, default 0.99
        上分位数（0~1 之间）。

    Returns
    -------
    pl.DataFrame
        原地替换 factor_col 列的截尾结果（与 mad_clip 不同，不新增列）。
    """
    lo = pl.col(factor_col).quantile(lower, interpolation="linear").over("trade_date")
    hi = pl.col(factor_col).quantile(upper, interpolation="linear").over("trade_date")
    return df.with_columns(pl.col(factor_col).clip(lo, hi))


# ---------------------------------------------------------------------------
# Sigma clip (mean ± n·std)
# ---------------------------------------------------------------------------


def sigma_clip(
    df: pl.DataFrame,
    factor_col: str,
    n_sigma: float = 3.0,
) -> pl.DataFrame:
    """均值 ± n·std 截断。

    Parameters
    ----------
    df : pl.DataFrame
        必须包含 trade_date 和 factor_col 两列。
    factor_col : str
        待处理的因子列名。
    n_sigma : float, default 3.0
        标准差倍数，越大界越宽。

    Returns
    -------
    pl.DataFrame
        原地替换 factor_col 列的截尾结果。
    """
    mean = pl.col(factor_col).mean().over("trade_date")
    std = pl.col(factor_col).std().over("trade_date")
    return df.with_columns(
        pl.col(factor_col).clip(mean - n_sigma * std, mean + n_sigma * std)
    )
