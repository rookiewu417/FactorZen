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


def _winsorize_series(s: pl.Series, lower: float, upper: float) -> pl.Series:
    lo = s.quantile(lower, interpolation="linear")
    hi = s.quantile(upper, interpolation="linear")
    if lo is None or hi is None:
        return s
    return s.clip(lo, hi)


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
    return df.with_columns(
        pl.col(factor_col)
        .over("trade_date")
        .map_batches(lambda s: _winsorize_series(s, lower, upper))
        .alias(factor_col)
    )


# ---------------------------------------------------------------------------
# Sigma clip (mean ± n·std)
# ---------------------------------------------------------------------------


def _sigma_clip_series(s: pl.Series, n_sigma: float) -> pl.Series:
    mean = s.mean()
    std = s.std()
    if mean is None or std is None or std == 0:
        return s
    return s.clip(mean - n_sigma * std, mean + n_sigma * std)


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
    return df.with_columns(
        pl.col(factor_col)
        .over("trade_date")
        .map_batches(lambda s: _sigma_clip_series(s, n_sigma))
        .alias(factor_col)
    )
