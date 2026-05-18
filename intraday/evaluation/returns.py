"""intraday/evaluation/returns.py — 日内前向收益计算。"""

from __future__ import annotations

import polars as pl


def compute_intraday_fwd_returns(
    minute_df: pl.DataFrame,
    periods: list[int] | None = None,
    close_col: str = "close",
    time_col: str = "trade_time",
    code_col: str = "ts_code",
) -> pl.DataFrame:
    """计算分钟级前向收益，不跨交易日取下一根 bar。

    Args:
        minute_df: 含 trade_time、ts_code、close 的分钟线 DataFrame。
        periods: 前向 bar 数列表，默认 [1, 5, 15, 60]。
        close_col: 价格列名。
        time_col: 时间列名。
        code_col: 股票代码列名。

    Returns:
        原 DataFrame 追加 fwd_ret_{N}bar 列（末尾 N 行每股为 null）。
    """
    if periods is None:
        periods = [1, 5, 15, 60]

    helper_col = "_trade_date_for_fwd_ret"
    df = minute_df.sort([code_col, time_col]).with_columns(
        pl.col(time_col).dt.date().alias(helper_col)
    )
    group_keys = [code_col, helper_col]
    for n in periods:
        future_close = pl.col(close_col).shift(-n).over(group_keys)
        df = df.with_columns((future_close / pl.col(close_col) - 1).alias(f"fwd_ret_{n}bar"))
    return df.drop(helper_col)
