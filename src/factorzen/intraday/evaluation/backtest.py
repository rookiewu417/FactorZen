"""intraday/evaluation/backtest.py — 分钟因子聚合→日频信号层分层评估（毛收益口径）。"""

from __future__ import annotations

import polars as pl

from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
from factorzen.daily.evaluation.signal_backtest import (
    SignalBacktestResult,
    run_signal_backtest,
)


def aggregate_intraday_factor(
    minute_factor: pl.DataFrame,
    factor_col: str = "factor_value",
    time_col: str = "trade_time",
    date_col: str = "trade_date",
    code_col: str = "ts_code",
) -> pl.DataFrame:
    """将分钟级因子聚合到日频（取每日每股最后一个有效因子值）。

    Returns:
        日频 DataFrame，列：trade_date, ts_code, {factor_col}。
    """
    df = minute_factor.sort([code_col, time_col])

    if date_col not in df.columns:
        df = df.with_columns(pl.col(time_col).dt.date().alias(date_col))

    return (
        df.filter(pl.col(factor_col).is_not_null())
        .group_by([date_col, code_col])
        .agg(pl.col(factor_col).last())
        .sort([date_col, code_col])
    )


def run_intraday_backtest(
    minute_factor: pl.DataFrame,
    daily_price: pl.DataFrame,
    factor_col: str = "factor_value",
    n_groups: int = 10,
    factor_name: str = "",
    *,
    exec_lag: int = 0,
    exec_price_col: str | None = None,
) -> SignalBacktestResult:
    """分钟因子聚合后做日频信号层分层评估（毛收益口径）。

    将分钟因子聚合到日频，经 ``compute_fwd_returns`` 得到前向收益，
    再走 ``run_signal_backtest``。输出为研究口径毛收益，不含可交易性约束。

    Args:
        minute_factor: 分钟级因子 DataFrame，含 trade_time/trade_date、ts_code、{factor_col}。
        daily_price: 日频价格 DataFrame，含 trade_date、ts_code、open/close 等价格列。
        factor_col: 因子列名。
        n_groups: 分组数。
        factor_name: 因子名称。
        exec_lag: 成交滞后（交易日），原样透传 ``compute_fwd_returns``；默认 0。
        exec_price_col: 成交价格列，原样透传 ``compute_fwd_returns``；默认 None。

    Returns:
        SignalBacktestResult（信号层毛收益口径）。
    """
    daily_factor = aggregate_intraday_factor(minute_factor, factor_col=factor_col)
    fwd_returns = compute_fwd_returns(
        daily_price,
        exec_lag=exec_lag,
        exec_price_col=exec_price_col,
    )
    return run_signal_backtest(
        daily_factor,
        fwd_returns,
        factor_col=factor_col,
        n_groups=n_groups,
        factor_name=factor_name,
    )
