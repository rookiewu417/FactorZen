"""intraday/evaluation/backtest.py — 日内因子分层回测（聚合到日频后复用 daily 回测框架）。"""

from __future__ import annotations

import polars as pl

from daily.evaluation.backtest import BacktestResult, run_stratified_backtest


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
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_value",
    n_groups: int = 10,
    factor_name: str = "",
) -> BacktestResult:
    """日内因子分层回测。

    将分钟因子聚合到日频后，对齐日频收益进行分层回测。

    Args:
        minute_factor: 分钟级因子 DataFrame，含 trade_time/trade_date、ts_code、{factor_col}。
        daily_ret: 日频收益 DataFrame，含 trade_date、ts_code、ret。
        factor_col: 因子列名。
        n_groups: 分组数。
        factor_name: 因子名称。

    Returns:
        BacktestResult（复用 daily 框架）。
    """
    daily_factor = aggregate_intraday_factor(minute_factor, factor_col=factor_col)
    return run_stratified_backtest(
        daily_factor,
        daily_ret,
        factor_col=factor_col,
        n_groups=n_groups,
        factor_name=factor_name,
    )
