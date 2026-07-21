"""多因子合成评估管线。加载因子、合成、信号层评估一体化。"""

from __future__ import annotations

import logging

import polars as pl

from factorzen.daily.evaluation.ic_analysis import (
    ICAnalysisResult,
    compute_fwd_returns,
    compute_rank_ic,
)
from factorzen.daily.evaluation.signal_backtest import (
    SignalBacktestResult,
    run_signal_backtest,
)
from factorzen.daily.factors.registry import get_factor
from factorzen.research.combination.methods import equal_weight, ic_weighted, max_ir

_logger = logging.getLogger(__name__)

_IN_SAMPLE_METHODS = {"ic_weighted", "max_ir"}


def instantiate_factor(fname: str, registry_getter=get_factor):
    factor_cls = registry_getter(fname)
    return factor_cls()


def prepare_return_frame(
    price_df: pl.DataFrame, horizons: list[int] | None = None
) -> pl.DataFrame:
    ret_df = (
        price_df.select(["trade_date", "ts_code", "close"])
        .sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret")
        )
    )
    return compute_fwd_returns(ret_df, horizons=horizons, ret_col="ret")


def combine_and_evaluate(
    factor_dfs: dict[str, pl.DataFrame],
    price_df: pl.DataFrame,
    method: str = "equal_weight",
    horizons: list[int] | None = None,
    cost_bps: float = 0.0,
    ret_col: str = "ret",
) -> tuple[pl.DataFrame, ICAnalysisResult, SignalBacktestResult]:
    """合成多个因子并做信号层 IC / 分层评估。

    返回的是**信号层毛收益口径**评估（``return_basis=gross_signal_level``），
    不含停牌/涨跌停/T+1 等可交易性约束，**不是**可交易净值。
    可交易净值请走 ``fz combine backtest``。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}
        price_df: 含 trade_date, ts_code, {ret_col} 的价格 DataFrame
        method: 合成方法 ("equal_weight" | "ic_weighted" | "max_ir")
        horizons: IC 衰减窗口（默认 [1, 5]）
        cost_bps: 信号层提示性单边成本（bp），默认 0
        ret_col: 单日收益列名

    Returns:
        (combined_factor_df, ic_result, signal_backtest_result)
    """
    if horizons is None:
        horizons = [1, 5]

    ret_df = compute_fwd_returns(price_df, horizons=horizons, ret_col=ret_col)

    if method in _IN_SAMPLE_METHODS:
        _logger.warning(
            "[样本内警告] 合成方法 '%s' 使用全样本 IC 估权重，回测结果含样本内偏差，"
            "请勿将其视为 OOS 表现。",
            method,
        )

    # 合成
    if method == "equal_weight":
        combined = equal_weight(factor_dfs)
    elif method == "ic_weighted":
        # 用 1 日前向收益计算 IC 权重
        fwd1 = ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename({"fwd_ret_1d": "ret"})
        combined = ic_weighted(factor_dfs, fwd1)
    elif method == "max_ir":
        fwd1 = ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename({"fwd_ret_1d": "ret"})
        combined = max_ir(factor_dfs, fwd1)
    else:
        raise ValueError(f"未知合成方法: {method}，可选: equal_weight, ic_weighted, max_ir")

    combined = combined.rename({"factor_value": "factor_clean"})

    # IC 评估
    ic_result = compute_rank_ic(
        combined.rename({"factor_clean": "factor_clean"}),
        ret_df,
        factor_col="factor_clean",
        horizons=horizons,
    )

    # 信号层分层评估（复用已算好的 ret_df，不再额外算前向收益）
    signal_result = run_signal_backtest(
        combined,
        ret_df,
        factor_col="factor_clean",
        n_groups=5,
        cost_bps=cost_bps,
    )

    return combined, ic_result, signal_result
