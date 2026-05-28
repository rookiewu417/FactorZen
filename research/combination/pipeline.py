"""多因子合成评估管线。加载因子、合成、回测一体化。"""

from __future__ import annotations

import logging

import polars as pl

from daily.evaluation.backtest import BacktestResult, CostModel, run_stratified_backtest
from daily.evaluation.ic_analysis import ICAnalysisResult, compute_fwd_returns, compute_rank_ic
from research.combination.methods import equal_weight, ic_weighted, max_ir

_logger = logging.getLogger(__name__)

_IN_SAMPLE_METHODS = {"ic_weighted", "max_ir"}


def combine_and_evaluate(
    factor_dfs: dict[str, pl.DataFrame],
    price_df: pl.DataFrame,
    method: str = "equal_weight",
    horizons: list[int] | None = None,
    cost_model: CostModel | None = None,
    ret_col: str = "ret",
) -> tuple[pl.DataFrame, ICAnalysisResult, BacktestResult]:
    """合成多个因子并评估 IC / 回测性能。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}
        price_df: 含 trade_date, ts_code, {ret_col} 的价格 DataFrame
        method: 合成方法 ("equal_weight" | "ic_weighted" | "max_ir")
        horizons: IC 衰减窗口（默认 [1, 5]）
        cost_model: 成本模型，None 表示不扣成本
        ret_col: 单日收益列名

    Returns:
        (combined_factor_df, ic_result, backtest_result)
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

    # 回测
    bt_result = run_stratified_backtest(
        combined,
        price_df,
        factor_col="factor_clean",
        n_groups=5,
        cost_model=cost_model,
    )

    return combined, ic_result, bt_result
