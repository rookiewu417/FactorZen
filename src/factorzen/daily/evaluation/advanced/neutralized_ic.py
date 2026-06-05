"""Neutralized IC — 行业/市值中性化后的 Rank IC。"""

from __future__ import annotations

import polars as pl

from factorzen.core.logger import get_logger
from factorzen.daily.evaluation.ic_analysis import IcStats, _build_ic_stats, _rank_ic_by_date

logger = get_logger(__name__)


def compute_neutralized_ic(
    factor_df: pl.DataFrame,
    ret_col: str = "ret_1d",
    neutralize_by: str = "industry+size",
    factor_col: str = "factor_clean",
) -> IcStats:
    """中性化因子后计算 Rank IC。

    Args:
        factor_df: DataFrame，必须含 trade_date, ts_code, {factor_col}, {ret_col}。
                   - 行业中性化需要 "industry" 列
                   - 市值中性化需要 "log_mktcap" 列（或 "total_mv" 列作为备选）
        ret_col: 收益列名（默认 "ret_1d"）
        neutralize_by: "industry" / "size" / "industry+size"（默认）
        factor_col: 因子列名（默认 "factor_clean"）

    Returns:
        IcStats — 中性化后的 Rank IC 统计结果
    """
    from factorzen.daily.preprocessing.neutralizer import neutralize_ols

    # 根据 neutralize_by 决定传入 neutralize_ols 的参数
    stock_basic: pl.DataFrame | None = None
    daily_basic: pl.DataFrame | None = None

    do_industry = "industry" in neutralize_by
    do_size = "size" in neutralize_by

    if do_industry and "industry" in factor_df.columns:
        # 构造 stock_basic DataFrame（ts_code, industry）
        stock_basic = factor_df.select(["ts_code", "industry"]).unique(subset=["ts_code"])

    if do_size:
        # 支持 log_mktcap 或 total_mv 作为市值列
        if "log_mktcap" in factor_df.columns:
            # 将 log_mktcap 转为 total_mv（exp 反变换），供 neutralize_ols 使用
            daily_basic = factor_df.select(
                ["trade_date", "ts_code", pl.col("log_mktcap").exp().alias("total_mv")]
            )
        elif "total_mv" in factor_df.columns:
            daily_basic = factor_df.select(["trade_date", "ts_code", "total_mv"])

    if stock_basic is None and daily_basic is None:
        # 无法中性化，直接计算 Rank IC
        logger.warning(
            "compute_neutralized_ic: 缺少 %s 等中性化所需列，返回未中性化 IC", neutralize_by
        )
        ic_series = _rank_ic_by_date(factor_df, factor_col, ret_col)
        return _build_ic_stats(ic_series)

    # 调用 neutralize_ols，col 参数为 factor_col
    neutralized_df = neutralize_ols(
        factor_df,
        col=factor_col,
        stock_basic=stock_basic,
        daily_basic=daily_basic,
    )

    # 残差列名为 {factor_col}_neutral
    residual_col = f"{factor_col}_neutral"
    if residual_col not in neutralized_df.columns:
        # 回退：直接使用原因子
        ic_series = _rank_ic_by_date(factor_df, factor_col, ret_col)
        return _build_ic_stats(ic_series)

    # 用残差计算 Rank IC
    ic_series = _rank_ic_by_date(neutralized_df, residual_col, ret_col)
    return _build_ic_stats(ic_series)
