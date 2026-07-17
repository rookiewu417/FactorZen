"""行业因子构建：基于行业分类生成哑变量矩阵。"""

from __future__ import annotations

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def get_industry_dummies(
    stocks: pl.DataFrame,
    industry_col: str = "industry",
    industries: list[str] | None = None,
) -> pl.DataFrame:
    """生成行业哑变量矩阵（One-Hot 编码）。

    Args:
        stocks: 股票基本信息 DataFrame，需含 ts_code 和 industry_col 列。
                若含 trade_date 列，则按 (trade_date, ts_code) 维度输出；
                否则仅按 ts_code 维度输出。
        industry_col: 行业列名，默认 "industry"。
        industries: 固定行业全集（裸名，不含 ind_ 前缀）。若给定，即使当日/本批
            未出现某行业也输出该列（全 0），用于全窗并集稳定化（W2）。
            None 时用本批数据中出现的行业。

    Returns:
        DataFrame，含 ts_code（及 trade_date，若输入有）+ ind_XXX 列（每个行业一列，值为 0/1）。
    """
    if industry_col not in stocks.columns:
        raise ValueError(f"输入 DataFrame 中缺少行业列 '{industry_col}'")

    # 过滤掉行业为空的记录
    df = stocks.filter(pl.col(industry_col).is_not_null())

    if df.is_empty():
        logger.warning("过滤行业空值后无剩余数据")
        return df

    # 行业列集：固定全集或本批出现
    if industries is None:
        ind_list = sorted(df[industry_col].unique().to_list())
    else:
        ind_list = list(industries)

    if not ind_list:
        logger.warning("行业列表为空")
        return df.clear()

    # 确定 key 列
    has_trade_date = "trade_date" in df.columns
    key_cols = ["trade_date", "ts_code"] if has_trade_date else ["ts_code"]

    # 构建哑变量列（固定全集中未出现的行业 → 全 0）
    dummy_exprs = [
        pl.when(pl.col(industry_col) == ind)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias(f"ind_{ind}")
        for ind in ind_list
    ]

    result = df.select([*key_cols, pl.col(industry_col)]).with_columns(dummy_exprs)

    # 删除原始行业列
    result = result.drop(industry_col)

    return result


def get_industry_names(stocks: pl.DataFrame, industry_col: str = "industry") -> list[str]:
    """从股票基本信息中提取所有行业名称（排序后）。

    Args:
        stocks: 股票基本信息 DataFrame。
        industry_col: 行业列名。

    Returns:
        行业名称列表（按字典序排序）。
    """
    if industry_col not in stocks.columns:
        return []

    industries = (
        stocks.filter(pl.col(industry_col).is_not_null())[industry_col]
        .unique()
        .sort()
        .to_list()
    )
    return industries
