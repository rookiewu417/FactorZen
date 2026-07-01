"""因子暴露矩阵计算：汇总风格因子 + 行业哑变量。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger
from factorzen.risk.industry_factors import get_industry_dummies
from factorzen.risk.style_factors import (
    STYLE_FACTOR_NAMES,
    STYLE_FACTOR_REGISTRY,
    cs_standardize,
)

logger = get_logger(__name__)


@dataclass
class ExposureMatrix:
    """因子暴露矩阵。

    Attributes:
        codes: 股票代码列表，长度 n_stocks。
        factor_names: 因子名称列表，长度 n_factors（风格 + 行业）。
        matrix: 暴露矩阵，shape (n_stocks, n_factors)。
    """

    codes: list[str] = field(default_factory=list)
    factor_names: list[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)


def compute_exposures(
    daily_data: pl.DataFrame,
    daily_basic: pl.DataFrame,
    stocks: pl.DataFrame,
    trade_date: str | object,
    style_registry: dict[str, Callable] | None = None,
    style_names: list[str] | None = None,
) -> ExposureMatrix:
    """计算指定日期的因子暴露矩阵。

    流程：
    1. 对每个风格因子，使用注册函数计算因子值
    2. 截面标准化风格因子
    3. 生成行业哑变量
    4. 合并为暴露矩阵

    Args:
        daily_data: 日线行情 DataFrame（含历史数据，用于滚动窗口计算）。
        daily_basic: 每日估值指标 DataFrame（crypto 可复用 daily_data）。
        stocks: 标的基本信息 DataFrame，需含 ts_code, industry 列（crypto 的 industry=sector）。
        trade_date: 目标日期，str("YYYYMMDD" 或 "YYYY-MM-DD") 或 date 对象。
        style_registry: 风格因子注册表（默认 A 股 STYLE_FACTOR_REGISTRY），
            每个 fn 签名 ``(daily_data, daily_basic) -> [trade_date, ts_code, factor_value]``。
        style_names: 风格因子名列表（默认 A 股 STYLE_FACTOR_NAMES）。

    Returns:
        ExposureMatrix，包含该日所有可用标的的因子暴露。
    """
    import datetime as dt

    registry = STYLE_FACTOR_REGISTRY if style_registry is None else style_registry
    names = STYLE_FACTOR_NAMES if style_names is None else style_names

    # 标准化日期
    if isinstance(trade_date, str):
        if "-" in trade_date:
            target_date = dt.date.fromisoformat(trade_date)
        else:
            target_date = dt.datetime.strptime(trade_date, "%Y%m%d").date()
    elif isinstance(trade_date, dt.datetime):
        target_date = trade_date.date()
    elif isinstance(trade_date, dt.date):
        target_date = trade_date
    else:
        raise TypeError(f"不支持的日期类型: {type(trade_date)}")

    # ── 1. 计算风格因子并标准化 ─────────────────────────────────────────────────
    style_dfs: dict[str, pl.DataFrame] = {}
    for name in names:
        fn = registry[name]
        try:
            factor_df = fn(daily_data, daily_basic)
            if factor_df.is_empty():
                logger.warning(f"因子 {name} 在 {trade_date} 计算结果为空")
                continue

            # 截面标准化
            factor_df = cs_standardize(factor_df, "factor_value", method="mad")

            # 过滤到目标日期
            if factor_df["trade_date"].dtype == pl.Date:
                day_df = factor_df.filter(pl.col("trade_date") == target_date)
            else:
                day_df = factor_df.filter(pl.col("trade_date") == pl.lit(target_date))

            if not day_df.is_empty():
                style_dfs[name] = day_df.select(["ts_code", "factor_value"]).rename(
                    {"factor_value": name}
                )
        except Exception as e:
            logger.warning(f"因子 {name} 计算失败: {e}")

    if not style_dfs:
        logger.warning(f"日期 {trade_date} 无可用风格因子")
        return ExposureMatrix()

    # 合并风格因子
    style_names = list(style_dfs.keys())
    merged = style_dfs[style_names[0]]
    for name in style_names[1:]:
        merged = merged.join(style_dfs[name], on="ts_code", how="full", coalesce=True)

    # ── 2. 行业哑变量 ──────────────────────────────────────────────────────────
    # 匹配股票的行业信息
    ind_col_names: list[str] = []

    if "industry" in stocks.columns:
        # 获取当日有效股票的行业
        stock_ind = stocks.select(["ts_code", "industry"]).unique(subset=["ts_code"])
        ind_dummies = get_industry_dummies(stock_ind, industry_col="industry")

        if not ind_dummies.is_empty():
            # 获取行业列名
            ind_col_names = [c for c in ind_dummies.columns if c.startswith("ind_")]

            # 合并行业哑变量
            merged = merged.join(
                ind_dummies.select(["ts_code", *ind_col_names]),
                on="ts_code",
                how="left",
            )

            # 行业哑变量空值填 0
            merged = merged.with_columns(
                [pl.col(c).fill_null(0.0) for c in ind_col_names]
            )

    # ── 3. 构建暴露矩阵 ────────────────────────────────────────────────────────
    # 过滤掉所有风格因子全为 null 的股票
    merged = merged.drop_nulls(subset=style_names[:1])  # 至少第一个风格因子不为空

    # 填充剩余 null 为 0
    for name in style_names:
        merged = merged.with_columns(pl.col(name).fill_null(0.0))

    codes = merged["ts_code"].to_list()
    factor_names = style_names + ind_col_names

    # 构建 numpy 矩阵
    matrix_cols = [merged[c].to_numpy().astype(np.float64) for c in factor_names]
    if matrix_cols:
        matrix = np.column_stack(matrix_cols)
    else:
        matrix = np.empty((len(codes), 0))

    return ExposureMatrix(codes=codes, factor_names=factor_names, matrix=matrix)
