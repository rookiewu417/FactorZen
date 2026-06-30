"""因子暴露矩阵计算：汇总风格因子 + 行业哑变量。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.loader import fetch_index_member_all
from factorzen.core.logger import get_logger
from factorzen.risk.industry_factors import get_industry_dummies
from factorzen.risk.style_factors import (
    STYLE_FACTOR_NAMES,
    STYLE_FACTOR_REGISTRY,
    cs_standardize,
)

logger = get_logger(__name__)

# 行业归属降级为非 PIT（stocks.industry）时，进程内只警告一次，避免逐日刷屏。
_pit_industry_warned = False


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
) -> ExposureMatrix:
    """计算指定日期的因子暴露矩阵。

    流程：
    1. 对每个风格因子，使用注册函数计算因子值
    2. 截面标准化风格因子
    3. 生成行业哑变量
    4. 合并为暴露矩阵

    Args:
        daily_data: 日线行情 DataFrame（含历史数据，用于滚动窗口计算）。
        daily_basic: 每日估值指标 DataFrame。
        stocks: 股票基本信息 DataFrame，需含 ts_code, industry 列。
        trade_date: 目标日期，str("YYYYMMDD" 或 "YYYY-MM-DD") 或 date 对象。

    Returns:
        ExposureMatrix，包含该日所有可用股票的因子暴露。
    """
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
    for name in STYLE_FACTOR_NAMES:
        fn = STYLE_FACTOR_REGISTRY[name]
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
    # 匹配股票的行业信息：优先用 PIT 历史行业成分（index_member_all）按 trade_date
    # 做归属查找；任何原因不可用时降级为 stocks 里的（非 PIT）industry 列。
    ind_col_names: list[str] = []

    stock_ind = _lookup_pit_industry(target_date)
    if stock_ind is None and "industry" in stocks.columns:
        # 降级：使用调用方传入的（当前/非 PIT）行业分类
        stock_ind = stocks.select(["ts_code", "industry"]).unique(subset=["ts_code"])

    if stock_ind is not None and not stock_ind.is_empty():
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


def _warn_pit_industry_unavailable(reason: str) -> None:
    """降级为非 PIT 行业分类时发出一次性警告（避免每次 compute_exposures 调用刷屏）。"""
    global _pit_industry_warned
    if not _pit_industry_warned:
        logger.warning(
            f"[compute_exposures] PIT 历史行业数据不可用（{reason}），"
            "降级使用 stocks.industry（非 PIT，可能用当前行业分类污染历史窗口的因子收益回归）"
        )
        _pit_industry_warned = True


def _lookup_pit_industry(target_date: dt.date) -> pl.DataFrame | None:
    """优先尝试用 Tushare 历史行业成分（``index_member_all``）做 PIT 行业归属查找。

    任何原因不可用（无权限/无 token/网络失败/字段缺失/该日期无匹配记录等）均
    优雅降级，返回 ``None``，调用方应回退到 ``stocks`` 里的（非 PIT）industry
    列。降级只 warning 一次，不逐日刷屏。

    Args:
        target_date: 查询日期（PIT 截面日期）。

    Returns:
        pl.DataFrame，含 ts_code、industry 两列（该 target_date 实际归属的一级
        行业名）；PIT 数据不可用时返回 ``None``。
    """
    try:
        membership = fetch_index_member_all()
    except Exception as e:  # 双保险：fetch_index_member_all 自身已兜底，理论不会抛出
        _warn_pit_industry_unavailable(f"获取异常: {e}")
        return None

    if membership is None or membership.is_empty():
        _warn_pit_industry_unavailable("数据源不可用或为空")
        return None

    required_cols = {"ts_code", "l1_name", "in_date", "out_date"}
    if not required_cols.issubset(set(membership.columns)):
        _warn_pit_industry_unavailable(f"字段缺失，需要 {sorted(required_cols)}")
        return None

    # PIT 归属查找：in_date <= target_date < (out_date 或仍在该行业则不设上限)
    asof = (
        membership.filter(
            (pl.col("in_date") <= pl.lit(target_date))
            & (pl.col("out_date").is_null() | (pl.col("out_date") > pl.lit(target_date)))
        )
        .sort("in_date")
        .unique(subset=["ts_code"], keep="last")
        .select(["ts_code", "l1_name"])
        .rename({"l1_name": "industry"})
    )

    if asof.is_empty():
        _warn_pit_industry_unavailable(f"{target_date} 无匹配记录")
        return None

    return asof
