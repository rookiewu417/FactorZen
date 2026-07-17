"""因子暴露矩阵计算：汇总风格因子 + 行业哑变量。

性能：``materialize_style_panel`` / ``materialize_industry_panel`` 在 build 入口
对全窗一次物化；逐日 ``compute_exposures`` 仅切片，避免 O(T)×全窗重算。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
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


def materialize_style_panel(
    daily_data: pl.DataFrame,
    daily_basic: pl.DataFrame,
    style_registry: dict[str, Callable] | None = None,
    style_names: list[str] | None = None,
    *,
    standardize: bool = True,
) -> pl.DataFrame:
    """全窗一次计算风格因子，可选按日截面标准化。

    每个风格因子对整段输入只算一次；``cs_standardize`` 按 trade_date 分组
    （见 style_factors.cs_standardize），与逐日全窗重算再 filter 当日数值等价。

    Args:
        daily_data / daily_basic: 含 lookback 预热的历史。
        style_registry / style_names: 默认可走 A 股 8 因子。
        standardize: True 时对每列做 MAD+Z-score 截面标准化。

    Returns:
        DataFrame ``[trade_date, ts_code, <style...>]``；某因子全程为空则不出现该列。
    """
    registry = STYLE_FACTOR_REGISTRY if style_registry is None else style_registry
    names = STYLE_FACTOR_NAMES if style_names is None else style_names

    merged: pl.DataFrame | None = None
    for name in names:
        fn = registry[name]
        try:
            factor_df = fn(daily_data, daily_basic)
            if factor_df.is_empty():
                logger.warning(f"因子 {name} 计算结果为空（全窗）")
                continue
            if standardize:
                factor_df = cs_standardize(factor_df, "factor_value", method="mad")
            day_col = factor_df.select(
                ["trade_date", "ts_code", pl.col("factor_value").alias(name)]
            )
            if merged is None:
                merged = day_col
            else:
                merged = merged.join(day_col, on=["trade_date", "ts_code"], how="full", coalesce=True)
        except Exception as e:
            logger.warning(f"因子 {name} 计算失败: {e}")

    if merged is None:
        return pl.DataFrame(
            schema={"trade_date": pl.Date, "ts_code": pl.Utf8}
        )
    return merged


def standardize_style_panel(panel: pl.DataFrame, style_names: list[str] | None = None) -> pl.DataFrame:
    """对已物化的 raw 风格面板按列做截面标准化（research 按调仓 universe 重标时用）。

    每列按 trade_date 做 MAD winsorize + Z-score（与 ``cs_standardize`` 同公式），
    向量化一次完成，避免逐列 join。
    """
    if panel.is_empty():
        return panel
    names = style_names or [c for c in panel.columns if c not in ("trade_date", "ts_code")]
    names = [n for n in names if n in panel.columns]
    if not names:
        return panel

    out = panel
    for name in names:
        # 与 cs_standardize 同构：median/MAD/mean/std 均 .over("trade_date")
        out = out.with_columns(pl.col(name).cast(pl.Float64)).with_columns(
            pl.col(name).median().over("trade_date").alias("_m"),
            (pl.col(name) - pl.col(name).median().over("trade_date"))
            .abs()
            .median()
            .over("trade_date")
            .alias("_mad"),
        ).with_columns(
            (pl.col("_mad") * 1.4826).alias("_mads"),
        ).with_columns(
            pl.col(name)
            .clip(
                pl.col("_m") - 3.0 * pl.col("_mads"),
                pl.col("_m") + 3.0 * pl.col("_mads"),
            )
            .alias(name),
        ).with_columns(
            pl.col(name).mean().over("trade_date").alias("_mu"),
            pl.col(name).std().over("trade_date").alias("_sd"),
        ).with_columns(
            pl.when(pl.col("_sd") > 1e-12)
            .then((pl.col(name) - pl.col("_mu")) / pl.col("_sd"))
            .otherwise(pl.lit(0.0))
            .alias(name),
        ).drop(["_m", "_mad", "_mads", "_mu", "_sd"])
    return out


def materialize_industry_panel(
    stocks: pl.DataFrame,
    trade_dates: list,
    *,
    industry_names: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """全窗行业暴露面板：PIT 按日归属 + **行业列全窗并集**（缺列填 0）。

    行业归属仍按当日快照（PIT）；列集用窗口并集，消除「中途新行业 → 因子名漂移 → 丢日」。
    实现上优先向量化：dates ⨯ membership 区间 join，避免 484 次 Python 循环。

    Args:
        stocks: 需含 ts_code；fallback 用 industry 列。
        trade_dates: 回归窗口交易日列表。
        industry_names: 若给定则强制使用该行业全集（排序后加 ind_ 前缀的列名列表
            或裸行业名均可——裸名会自动加 ind_）；None 时从窗口并集推断。

    Returns:
        (panel, ind_col_names)：panel 含 trade_date, ts_code, ind_*；ind_col_names 排序稳定。
    """
    if not trade_dates:
        return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8}), []

    # 规范 trade_dates 为 date
    norm_dates: list[dt.date] = []
    for d in trade_dates:
        if isinstance(d, dt.datetime):
            norm_dates.append(d.date())
        elif isinstance(d, dt.date):
            norm_dates.append(d)
        elif isinstance(d, str):
            norm_dates.append(
                dt.date.fromisoformat(d) if "-" in d else dt.datetime.strptime(d, "%Y%m%d").date()
            )
        else:
            raise TypeError(f"不支持的日期类型: {type(d)}")

    fallback_ind = (
        stocks.select(["ts_code", "industry"]).unique(subset=["ts_code"])
        if "industry" in stocks.columns
        else None
    )
    relevant_codes = stocks["ts_code"].unique().to_list()

    long_ind = _vectorized_pit_industry(norm_dates, relevant_codes, fallback_ind)

    if long_ind is None or long_ind.is_empty():
        return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8}), []

    # 固定行业列集
    if industry_names is not None:
        bare = [n[4:] if n.startswith("ind_") else n for n in industry_names]
        industries = sorted(set(bare))
    else:
        industries = sorted(
            long_ind.filter(pl.col("industry").is_not_null())["industry"].unique().to_list()
        )

    if not industries:
        return long_ind.select(["trade_date", "ts_code"]).unique(), []

    ind_col_names = [f"ind_{ind}" for ind in industries]
    dummies = get_industry_dummies(
        long_ind, industry_col="industry", industries=industries
    )
    return dummies, ind_col_names


def _vectorized_pit_industry(
    norm_dates: list[dt.date],
    relevant_codes: list[str],
    fallback_ind: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """向量化构造 (trade_date × ts_code → industry)。

    优先 membership 区间匹配；缺口用 stocks.industry 补齐。membership 不可用时
    直接用 fallback × dates 笛卡尔积（静态行业，仍产出按日面板以统一下游）。
    """
    dates_df = pl.DataFrame({"trade_date": norm_dates}).with_columns(
        pl.col("trade_date").cast(pl.Date)
    )
    codes_df = pl.DataFrame({"ts_code": relevant_codes})

    membership = None
    try:
        membership = fetch_index_member_all()
    except Exception as e:
        _warn_pit_industry_unavailable(f"获取异常: {e}")
        membership = None

    if membership is None or membership.is_empty():
        _warn_pit_industry_unavailable("数据源不可用或为空")
        if fallback_ind is None:
            return None
        # 静态行业 × 全日期
        return dates_df.join(fallback_ind, how="cross")

    required_cols = {"ts_code", "l1_name", "in_date", "out_date"}
    if not required_cols.issubset(set(membership.columns)):
        _warn_pit_industry_unavailable(f"字段缺失，需要 {sorted(required_cols)}")
        if fallback_ind is None:
            return None
        return dates_df.join(fallback_ind, how="cross")

    mem = membership.filter(pl.col("ts_code").is_in(relevant_codes)).select(
        ["ts_code", "l1_name", "in_date", "out_date"]
    )
    if mem.is_empty():
        _warn_pit_industry_unavailable("PIT 数据对本次请求的股票代码无覆盖")
        if fallback_ind is None:
            return None
        return dates_df.join(fallback_ind, how="cross")

    # 对每个 membership 区间，展开落在 [in_date, out_date) 内的交易日
    # join 条件：in_date <= trade_date AND (out_date is null OR out_date > trade_date)
    # polars: cross join + filter（dates 少 ~500，membership 过滤后可控）
    expanded = (
        mem.join(dates_df, how="cross")
        .filter(
            (pl.col("in_date") <= pl.col("trade_date"))
            & (pl.col("out_date").is_null() | (pl.col("out_date") > pl.col("trade_date")))
        )
        .sort(["ts_code", "trade_date", "in_date"])
        .unique(subset=["ts_code", "trade_date"], keep="last")
        .select(["trade_date", "ts_code", pl.col("l1_name").alias("industry")])
    )

    # 完整骨架：所有 codes × dates，PIT 命中 left join，缺口 fallback
    skeleton = dates_df.join(codes_df, how="cross")
    long_ind = skeleton.join(expanded, on=["trade_date", "ts_code"], how="left")

    if fallback_ind is not None:
        long_ind = long_ind.join(
            fallback_ind.rename({"industry": "_fb_ind"}), on="ts_code", how="left"
        ).with_columns(
            pl.when(pl.col("industry").is_null())
            .then(pl.col("_fb_ind"))
            .otherwise(pl.col("industry"))
            .alias("industry")
        ).drop("_fb_ind")
        n_pit = expanded.height
        if n_pit < len(norm_dates) * len(relevant_codes):
            _warn_pit_industry_unavailable(
                "PIT 数据仅部分覆盖，缺口用 stocks.industry 按代码补齐"
            )

    return long_ind.filter(pl.col("industry").is_not_null())


def compute_exposures(
    daily_data: pl.DataFrame,
    daily_basic: pl.DataFrame,
    stocks: pl.DataFrame,
    trade_date: str | object,
    style_registry: dict[str, Callable] | None = None,
    style_names: list[str] | None = None,
    *,
    style_panel: pl.DataFrame | None = None,
    industry_names: list[str] | None = None,
    industry_panel: pl.DataFrame | None = None,
) -> ExposureMatrix:
    """计算指定日期的因子暴露矩阵。

    流程：
    1. 风格：优先从预物化 ``style_panel`` 切片；否则对全输入算因子（兼容旧调用）
    2. 行业：优先从预物化 ``industry_panel`` 切片；否则当日 PIT + 可选固定 ``industry_names``
    3. 合并为暴露矩阵

    Args:
        daily_data / daily_basic / stocks: 同前。
        trade_date: 目标日期。
        style_registry / style_names: 风格因子集。
        style_panel: 预物化风格面板（W1）；含 trade_date, ts_code, style 列。
        industry_names: 固定行业列全集（裸名或 ind_ 前缀）；W2 稳定化。
        industry_panel: 预物化行业哑变量面板（W1/W2）。

    Returns:
        ExposureMatrix。
    """
    registry = STYLE_FACTOR_REGISTRY if style_registry is None else style_registry
    names = STYLE_FACTOR_NAMES if style_names is None else style_names
    target_date = _parse_trade_date(trade_date)

    # ── 1. 风格因子 ──────────────────────────────────────────────────────────
    merged: pl.DataFrame | None = None
    style_col_list: list[str] = []

    if style_panel is not None and not style_panel.is_empty():
        day_style = style_panel.filter(pl.col("trade_date") == target_date)
        style_col_list = [c for c in names if c in day_style.columns]
        if not day_style.is_empty() and style_col_list:
            merged = day_style.select(["ts_code", *style_col_list])

    if merged is None:
        style_dfs = _compute_style_day(daily_data, daily_basic, registry, names, target_date)
        if not style_dfs:
            logger.warning(f"日期 {trade_date} 无可用风格因子")
            return ExposureMatrix()
        style_col_list = list(style_dfs.keys())
        merged = style_dfs[style_col_list[0]]
        for name in style_col_list[1:]:
            merged = merged.join(style_dfs[name], on="ts_code", how="full", coalesce=True)

    # ── 2. 行业哑变量 ────────────────────────────────────────────────────────
    ind_col_names: list[str] = []

    if industry_panel is not None and not industry_panel.is_empty():
        day_ind = industry_panel.filter(pl.col("trade_date") == target_date)
        ind_col_names = (
            [c for c in industry_panel.columns if c.startswith("ind_")]
            if industry_names is None
            else _normalize_ind_cols(industry_names)
        )
        # 确保缺列填 0
        for c in ind_col_names:
            if c not in day_ind.columns:
                day_ind = day_ind.with_columns(pl.lit(0.0).alias(c))
        if not day_ind.is_empty() and ind_col_names:
            merged = merged.join(
                day_ind.select(["ts_code", *ind_col_names]),
                on="ts_code",
                how="left",
            )
            merged = merged.with_columns(
                [pl.col(c).fill_null(0.0) for c in ind_col_names]
            )
    else:
        fallback_ind = (
            stocks.select(["ts_code", "industry"]).unique(subset=["ts_code"])
            if "industry" in stocks.columns
            else None
        )
        relevant_codes = stocks["ts_code"].unique().to_list()
        stock_ind = _resolve_stock_industry(target_date, relevant_codes, fallback_ind)

        if stock_ind is not None and not stock_ind.is_empty():
            if industry_names is not None:
                bare = [n[4:] if n.startswith("ind_") else n for n in industry_names]
                industries = sorted(set(bare))
            else:
                industries = None  # 当日出现的行业
            ind_dummies = get_industry_dummies(
                stock_ind, industry_col="industry", industries=industries
            )
            if not ind_dummies.is_empty():
                ind_col_names = [c for c in ind_dummies.columns if c.startswith("ind_")]
                # 若指定了 industry_names，对齐列序与全集
                if industry_names is not None:
                    want = _normalize_ind_cols(industry_names)
                    for c in want:
                        if c not in ind_dummies.columns:
                            ind_dummies = ind_dummies.with_columns(pl.lit(0.0).alias(c))
                    ind_col_names = want
                merged = merged.join(
                    ind_dummies.select(["ts_code", *ind_col_names]),
                    on="ts_code",
                    how="left",
                )
                merged = merged.with_columns(
                    [pl.col(c).fill_null(0.0) for c in ind_col_names]
                )

    # ── 3. 构建暴露矩阵 ──────────────────────────────────────────────────────
    # 至少第一个风格因子不为空
    first_style = style_col_list[0]
    merged = merged.drop_nulls(subset=[first_style])

    for name in style_col_list:
        merged = merged.with_columns(pl.col(name).fill_null(0.0))

    codes = merged["ts_code"].to_list()
    factor_names = style_col_list + ind_col_names

    matrix_cols = [merged[c].to_numpy().astype(np.float64) for c in factor_names]
    if matrix_cols:
        matrix = np.column_stack(matrix_cols)
    else:
        matrix = np.empty((len(codes), 0))

    return ExposureMatrix(codes=codes, factor_names=factor_names, matrix=matrix)


def reindex_exposure(
    exposure: ExposureMatrix,
    factor_names: list[str],
) -> ExposureMatrix:
    """将暴露矩阵列重排/补齐到固定 ``factor_names``（缺列填 0，多余列丢弃）。

    用于 build 内保证逐日回归系数与全局因子名对齐。
    """
    if exposure.n_stocks == 0:
        return ExposureMatrix(
            codes=[],
            factor_names=list(factor_names),
            matrix=np.empty((0, len(factor_names))),
        )
    name_to_idx = {n: i for i, n in enumerate(exposure.factor_names)}
    cols = []
    for n in factor_names:
        if n in name_to_idx:
            cols.append(exposure.matrix[:, name_to_idx[n]])
        else:
            cols.append(np.zeros(exposure.n_stocks, dtype=np.float64))
    matrix = np.column_stack(cols) if cols else np.empty((exposure.n_stocks, 0))
    return ExposureMatrix(
        codes=list(exposure.codes),
        factor_names=list(factor_names),
        matrix=matrix,
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_trade_date(trade_date: str | object) -> dt.date:
    if isinstance(trade_date, str):
        if "-" in trade_date:
            return dt.date.fromisoformat(trade_date)
        return dt.datetime.strptime(trade_date, "%Y%m%d").date()
    if isinstance(trade_date, dt.datetime):
        return trade_date.date()
    if isinstance(trade_date, dt.date):
        return trade_date
    raise TypeError(f"不支持的日期类型: {type(trade_date)}")


def _normalize_ind_cols(industry_names: list[str]) -> list[str]:
    out = []
    for n in industry_names:
        out.append(n if n.startswith("ind_") else f"ind_{n}")
    return out


def _compute_style_day(
    daily_data: pl.DataFrame,
    daily_basic: pl.DataFrame,
    registry: dict[str, Callable],
    names: list[str],
    target_date: dt.date,
) -> dict[str, pl.DataFrame]:
    """兼容路径：逐因子全窗计算 + 标准化 + filter 当日（旧逻辑，单日调用用）。"""
    style_dfs: dict[str, pl.DataFrame] = {}
    for name in names:
        fn = registry[name]
        try:
            factor_df = fn(daily_data, daily_basic)
            if factor_df.is_empty():
                logger.warning(f"因子 {name} 在 {target_date} 计算结果为空")
                continue
            factor_df = cs_standardize(factor_df, "factor_value", method="mad")
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
    return style_dfs


def _resolve_stock_industry(
    target_date: dt.date,
    relevant_codes: list[str],
    fallback_ind: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """PIT 行业归属；不可用或覆盖不全时用 stocks.industry 补齐。"""
    pit_ind = _lookup_pit_industry(target_date)
    if pit_ind is not None:
        pit_ind = pit_ind.filter(pl.col("ts_code").is_in(relevant_codes))

    if pit_ind is not None and not pit_ind.is_empty():
        covered = set(pit_ind["ts_code"].to_list())
        if fallback_ind is not None and len(covered) < len(relevant_codes):
            gap = fallback_ind.filter(~pl.col("ts_code").is_in(covered))
            if not gap.is_empty():
                _warn_pit_industry_unavailable(
                    f"PIT 数据仅覆盖 {len(covered)}/{len(relevant_codes)} 只股票，"
                    "其余用 stocks.industry 按代码补齐"
                )
                pit_ind = pl.concat([pit_ind, gap], how="vertical_relaxed")
        return pit_ind

    if fallback_ind is not None:
        if pit_ind is not None:
            _warn_pit_industry_unavailable("PIT 数据对本次请求的股票代码无覆盖")
        return fallback_ind
    return None


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

    任何原因不可用均优雅降级，返回 ``None``。
    """
    try:
        membership = fetch_index_member_all()
    except Exception as e:
        _warn_pit_industry_unavailable(f"获取异常: {e}")
        return None

    if membership is None or membership.is_empty():
        _warn_pit_industry_unavailable("数据源不可用或为空")
        return None

    required_cols = {"ts_code", "l1_name", "in_date", "out_date"}
    if not required_cols.issubset(set(membership.columns)):
        _warn_pit_industry_unavailable(f"字段缺失，需要 {sorted(required_cols)}")
        return None

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
