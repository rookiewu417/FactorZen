"""Benchmark 对比：计算策略相对基准的超额表现。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from config.constants import BENCHMARK_INDICES, TRADING_DAYS_PER_YEAR


@dataclass
class BenchmarkResult:
    """策略相对基准的超额表现。"""

    benchmark_code: str
    benchmark_name: str
    # 日度序列：trade_date, strategy_ret, benchmark_ret, excess_ret, strategy_nav, benchmark_nav, excess_nav
    daily: pl.DataFrame
    ann_excess_ret: float  # 年化超额收益
    tracking_error: float  # 年化跟踪误差（超额收益的年化标准差）
    information_ratio: float  # IR = ann_excess_ret / tracking_error
    excess_max_dd: float  # 超额净值的最大回撤（负数或 0）

    def summary(self) -> str:
        name = self.benchmark_name
        return (
            f"vs {name}: 超额={self.ann_excess_ret:.2%} "
            f"IR={self.information_ratio:.2f} "
            f"TE={self.tracking_error:.2%} "
            f"超额回撤={self.excess_max_dd:.1%}"
        )


def compute_excess_return(
    strategy_nav: pl.DataFrame,
    benchmark_code: str,
    start: str,
    end: str,
) -> BenchmarkResult:
    """计算策略相对指数的超额表现。

    Args:
        strategy_nav: 含 trade_date, net_return(或 ret), nav 列的 DataFrame。
                      trade_date 可以是 pl.Date 或 str("YYYY-MM-DD"/"YYYYMMDD")。
        benchmark_code: 指数代码，如 "000300.SH"。
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。

    Returns:
        BenchmarkResult
    """
    from common.loader import fetch_index_daily

    # ── 1. 读取基准数据 ───────────────────────────────────────────────────────────
    index_df = fetch_index_daily(benchmark_code, start, end)
    if index_df.is_empty():
        raise ValueError(f"基准 {benchmark_code} 数据不足")

    index_df = (
        index_df.select(["trade_date", "close"])
        .sort("trade_date")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("benchmark_ret")
        )
        .drop_nulls("benchmark_ret")
    )

    # ── 2. 对齐策略收益 ──────────────────────────────────────────────────────────
    # 确保 trade_date 是 pl.Date 类型
    strat = _ensure_date(strategy_nav, "trade_date")

    # 兼容 net_return / ret 两种列名
    if "net_return" in strat.columns:
        ret_col = "net_return"
    elif "ret" in strat.columns:
        ret_col = "ret"
    else:
        raise ValueError("strategy_nav 缺少 net_return 或 ret 列")

    strat = strat.select(["trade_date", pl.col(ret_col).alias("strategy_ret")])

    # inner join 对齐，去掉 NaN
    joined = (
        strat.join(
            index_df.select(["trade_date", "benchmark_ret"]),
            on="trade_date",
            how="inner",
        )
        .drop_nulls(["strategy_ret", "benchmark_ret"])
        .sort("trade_date")
    )

    if joined.height < 2:
        raise ValueError(f"基准 {benchmark_code} 数据不足")

    # ── 3. 计算超额净值 ──────────────────────────────────────────────────────────
    joined = joined.with_columns(
        (pl.col("strategy_ret") - pl.col("benchmark_ret")).alias("excess_ret")
    ).with_columns(
        [
            (pl.col("strategy_ret") + 1).cum_prod().alias("strategy_nav"),
            (pl.col("benchmark_ret") + 1).cum_prod().alias("benchmark_nav"),
            (pl.col("excess_ret") + 1).cum_prod().alias("excess_nav"),
        ]
    )

    # ── 4. 统计指标 ──────────────────────────────────────────────────────────────
    excess_arr: np.ndarray = joined["excess_ret"].to_numpy()
    excess_nav_arr: np.ndarray = joined["excess_nav"].to_numpy()

    ann_excess_ret = float(np.mean(excess_arr) * TRADING_DAYS_PER_YEAR)
    tracking_error = float(np.std(excess_arr) * np.sqrt(TRADING_DAYS_PER_YEAR))
    information_ratio = ann_excess_ret / tracking_error if tracking_error > 1e-8 else 0.0

    nav_with_base = np.concatenate([[1.0], excess_nav_arr])
    cum_max = np.maximum.accumulate(nav_with_base)
    drawdowns = nav_with_base / cum_max - 1.0
    excess_max_dd = float(np.min(drawdowns))

    # ── 5. benchmark_name ────────────────────────────────────────────────────────
    benchmark_name = BENCHMARK_INDICES.get(benchmark_code, benchmark_code)

    daily_df = joined.select(
        [
            "trade_date",
            "strategy_ret",
            "benchmark_ret",
            "excess_ret",
            "strategy_nav",
            "benchmark_nav",
            "excess_nav",
        ]
    )

    return BenchmarkResult(
        benchmark_code=benchmark_code,
        benchmark_name=benchmark_name,
        daily=daily_df,
        ann_excess_ret=ann_excess_ret,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        excess_max_dd=excess_max_dd,
    )


def _ensure_date(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """将 trade_date 列统一转换为 pl.Date 类型。"""
    dtype = df.schema[col]
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(col).dt.date().alias(col))
    if dtype == pl.Utf8:
        parsed_dash = pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        parsed_plain = pl.col(col).str.strptime(pl.Date, "%Y%m%d", strict=False)
        return df.with_columns(parsed_dash.fill_null(parsed_plain).alias(col))
    return df
