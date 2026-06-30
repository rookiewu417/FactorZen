"""Barra 风格因子计算：8 个经典风格因子 + 截面标准化。

每个因子函数签名统一为:
    factor_fn(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame

返回 DataFrame 含 [trade_date, ts_code, factor_value] 三列。

Style factors:
1. Size       — ln(total_mv)
2. Value      — 1/pb (Book-to-Price)
3. Momentum   — 252 日累计收益（跳过最近 21 日）
4. Volatility — 60 日收益率标准差
5. Liquidity  — ln(20 日换手率均值)
6. Quality    — ROE 近似 = pb / pe_ttm
7. Growth     — 盈利增长近似 = pe_ttm 倒数变化率
8. Leverage   — 杠杆近似 = pb - 1（净资产乘数代理）
"""

from __future__ import annotations

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 截面标准化
# ═══════════════════════════════════════════════════════════════════════════════


def cs_standardize(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    method: str = "mad",
) -> pl.DataFrame:
    """截面 Winsorize + Z-score 标准化。

    每个 trade_date 内独立执行：
    1. 计算中位数 med 与 MAD（Median Absolute Deviation）
    2. Winsorize：将超出 [med - 3*1.4826*MAD, med + 3*1.4826*MAD] 的值截断
    3. Z-score：减去均值，除以标准差

    Args:
        df: 含 trade_date 和 factor_col 列的 DataFrame。
        factor_col: 因子值列名。
        method: 截断方法，目前仅支持 "mad"。

    Returns:
        同结构 DataFrame，factor_col 已被标准化值替换。
    """
    if method != "mad":
        raise ValueError(f"不支持的标准化方法: {method}，目前仅支持 'mad'")

    # MAD Winsorize + Z-score，按 trade_date 分组
    result = df.with_columns(
        pl.col(factor_col).cast(pl.Float64)
    ).with_columns(
        # 计算截面中位数和 MAD
        pl.col(factor_col).median().over("trade_date").alias("_cs_median"),
        (pl.col(factor_col) - pl.col(factor_col).median().over("trade_date"))
        .abs()
        .median()
        .over("trade_date")
        .alias("_cs_mad"),
    ).with_columns(
        # 1.4826 * MAD ≈ σ（正态分布下）
        (pl.col("_cs_mad") * 1.4826).alias("_cs_mad_scaled"),
    ).with_columns(
        # Winsorize: 截断到 [median - 3σ, median + 3σ]
        pl.col(factor_col)
        .clip(
            pl.col("_cs_median") - 3.0 * pl.col("_cs_mad_scaled"),
            pl.col("_cs_median") + 3.0 * pl.col("_cs_mad_scaled"),
        )
        .alias(factor_col),
    ).with_columns(
        # Z-score
        pl.col(factor_col).mean().over("trade_date").alias("_cs_mean"),
        pl.col(factor_col).std().over("trade_date").alias("_cs_std"),
    ).with_columns(
        pl.when(pl.col("_cs_std") > 1e-12)
        .then((pl.col(factor_col) - pl.col("_cs_mean")) / pl.col("_cs_std"))
        .otherwise(pl.lit(0.0))
        .alias(factor_col),
    ).drop(["_cs_median", "_cs_mad", "_cs_mad_scaled", "_cs_mean", "_cs_std"])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 8 个风格因子
# ═══════════════════════════════════════════════════════════════════════════════


def factor_size(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Size 因子：ln(total_mv)。

    Args:
        daily_data: 日线行情 DataFrame。
        daily_basic: 每日估值指标 DataFrame，需含 total_mv 列。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    return (
        daily_basic.select(["trade_date", "ts_code", "total_mv"])
        .filter(pl.col("total_mv").is_not_null() & (pl.col("total_mv") > 0))
        .with_columns(pl.col("total_mv").log().alias("factor_value"))
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_value(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Value 因子：1/pb（Book-to-Price）。

    Args:
        daily_data: 日线行情 DataFrame。
        daily_basic: 每日估值指标 DataFrame，需含 pb 列。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    return (
        daily_basic.select(["trade_date", "ts_code", "pb"])
        .filter(pl.col("pb").is_not_null() & (pl.col("pb").abs() > 1e-8))
        .with_columns((1.0 / pl.col("pb")).alias("factor_value"))
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_momentum(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Momentum 因子：过去 252 日累计收益（跳过最近 21 日）。

    计算 pct_chg 的滚动窗口收益：
    - 窗口 = [t-252, t-22] 的 231 日累计收益
    - 使用对数收益率累加后取 exp

    Args:
        daily_data: 日线行情 DataFrame，需含 pct_chg 列。
        daily_basic: 每日估值指标 DataFrame（此因子不使用）。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    # pct_chg 为百分比形式（如 2.5 表示 2.5%），转为小数
    df = (
        daily_data.select(["trade_date", "ts_code", "pct_chg"])
        .filter(pl.col("pct_chg").is_not_null())
        .sort(["ts_code", "trade_date"])
        .with_columns((pl.col("pct_chg") / 100.0).alias("ret"))
    )

    # 对数收益率
    df = df.with_columns(
        (1.0 + pl.col("ret")).log().alias("log_ret")
    )

    # 252 日累计对数收益 与 21 日累计对数收益
    df = df.with_columns(
        pl.col("log_ret")
        .rolling_sum(window_size=252, min_periods=252)
        .over("ts_code")
        .alias("cum_252"),
        pl.col("log_ret")
        .rolling_sum(window_size=21, min_periods=21)
        .over("ts_code")
        .alias("cum_21"),
    )

    # momentum = cum_252 - cum_21（跳过最近 21 日）
    df = df.with_columns(
        (pl.col("cum_252") - pl.col("cum_21")).alias("factor_value")
    )

    return (
        df.filter(pl.col("factor_value").is_not_null())
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_volatility(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Volatility 因子：60 日日收益率标准差。

    Args:
        daily_data: 日线行情 DataFrame，需含 pct_chg 列。
        daily_basic: 每日估值指标 DataFrame（此因子不使用）。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    df = (
        daily_data.select(["trade_date", "ts_code", "pct_chg"])
        .filter(pl.col("pct_chg").is_not_null())
        .sort(["ts_code", "trade_date"])
        .with_columns((pl.col("pct_chg") / 100.0).alias("ret"))
    )

    df = df.with_columns(
        pl.col("ret")
        .rolling_std(window_size=60, min_periods=60)
        .over("ts_code")
        .alias("factor_value")
    )

    return (
        df.filter(pl.col("factor_value").is_not_null())
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_liquidity(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Liquidity 因子：ln(20 日平均换手率)。

    Args:
        daily_data: 日线行情 DataFrame（此因子不使用）。
        daily_basic: 每日估值指标 DataFrame，需含 turnover_rate 列。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    # turnover_rate 可能来自 daily_basic；若无，则尝试从 daily_data 近似
    if "turnover_rate" not in daily_basic.columns:
        logger.warning("daily_basic 中无 turnover_rate 列，Liquidity 因子将为空")
        return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64})

    df = (
        daily_basic.select(["trade_date", "ts_code", "turnover_rate"])
        .filter(pl.col("turnover_rate").is_not_null() & (pl.col("turnover_rate") > 0))
        .sort(["ts_code", "trade_date"])
    )

    df = df.with_columns(
        pl.col("turnover_rate")
        .rolling_mean(window_size=20, min_periods=20)
        .over("ts_code")
        .alias("avg_turnover")
    )

    df = df.with_columns(
        pl.col("avg_turnover").log().alias("factor_value")
    )

    return (
        df.filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_quality(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Quality 因子：ROE 近似 = pb / pe_ttm。

    ROE_ttm ≈ EPS_ttm / BPS = (Price/PE_ttm) / (Price/PB) = PB / PE_ttm

    Args:
        daily_data: 日线行情 DataFrame（此因子不使用）。
        daily_basic: 每日估值指标 DataFrame，需含 pb 和 pe_ttm 列。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    return (
        daily_basic.select(["trade_date", "ts_code", "pb", "pe_ttm"])
        .filter(
            pl.col("pb").is_not_null()
            & pl.col("pe_ttm").is_not_null()
            & (pl.col("pe_ttm").abs() > 1e-8)
        )
        .with_columns((pl.col("pb") / pl.col("pe_ttm")).alias("factor_value"))
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_growth(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Growth 因子：盈利增长近似 = 1/pe_ttm 的同比变化率。

    E/P_ttm ≈ 盈利/价格，同比变化近似盈利增长。
    growth = (ep_t - ep_{t-252}) / |ep_{t-252}|

    Args:
        daily_data: 日线行情 DataFrame（此因子不使用）。
        daily_basic: 每日估值指标 DataFrame，需含 pe_ttm 列。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    df = (
        daily_basic.select(["trade_date", "ts_code", "pe_ttm"])
        .filter(pl.col("pe_ttm").is_not_null() & (pl.col("pe_ttm").abs() > 1e-8))
        .sort(["ts_code", "trade_date"])
        .with_columns((1.0 / pl.col("pe_ttm")).alias("ep"))
    )

    # 252 日前的 ep
    df = df.with_columns(
        pl.col("ep").shift(252).over("ts_code").alias("ep_lag")
    )

    df = df.with_columns(
        pl.when(pl.col("ep_lag").abs() > 1e-8)
        .then((pl.col("ep") - pl.col("ep_lag")) / pl.col("ep_lag").abs())
        .otherwise(None)
        .alias("factor_value")
    )

    return (
        df.filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
        .select(["trade_date", "ts_code", "factor_value"])
    )


def factor_leverage(daily_data: pl.DataFrame, daily_basic: pl.DataFrame) -> pl.DataFrame:
    """Leverage 因子：杠杆近似 = pb - 1。

    PB = 市值/净资产，PB - 1 = (市值 - 净资产)/净资产 ≈ 负债/净资产。
    值越大表示杠杆越高。

    Args:
        daily_data: 日线行情 DataFrame（此因子不使用）。
        daily_basic: 每日估值指标 DataFrame，需含 pb 列。

    Returns:
        含 [trade_date, ts_code, factor_value] 的 DataFrame。
    """
    return (
        daily_basic.select(["trade_date", "ts_code", "pb"])
        .filter(pl.col("pb").is_not_null())
        .with_columns((pl.col("pb") - 1.0).alias("factor_value"))
        .select(["trade_date", "ts_code", "factor_value"])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 因子注册
# ═══════════════════════════════════════════════════════════════════════════════

STYLE_FACTOR_NAMES: list[str] = [
    "size",
    "value",
    "momentum",
    "volatility",
    "liquidity",
    "quality",
    "growth",
    "leverage",
]

STYLE_FACTOR_REGISTRY: dict[str, callable] = {
    "size": factor_size,
    "value": factor_value,
    "momentum": factor_momentum,
    "volatility": factor_volatility,
    "liquidity": factor_liquidity,
    "quality": factor_quality,
    "growth": factor_growth,
    "leverage": factor_leverage,
}
