"""归因分析：Brinson BHB 收益归因 与 Barra 风格归因。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm

from config.constants import TRADING_DAYS_PER_YEAR

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Brinson BHB 收益归因
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BrinsonResult:
    """Brinson-Hood-Beebower 三因素归因结果。

    Attributes:
        sector_df: 行业维度汇总 DataFrame (sector, allocation, selection, interaction,
                   total_contribution)，每列为各期之和。
        period_df: 时间维度 DataFrame (trade_date, allocation, selection, interaction,
                   active_ret)，各列为当期所有行业之和。
        ann_allocation: 年化配置效应
        ann_selection: 年化选股效应
        ann_interaction: 年化交互效应
        ann_active_return: 年化超额收益 (= ann_allocation + ann_selection + ann_interaction)
    """

    sector_df: pl.DataFrame
    period_df: pl.DataFrame
    ann_allocation: float
    ann_selection: float
    ann_interaction: float
    ann_active_return: float


def brinson_attribution(
    portfolio_sector_weights: pl.DataFrame,
    benchmark_sector_weights: pl.DataFrame,
    sector_returns: pl.DataFrame,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BrinsonResult:
    """Brinson-Hood-Beebower 三因素收益归因。

    将组合超额收益分解为配置效应、选股效应和交互效应：
        allocation_i  = (w_p,i - w_b,i) * r_b,i
        selection_i   = w_b,i * (r_p,i - r_b,i)
        interaction_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)
        active_ret    = Σ(allocation_i + selection_i + interaction_i)

    Args:
        portfolio_sector_weights: 含 trade_date, sector, port_weight 列的 DataFrame。
        benchmark_sector_weights: 含 trade_date, sector, bench_weight 列的 DataFrame。
        sector_returns: 含 trade_date, sector, port_ret, bench_ret 列的 DataFrame。
        trading_days_per_year: 年化因子，默认 252。

    Returns:
        BrinsonResult
    """
    # ── 1. 合并权重与收益 ──────────────────────────────────────────────────────
    merged = (
        sector_returns.join(
            portfolio_sector_weights.select(["trade_date", "sector", "port_weight"]),
            on=["trade_date", "sector"],
            how="left",
        )
        .join(
            benchmark_sector_weights.select(["trade_date", "sector", "bench_weight"]),
            on=["trade_date", "sector"],
            how="left",
        )
        .with_columns(
            [
                pl.col("port_weight").fill_null(0.0).cast(pl.Float64),
                pl.col("bench_weight").fill_null(0.0).cast(pl.Float64),
                pl.col("port_ret").cast(pl.Float64),
                pl.col("bench_ret").cast(pl.Float64),
            ]
        )
    )

    # ── 2. 计算三项效应（行级别） ────────────────────────────────────────────
    merged = merged.with_columns(
        [
            ((pl.col("port_weight") - pl.col("bench_weight")) * pl.col("bench_ret")).alias(
                "allocation"
            ),
            (pl.col("bench_weight") * (pl.col("port_ret") - pl.col("bench_ret"))).alias(
                "selection"
            ),
            (
                (pl.col("port_weight") - pl.col("bench_weight"))
                * (pl.col("port_ret") - pl.col("bench_ret"))
            ).alias("interaction"),
        ]
    ).with_columns(
        (pl.col("allocation") + pl.col("selection") + pl.col("interaction")).alias(
            "total_contribution"
        )
    )

    # ── 3. 行业维度汇总（各行业对所有期的贡献之和）─────────────────────────────
    sector_df = (
        merged.group_by("sector")
        .agg(
            [
                pl.col("allocation").sum(),
                pl.col("selection").sum(),
                pl.col("interaction").sum(),
                pl.col("total_contribution").sum(),
            ]
        )
        .sort("sector")
    )

    # ── 4. 时间维度汇总（每期所有行业之和）──────────────────────────────────────
    period_df = (
        merged.group_by("trade_date")
        .agg(
            [
                pl.col("allocation").sum(),
                pl.col("selection").sum(),
                pl.col("interaction").sum(),
                pl.col("total_contribution").sum().alias("active_ret"),
            ]
        )
        .sort("trade_date")
    )

    # ── 5. 年化指标（均值 × 年化因子）─────────────────────────────────────────
    alloc_arr = period_df["allocation"].to_numpy()
    select_arr = period_df["selection"].to_numpy()
    interact_arr = period_df["interaction"].to_numpy()

    ann_allocation = float(np.mean(alloc_arr) * trading_days_per_year)
    ann_selection = float(np.mean(select_arr) * trading_days_per_year)
    ann_interaction = float(np.mean(interact_arr) * trading_days_per_year)
    ann_active_return = ann_allocation + ann_selection + ann_interaction

    return BrinsonResult(
        sector_df=sector_df,
        period_df=period_df,
        ann_allocation=ann_allocation,
        ann_selection=ann_selection,
        ann_interaction=ann_interaction,
        ann_active_return=ann_active_return,
    )


def aggregate_positions_to_sectors(
    positions: pl.DataFrame,
    sector_map: dict[str, str],
) -> pl.DataFrame:
    """将个股持仓聚合到行业层面，并归一化权重使每日之和为 1。

    Args:
        positions: 含 trade_date, ts_code, weight 列的 DataFrame。
        sector_map: {ts_code: sector_name} 映射字典。

    Returns:
        含 trade_date, sector, weight 列的 DataFrame，每日 weight 之和为 1。
    """
    # 将 sector_map 转换为 DataFrame 用于 join
    codes = list(sector_map.keys())
    sectors = [sector_map[c] for c in codes]
    sector_lut = pl.DataFrame({"ts_code": codes, "sector": sectors})

    merged = (
        positions.with_columns(pl.col("weight").cast(pl.Float64))
        .join(sector_lut, on="ts_code", how="left")
        .filter(pl.col("sector").is_not_null())
    )

    # 按 (trade_date, sector) 汇总权重
    agg = (
        merged.group_by(["trade_date", "sector"])
        .agg(pl.col("weight").sum().alias("weight_raw"))
    )

    # 每日权重归一化到总和为 1
    daily_total = agg.group_by("trade_date").agg(
        pl.col("weight_raw").sum().alias("total_weight")
    )

    result = (
        agg.join(daily_total, on="trade_date", how="left")
        .with_columns(
            (pl.col("weight_raw") / pl.col("total_weight")).alias("weight")
        )
        .select(["trade_date", "sector", "weight"])
        .sort(["trade_date", "sector"])
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Barra 风格归因（时序 OLS）
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BarraStyleResult:
    """Barra 风格归因结果。

    Attributes:
        exposures: {style_name: beta 系数}
        contributions: {style_name: beta * mean(style_ret)}
        alpha: 截距（已年化：intercept × trading_days_per_year）
        r_squared: OLS 拟合优度 R²
        residual_series: 残差序列 DataFrame (trade_date, residual)
    """

    exposures: dict[str, float]
    contributions: dict[str, float]
    alpha: float
    r_squared: float
    residual_series: pl.DataFrame


def barra_style_attribution(
    portfolio_excess_returns: pl.DataFrame,
    style_factor_returns: pl.DataFrame,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BarraStyleResult:
    """Barra 风格归因：时序 OLS 回归将组合超额收益分解到风格因子上。

    回归模型：
        excess_ret_t = α + Σ β_s × style_ret_s,t + ε_t

    风格贡献：
        contribution_s = β_s × mean(style_ret_s)

    Args:
        portfolio_excess_returns: 含 trade_date, excess_ret 列的 DataFrame。
        style_factor_returns: 含 trade_date + 各风格因子列的 DataFrame。
        trading_days_per_year: 年化因子，默认 252。

    Returns:
        BarraStyleResult
    """
    # ── 1. 识别风格列 ──────────────────────────────────────────────────────────
    style_cols = [c for c in style_factor_returns.columns if c != "trade_date"]

    # ── 2. 合并并去掉 NaN ─────────────────────────────────────────────────────
    joined = portfolio_excess_returns.join(
        style_factor_returns,
        on="trade_date",
        how="inner",
    ).drop_nulls()

    if joined.is_empty():
        raise ValueError("barra_style_attribution: 合并后无有效数据")

    joined = joined.sort("trade_date")

    # ── 3. 构建设计矩阵与因变量 ──────────────────────────────────────────────
    y: np.ndarray = joined["excess_ret"].to_numpy().astype(float)
    style_arrays = [joined[c].to_numpy().astype(float) for c in style_cols]
    X_raw = np.column_stack(style_arrays) if style_arrays else np.empty((len(y), 0))
    X = sm.add_constant(X_raw, prepend=True)  # 第 0 列为截距

    # ── 4. OLS 回归 ──────────────────────────────────────────────────────────
    model = sm.OLS(y, X).fit()
    params: np.ndarray = model.params

    intercept_raw: float = float(params[0])
    beta_arr: np.ndarray = params[1:]

    # ── 5. 构建结果 ──────────────────────────────────────────────────────────
    exposures: dict[str, float] = {
        style_cols[i]: float(beta_arr[i]) for i in range(len(style_cols))
    }

    contributions: dict[str, float] = {
        style_cols[i]: float(beta_arr[i]) * float(np.mean(style_arrays[i]))
        for i in range(len(style_cols))
    }

    # 截距年化
    alpha = intercept_raw * trading_days_per_year

    # R²
    r_squared = float(model.rsquared)

    # 残差序列
    residuals_arr: np.ndarray = y - model.predict(X)
    residual_series = joined.select("trade_date").with_columns(
        pl.Series("residual", residuals_arr)
    )

    return BarraStyleResult(
        exposures=exposures,
        contributions=contributions,
        alpha=alpha,
        r_squared=r_squared,
        residual_series=residual_series,
    )
