"""分层回测。按因子值分组，计算各组收益与多空对冲表现。"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from config.constants import (
    BORROW_RATE_ANNUAL,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    TRADING_DAYS_PER_YEAR,
)


@dataclass
class CostModel:
    """A 股交易成本模型。

    所有费率均为小数（非百分比），例如 commission=0.00025 表示万 2.5。
    """

    commission: float = COMMISSION_RATE  # 单边佣金
    stamp_tax: float = STAMP_TAX_RATE  # 卖出印花税（仅卖出）
    slippage: float = SLIPPAGE_RATE  # 单边冲击成本/滑点
    borrow_annual: float = BORROW_RATE_ANNUAL  # 融券年化利率（做空时）

    def one_way_cost(self) -> float:
        """单边成本率（买入或卖出各自的成本）。"""
        return self.commission + self.slippage

    def round_trip_cost(self) -> float:
        """一次完整换手（卖旧 + 买新）的成本率。"""
        sell = self.commission + self.slippage + self.stamp_tax
        buy = self.commission + self.slippage
        return sell + buy

    def borrow_rate_per_period(self, frequency: str = "daily") -> float:
        """融券日/周/月费率（按年化利率折算）。"""
        days = {"daily": 1, "weekly": 5, "monthly": 21}.get(frequency, 1)
        return self.borrow_annual * days / TRADING_DAYS_PER_YEAR


@dataclass
class BacktestResult:
    factor_name: str
    n_groups: int
    daily_returns: pl.DataFrame  # trade_date, group, ret
    nav: pl.DataFrame  # trade_date, group, nav (累计净值)
    long_short_nav: pl.DataFrame  # trade_date, nav
    summary_stats: dict[int | str, dict[str, float]]  # {group: {ann_ret, ann_vol, sharpe, max_dd}}
    frequency: str = "daily"
    ret_definition: str = "fwd_ret_1d"  # 记录输入收益类型，防止误用同日 ret

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(
            self.frequency, self.frequency
        )
        lines = []
        int_groups = sorted((k, v) for k, v in self.summary_stats.items() if isinstance(k, int))
        for g, stats in int_groups:
            lines.append(f"  G{g}: ret={stats['ann_ret']:.2%} Sharpe={stats['sharpe']:.2f}")
        if "long_short" in self.summary_stats:
            stats = self.summary_stats["long_short"]
            lines.append(f"  Long-Short: Sharpe={stats['sharpe']:.2f} MaxDD={stats['max_dd']:.1%}")
        return f"Backtest ({self.n_groups} groups, {freq_label}):\n" + "\n".join(lines)


def run_stratified_backtest(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    frequency: str = "daily",
    factor_name: str = "",
    cost_model: "CostModel | None" = None,
) -> BacktestResult:
    """分层回测。

    Args:
        factor_df: 因子值，列: trade_date, ts_code, {factor_col}
        daily_ret: **前向收益**，列: trade_date, ts_code, ret
            ret 必须是 t 日因子可见时能实现的下一期收益（如 fwd_ret_1d），
            不能是 t 日 close-to-close ret，否则引入同日 look-ahead bias。
        factor_col: 使用的因子列
        n_groups: 分组数
        cost_model: 成本模型。为 None 时不扣成本（兼容旧调用）。
    """
    # 合并因子和收益，过滤无效收益
    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner").filter(
        pl.col("ret").is_not_null() & pl.col("ret").is_finite()
    )

    # 每日截面分组 (qcut)
    merged = (
        merged.with_columns(
            pl.col(factor_col).rank("ordinal", descending=False).over("trade_date").alias("_rank")
        )
        .with_columns(
            ((pl.col("_rank") - 1) * n_groups // pl.col("_rank").max().over("trade_date"))
            .cast(pl.Int32)
            .alias("group")
        )
        .drop("_rank")
    )

    # ── 换手率计算（仅在启用成本模型时）──────────────────────────────────
    turnover_by_date: pl.DataFrame | None = None
    if cost_model is not None:
        # 按 (ts_code, trade_date) 排序，获取上期组别
        merged = merged.sort(["ts_code", "trade_date"])
        merged = merged.with_columns(pl.col("group").shift(1).over("ts_code").alias("_prev_group"))
        # 换手 = 本期组别与上期不同（或首次出现）的比例
        merged = merged.with_columns(
            (pl.col("group").ne(pl.col("_prev_group")).fill_null(True)).alias("_changed")
        )
        turnover_by_date = (
            merged.group_by("trade_date")
            .agg(pl.col("_changed").mean().alias("turnover"))
            .sort("trade_date")
        )
        merged = merged.drop(["_prev_group", "_changed"])

    # 组日收益（等权）
    group_ret = (
        merged.group_by(["trade_date", "group"])
        .agg(pl.col("ret").mean().alias("ret"))
        .sort(["trade_date", "group"])
    )

    # ── 扣除交易成本 ──────────────────────────────────────────────────────
    if cost_model is not None and turnover_by_date is not None:
        cost_per_turnover = cost_model.round_trip_cost()
        group_ret = (
            group_ret.join(turnover_by_date, on="trade_date", how="left")
            .with_columns(
                (pl.col("ret") - pl.col("turnover").fill_null(0.0) * cost_per_turnover).alias("ret")
            )
            .drop("turnover")
        )

    # 累计净值
    nav = group_ret.with_columns((1 + pl.col("ret")).cum_prod().over("group").alias("nav"))

    # Long-Short: group 0 vs group n_groups-1
    ls_ret = (
        group_ret.filter(pl.col("group").is_in([0, n_groups - 1]))
        .group_by("trade_date")
        .agg(
            [
                (
                    pl.when(pl.col("group") == n_groups - 1)
                    .then(pl.col("ret"))
                    .otherwise(-pl.col("ret"))
                )
                .sum()
                .alias("ret")
            ]
        )
        .sort("trade_date")
    )

    # ── 多空对冲额外扣融券费率 ─────────────────────────────────────────────
    if cost_model is not None:
        borrow_per_period = cost_model.borrow_rate_per_period(frequency)
        ls_ret = ls_ret.with_columns((pl.col("ret") - borrow_per_period).alias("ret"))

    ls_nav = ls_ret.with_columns((1 + pl.col("ret")).cum_prod().alias("nav"))

    def _group_stats(rets: np.ndarray) -> dict[str, float]:
        valid = rets[np.isfinite(rets)]
        if len(valid) == 0:
            return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        ann_ret = float(np.mean(valid) * TRADING_DAYS_PER_YEAR)
        ann_vol = float(np.std(valid) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum = np.cumprod(1 + valid)
        max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
        return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}

    summary_stats: dict[int | str, dict[str, float]] = {}
    for g in range(n_groups):
        rets = group_ret.filter(pl.col("group") == g)["ret"].to_numpy()
        summary_stats[g] = _group_stats(rets)

    ls_rets = ls_nav["ret"].to_numpy()
    summary_stats["long_short"] = _group_stats(ls_rets)

    return BacktestResult(
        factor_name=factor_name,
        n_groups=n_groups,
        daily_returns=group_ret,
        nav=nav,
        long_short_nav=ls_nav,
        summary_stats=summary_stats,
        frequency=frequency,
        ret_definition="fwd_ret_1d",
    )
