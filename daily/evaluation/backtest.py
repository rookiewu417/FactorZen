"""分层回测。按因子值分组，计算各组收益与多空对冲表现。"""

from dataclasses import dataclass
import polars as pl
import numpy as np


@dataclass
class BacktestResult:
    factor_name: str
    n_groups: int
    daily_returns: pl.DataFrame   # trade_date, group, ret
    nav: pl.DataFrame              # trade_date, group, nav (累计净值)
    long_short_nav: pl.DataFrame   # trade_date, nav
    summary_stats: dict            # {group: {ann_ret, ann_vol, sharpe, max_dd}}
    frequency: str = "daily"

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(self.frequency, self.frequency)
        lines = []
        for g, stats in sorted(self.summary_stats.items()):
            if g == "long_short":
                lines.append(f"  Long-Short: Sharpe={stats['sharpe']:.2f} MaxDD={stats['max_dd']:.1%}")
            else:
                lines.append(f"  G{g}: ret={stats['ann_ret']:.2%} Sharpe={stats['sharpe']:.2f}")
        return f"Backtest ({self.n_groups} groups, {freq_label}):\n" + "\n".join(lines)


def run_stratified_backtest(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    frequency: str = "daily",
) -> BacktestResult:
    """分层回测。
    
    Args:
        factor_df: 因子值，列: trade_date, ts_code, {factor_col}
        daily_ret: 日收益，列: trade_date, ts_code, ret（单日收益）
        factor_col: 使用的因子列
        n_groups: 分组数
    """
    # 合并因子和收益
    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")
    
    # 每日截面分组 (qcut)
    merged = merged.with_columns(
        pl.col(factor_col).rank("ordinal", descending=False).over("trade_date")
        .alias("_rank")
    ).with_columns(
        ((pl.col("_rank") - 1) * n_groups // pl.col("_rank").max().over("trade_date"))
        .cast(pl.Int32)
        .alias("group")
    ).drop("_rank")
    
    # 组日收益（等权）
    group_ret = (
        merged.group_by(["trade_date", "group"])
        .agg(pl.col("ret").mean().alias("ret"))
        .sort(["trade_date", "group"])
    )
    
    # 累计净值
    nav = group_ret.with_columns(
        (1 + pl.col("ret")).cum_prod().over("group").alias("nav")
    )
    
    # Long-Short: group 0 vs group n_groups-1
    ls_ret = (
        group_ret
        .filter(pl.col("group").is_in([0, n_groups - 1]))
        .group_by("trade_date")
        .agg([
            (pl.when(pl.col("group") == n_groups - 1).then(pl.col("ret"))
             .otherwise(-pl.col("ret"))).sum().alias("ret")
        ])
        .sort("trade_date")
    )
    ls_nav = ls_ret.with_columns(
        (1 + pl.col("ret")).cum_prod().alias("nav")
    )
    
    # 统计
    summary_stats = {}
    for g in range(n_groups):
        rets = group_ret.filter(pl.col("group") == g)["ret"].to_numpy()
        ann_ret = float(np.mean(rets) * 252)
        ann_vol = float(np.std(rets) * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + rets)
        max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
        summary_stats[g] = {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}
    
    ls_rets = ls_nav["ret"].to_numpy()
    ls_ann_ret = float(np.mean(ls_rets) * 252)
    ls_ann_vol = float(np.std(ls_rets) * np.sqrt(252))
    ls_sharpe = ls_ann_ret / ls_ann_vol if ls_ann_vol > 0 else 0
    ls_cum = np.cumprod(1 + ls_rets)
    ls_max_dd = float(np.min(ls_cum / np.maximum.accumulate(ls_cum) - 1))
    summary_stats["long_short"] = {"ann_ret": ls_ann_ret, "ann_vol": ls_ann_vol, "sharpe": ls_sharpe, "max_dd": ls_max_dd}
    
    return BacktestResult(
        factor_name="",
        n_groups=n_groups,
        daily_returns=group_ret,
        nav=nav,
        long_short_nav=ls_nav,
        summary_stats=summary_stats,
        frequency=frequency,
    )
