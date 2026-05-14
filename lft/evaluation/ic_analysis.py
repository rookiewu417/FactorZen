"""Rank IC 分析。计算因子值与未来收益的截面 Spearman 相关系数。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def compute_fwd_returns(
    price_df: pl.DataFrame,
    horizons: list[int] | None = None,
    ret_col: str = "ret_1d",
) -> pl.DataFrame:
    """预计算各时间窗口的前向收益。

    Args:
        price_df: 含 trade_date, ts_code, {ret_col} 的日收益 DataFrame。
            ret_col 为当日收益率（如涨跌幅 / 100）。
        horizons: 前向窗口（交易日），默认 [1, 5, 10, 20]。
        ret_col: 单日收益列名。

    Returns:
        含 fwd_ret_{h}d 列的 DataFrame。
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    df = price_df.sort(["ts_code", "trade_date"])
    for h in horizons:
        df = df.with_columns(
            (pl.col(ret_col).shift(-h).over("ts_code").alias(f"fwd_ret_{h}d"))
        )
    return df


@dataclass
class ICAnalysisResult:
    factor_name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_ratio: float
    n_periods: int
    ic_series: pl.DataFrame  # trade_date, ic
    decay: dict[int, float] = field(default_factory=dict)  # {horizon_days: ic_mean}
    frequency: str = "daily"

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(self.frequency, self.frequency)
        lines = [
            f"Factor: {self.factor_name} [{freq_label}]",
            f"  IC Mean: {self.ic_mean:.4f}  |  IC Std: {self.ic_std:.4f}  |  IR: {self.ir:.2f}",
            f"  IC > 0 Ratio: {self.ic_positive_ratio:.1%}  |  Periods: {self.n_periods}",
        ]
        if self.decay:
            decay_parts = [f"{h}d={v:.4f}" for h, v in sorted(self.decay.items())]
            lines.append(f"  IC Decay: {', '.join(decay_parts)}")
        return "\n".join(lines)


def compute_rank_ic(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    horizons: list[int] | None = None,
    frequency: str = "daily",
) -> ICAnalysisResult:
    """计算 Rank IC。

    Args:
        factor_df: 因子值 DataFrame，列: trade_date, ts_code, {factor_col}
        daily_ret: 日收益 DataFrame，列: trade_date, ts_code, fwd_ret_1d
                   （需通过 compute_fwd_returns() 预计算各 horizon 的前向收益）
        factor_col: 因子列名。
        horizons: IC decay 计算的时间窗口，默认 [1, 5, 10, 20]。

    Returns:
        ICAnalysisResult: 含 IC 统计量、IC 序列、IC Decay。
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    # 合并因子和收益
    merged = factor_df.join(
        daily_ret,
        on=["trade_date", "ts_code"],
        how="inner",
    )

    # ---------- 每日截面 Rank IC (horizon=1) ----------
    ic_records: list[dict] = []
    trade_dates = merged["trade_date"].unique().sort()
    for d in trade_dates.to_list():
        cross = merged.filter(pl.col("trade_date") == d)
        # 对因子值和 fwd_ret 做联合有效掩码，避免 fwd_ret 末尾 NaN 污染 spearmanr
        valid = cross[factor_col].is_not_null() & cross["fwd_ret_1d"].is_not_null() & cross["fwd_ret_1d"].is_finite()
        sub = cross.filter(valid)
        if len(sub) < 30:
            continue
        x = sub[factor_col].to_numpy()
        y = sub["fwd_ret_1d"].to_numpy()
        ic, _ = spearmanr(x, y)
        if not np.isnan(ic):
            ic_records.append({"trade_date": d, "ic": ic})

    ic_series = pl.DataFrame(ic_records).sort("trade_date")
    ic_values = ic_series["ic"].to_numpy()
    ic_mean = float(np.mean(ic_values)) if len(ic_values) > 0 else 0.0
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_values > 0)) if len(ic_values) > 0 else 0.0

    # ---------- IC Decay ----------
    decay: dict[int, float] = {}
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ics: list[float] = []
        for d in trade_dates.to_list():
            cross = merged.filter(pl.col("trade_date") == d)
            valid = cross[factor_col].is_not_null() & cross[ret_col].is_not_null() & cross[ret_col].is_finite()
            sub = cross.filter(valid)
            if len(sub) < 30:
                continue
            ic, _ = spearmanr(sub[factor_col].to_numpy(), sub[ret_col].to_numpy())
            if not np.isnan(ic):
                h_ics.append(ic)
        if h_ics:
            decay[h] = float(np.mean(h_ics))

    return ICAnalysisResult(
        factor_name=factor_col,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_positive_ratio=ic_pos,
        n_periods=len(ic_values),
        ic_series=ic_series,
        decay=decay,
        frequency=frequency,
    )
