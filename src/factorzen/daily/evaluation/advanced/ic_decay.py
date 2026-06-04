"""IC Decay 增强分析 — 多持有期 IC 衰减。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger
from factorzen.daily.evaluation.ic_analysis import _rank_ic_by_date

logger = get_logger(__name__)


@dataclass
class ICDecayResult:
    """单个持有期的 IC 衰减结果。"""

    horizon: int
    ic_mean: float
    ic_std: float
    ic_series: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return f"Horizon {self.horizon}d: IC_mean={self.ic_mean:.4f}, IC_std={self.ic_std:.4f}"


def compute_ic_decay(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    horizons: list[int] | None = None,
) -> list[ICDecayResult]:
    """计算因子 IC 随持有期的衰减。

    对每个持有期 h，计算因子值与 fwd_ret_{h}d 的 Rank IC。

    Args:
        factor_df: 因子值 DataFrame，列: trade_date, ts_code, {factor_col}
        daily_ret: 前向收益 DataFrame，列: trade_date, ts_code, fwd_ret_{h}d
        factor_col: 因子列名
        horizons: 持有期列表；None 时从 daily_ret 列名自动检测

    Returns:
        list[ICDecayResult]: 每个 holding period 一个结果
    """
    # 自动检测 horizons
    if horizons is None:
        detected: list[int] = []
        for c in daily_ret.columns:
            match = re.fullmatch(r"fwd_ret_(\d+)d", c)
            if match is not None:
                detected.append(int(match.group(1)))
        horizons = sorted(detected)

    if not horizons:
        return []

    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")

    results: list[ICDecayResult] = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
        ic_arr = h_ic_df["ic"].drop_nulls().to_numpy()
        results.append(
            ICDecayResult(
                horizon=h,
                ic_mean=float(np.mean(ic_arr)) if len(ic_arr) > 0 else float("nan"),
                ic_std=float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else float("nan"),
                ic_series=ic_arr.tolist(),
            )
        )

    return results
