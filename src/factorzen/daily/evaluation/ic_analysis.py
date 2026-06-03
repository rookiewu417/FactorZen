"""Rank IC 分析。计算因子值与未来收益的截面 Spearman 相关系数。

性能说明：
    使用 polars group_by + pearson_corr(ranks) 替代逐日 Python for 循环，
    Spearman 相关系数 = 对排名后序列求 Pearson 相关系数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
import polars as pl
import statsmodels.api as sm

from factorzen.core.validation import require_columns

# 最少截面样本数（低于此值的交易日跳过）
_MIN_CROSS_SAMPLES = 30


def compute_fwd_returns(
    price_df: pl.DataFrame,
    horizons: list[int] | None = None,
    ret_col: str = "ret_1d",
    price_col: str = "close",
    code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """预计算各时间窗口的前向持有期收益。

    Args:
        price_df: 含 trade_date, ts_code，以及 close 或 {ret_col} 的 DataFrame。
        horizons: 前向窗口（交易日），默认 [1, 5, 10, 20]。
        ret_col: 单日收益列名。
        price_col: 价格列名。存在时优先使用 close[t+h] / close[t] - 1。
        code_col: 股票代码列名。
        date_col: 日期列名。

    Returns:
        含 fwd_ret_{h}d 列的 DataFrame。
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    require_columns(price_df, [code_col, date_col], context="compute_fwd_returns")
    if price_col not in price_df.columns and ret_col not in price_df.columns:
        raise ValueError(
            f"compute_fwd_returns: 需要价格列 '{price_col}' 或单日收益列 '{ret_col}' 之一;"
            f"实际列为 {list(price_df.columns)}"
        )

    df = price_df.sort([code_col, date_col])
    for h in horizons:
        if price_col in df.columns:
            future_price = pl.col(price_col).shift(-h).over(code_col)
            df = df.with_columns((future_price / pl.col(price_col) - 1.0).alias(f"fwd_ret_{h}d"))
        else:
            compounded = pl.lit(1.0)
            for step in range(1, h + 1):
                compounded = compounded * (1.0 + pl.col(ret_col).shift(-step).over(code_col))
            df = df.with_columns((compounded - 1.0).alias(f"fwd_ret_{h}d"))
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
    ic_tstat: float = 0.0
    ic_pvalue: float = 1.0
    # Multi-period IC: {horizon: ICAnalysisResult-like fields} for consistency check
    multi_period: dict[int, dict[str, float]] = field(default_factory=dict)
    # Out-of-sample split IC: {"train": ic_mean, "test": ic_mean}（向后兼容）
    # 语义上 train/test 分别对应历史观察期和未来验证期，不表示固定因子被重新拟合。
    oos_ic: dict[str, float] = field(default_factory=dict)
    # Walk-forward cross-validation: list of {"train_ic": float, "test_ic": float}
    walk_forward_ic: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(
            self.frequency, self.frequency
        )
        sig_label = (
            "***"
            if self.ic_pvalue < 0.01
            else ("**" if self.ic_pvalue < 0.05 else ("*" if self.ic_pvalue < 0.1 else ""))
        )
        lines = [
            f"Factor: {self.factor_name} [{freq_label}]",
            f"  IC Mean: {self.ic_mean:.4f}  |  IC Std: {self.ic_std:.4f}  |  IR: {self.ir:.2f}",
            f"  t-stat: {self.ic_tstat:.2f}{sig_label}  |  p-value: {self.ic_pvalue:.4f}",
            f"  IC > 0 Ratio: {self.ic_positive_ratio:.1%}  |  Periods: {self.n_periods}",
        ]
        if self.decay:
            decay_parts = [f"{h}d={v:.4f}" for h, v in sorted(self.decay.items())]
            lines.append(f"  IC Decay: {', '.join(decay_parts)}")
        if self.multi_period:
            mp_parts = [
                f"{h}d: IC={v['ic_mean']:.4f},IR={v['ir']:.2f}"
                for h, v in sorted(self.multi_period.items())
            ]
            lines.append(f"  Multi-period: {', '.join(mp_parts)}")
        if self.oos_ic:
            train_ic = self.oos_ic.get("train", float("nan"))
            test_ic = self.oos_ic.get("test", float("nan"))
            lines.append(f"  OOS: IS observation IC={train_ic:.4f}, OOS validation IC={test_ic:.4f}")
        return "\n".join(lines)


def _rank_ic_by_date(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    min_samples: int = _MIN_CROSS_SAMPLES,
) -> pl.DataFrame:
    """计算每日截面 Rank IC（polars vectorized 实现）。

    原理：Spearman 相关系数 = Pearson(rank(x), rank(y))
    用 polars 的 `.rank().over("trade_date")` + `pearson_corr` group_by 实现，
    避免 Python-level for 循环。

    Returns:
        pl.DataFrame with columns [trade_date, ic]，已按 trade_date 排序。
    """
    # 联合有效掩码（因子 + 前向收益都不为 null/inf）
    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    if valid_df.is_empty():
        return pl.DataFrame({"trade_date": [], "ic": []}).cast(
            {"trade_date": pl.Date, "ic": pl.Float64}
        )

    # 截面内排名（average 方法，与 scipy.stats.spearmanr 默认一致）
    ranked = valid_df.with_columns(
        [
            pl.col(factor_col).rank(method="average").over("trade_date").alias("_factor_rank"),
            pl.col(ret_col).rank(method="average").over("trade_date").alias("_ret_rank"),
        ]
    )

    # group_by 日期，计算排名 Pearson 相关（= Spearman），过滤样本不足的日期
    ic_df = (
        ranked.group_by("trade_date")
        .agg(
            [
                pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                pl.len().alias("_n"),
            ]
        )
        .filter(pl.col("_n") >= min_samples)
        .drop("_n")
        .sort("trade_date")
    )
    return ic_df


@dataclass
class IcStats:
    """轻量级 IC 统计结果，供 compute_ic 使用。"""

    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_ratio: float
    n_periods: int
    ic_tstat: float
    ic_pvalue: float
    ic_series: pl.DataFrame  # trade_date, ic


class BothIcResult(TypedDict):
    """compute_ic(method='both') 的返回类型。"""

    rank: IcStats
    pearson: IcStats


def _pearson_ic_by_date(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    min_samples: int = _MIN_CROSS_SAMPLES,
) -> pl.DataFrame:
    """按交易日计算 Pearson IC（皮尔逊相关系数）。

    Returns:
        pl.DataFrame with columns [trade_date, ic]，已按 trade_date 排序。
    """
    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    if valid_df.is_empty():
        return pl.DataFrame({"trade_date": [], "ic": []}).cast(
            {"trade_date": pl.Date, "ic": pl.Float64}
        )

    return (
        valid_df.group_by("trade_date")
        .agg(
            [
                pl.corr(factor_col, ret_col, method="pearson").alias("ic"),
                pl.len().alias("_n"),
            ]
        )
        .filter(pl.col("_n") >= min_samples)
        .drop("_n")
        .sort("trade_date")
    )


def _build_ic_stats(ic_series: pl.DataFrame) -> IcStats:
    """从 IC 序列 DataFrame 构建 IcStats。"""
    ic_values = ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    ic_mean, ic_std, ir, ic_pos, tstat, pvalue = _ic_stats(ic_values)
    return IcStats(
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_positive_ratio=ic_pos,
        n_periods=len(ic_values),
        ic_tstat=tstat,
        ic_pvalue=pvalue,
        ic_series=ic_series,
    )


def compute_ic(
    df: pl.DataFrame,
    factor_col: str = "factor_clean",
    ret_col: str = "ret_1d",
    method: Literal["rank", "pearson", "both"] = "rank",
) -> IcStats | BothIcResult:
    """计算因子 IC（Rank 或 Pearson），简化版接口。

    与 compute_rank_ic 不同，此函数直接接受含因子值和收益的单个 DataFrame，
    不需要预先计算前向收益。

    Args:
        df: DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}
        factor_col: 因子列名（默认 "factor_clean"）
        ret_col: 收益列名（默认 "ret_1d"）
        method: "rank"（Spearman）/ "pearson" / "both"

    Returns:
        - method="rank" 或 "pearson": IcStats
        - method="both": BothIcResult {"rank": IcStats, "pearson": IcStats}
    """
    if method == "rank":
        ic_series = _rank_ic_by_date(df, factor_col, ret_col)
        return _build_ic_stats(ic_series)
    elif method == "pearson":
        ic_series = _pearson_ic_by_date(df, factor_col, ret_col)
        return _build_ic_stats(ic_series)
    else:  # "both"
        rank_series = _rank_ic_by_date(df, factor_col, ret_col)
        pearson_series = _pearson_ic_by_date(df, factor_col, ret_col)
        return BothIcResult(
            rank=_build_ic_stats(rank_series),
            pearson=_build_ic_stats(pearson_series),
        )


def _hac_maxlags(n: int) -> int:
    """Newey-West 最优滞后阶数：floor(4*(N/100)^(2/9))，最少 1。"""
    return max(1, math.floor(4 * (n / 100) ** (2 / 9)))


def _ic_stats(ic_values: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Compute (ic_mean, ic_std, ir, ic_pos, tstat, pvalue) from IC array.

    使用 Newey-West HAC 标准误计算 t 统计量，修正 IC 序列自相关导致的 t-stat 高估。

    Filters both NaN and inf before computing statistics.
    polars pl.corr() returns float NaN (not null) for degenerate cases,
    so drop_nulls() alone is not sufficient — we must strip NaN explicitly.
    """
    valid = ic_values[np.isfinite(ic_values)]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    ic_mean = float(np.mean(valid))
    ic_std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(valid > 0))

    if len(valid) > 4 and ic_std > 0:
        # Newey-West HAC OLS：被解释变量 = IC 序列，解释变量 = 常数项
        # H_0: IC 均值 = 0，HAC 标准误修正序列相关
        X = np.ones(len(valid))
        model = sm.OLS(valid, X)
        maxlags = _hac_maxlags(len(valid))
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        tstat = float(results.tvalues[0])
        pvalue = float(results.pvalues[0])
    else:
        tstat, pvalue = 0.0, 1.0

    return ic_mean, ic_std, ir, ic_pos, tstat, pvalue


def _compute_walk_forward_ic(
    ic_values: np.ndarray,
    n_folds: int = 5,
    embargo: int = 5,
) -> list[dict[str, float]]:
    """Walk-forward 5 折 IC 交叉验证。

    Args:
        ic_values: 时序 IC 数组（已过滤 nan/inf）。
        n_folds: 折数。
        embargo: 历史观察期末尾到未来验证期开头之间的 gap（防时序泄漏）。

    Returns:
        list of {"fold": int, "train_ic": float, "test_ic": float}，
        若样本太少（< n_folds * 2 + embargo）则返回空列表。
    """
    n = len(ic_values)
    min_required = n_folds * 2 + embargo
    if n < min_required:
        return []

    results: list[dict[str, float]] = []
    fold_size = n // (n_folds + 1)

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end + embargo
        test_end = test_start + fold_size

        if test_end > n:
            break

        train_vals = ic_values[:train_end]
        test_vals = ic_values[test_start:test_end]

        if len(train_vals) < 2 or len(test_vals) < 2:
            continue

        results.append(
            {
                "fold": fold + 1,
                "train_ic": float(np.mean(train_vals)),
                "test_ic": float(np.mean(test_vals)),
            }
        )

    return results


def compute_rank_ic(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    horizons: list[int] | None = None,
    frequency: str = "daily",
    oos_split: float = 0.7,
) -> ICAnalysisResult:
    """Compute Rank IC (polars vectorized).

    Args:
        factor_df: DataFrame with trade_date, ts_code, {factor_col}.
        daily_ret: DataFrame with trade_date, ts_code, fwd_ret_{h}d columns
                   (precomputed by compute_fwd_returns()).
        factor_col: Factor column name.
        horizons: IC decay horizons in trading days (default [1, 5, 10, 20]).
        frequency: Frequency label for summary display.
        oos_split: Fraction of dates used as in-sample training (default 0.7).

    Returns:
        ICAnalysisResult with IC stats, t-stat, p-value, multi-period IC, and OOS split.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    merged = factor_df.join(daily_ret, on=["trade_date", "ts_code"], how="inner")

    # ---------- Primary IC (horizon=1d) ----------
    ic_series = _rank_ic_by_date(merged, factor_col, "fwd_ret_1d")
    # drop_nulls() only removes polars nulls; pl.corr() can also return float NaN
    # for degenerate cases (constant column). Filter both here.
    ic_values = ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    ic_mean, ic_std, ir, ic_pos, tstat, pvalue = _ic_stats(ic_values)

    # ---------- IC Decay (all horizons) ----------
    decay: dict[int, float] = {}
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
        h_vals = h_ic_df["ic"].drop_nulls().drop_nans().to_numpy()
        if len(h_vals) > 0:
            decay[h] = float(np.mean(h_vals))

    # ---------- Multi-period IC consistency ----------
    multi_period: dict[int, dict[str, float]] = {}
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in merged.columns:
            continue
        h_ic_df = _rank_ic_by_date(merged, factor_col, ret_col)
        h_vals = h_ic_df["ic"].drop_nulls().drop_nans().to_numpy()
        if len(h_vals) > 0:
            h_mean, h_std, h_ir, h_pos, h_t, h_p = _ic_stats(h_vals)
            multi_period[h] = {
                "ic_mean": h_mean,
                "ic_std": h_std,
                "ir": h_ir,
                "ic_positive_ratio": h_pos,
                "tstat": h_t,
                "pvalue": h_p,
            }

    # ---------- Out-of-sample split (向后兼容，单次切分) ----------
    oos_ic: dict[str, float] = {}
    if len(ic_values) >= 4:
        n_train = max(2, int(len(ic_values) * oos_split))
        oos_ic["train"] = float(np.mean(ic_values[:n_train]))
        oos_ic["test"] = float(np.mean(ic_values[n_train:]))

    # ---------- Walk-forward cross-validation ----------
    walk_forward_ic = _compute_walk_forward_ic(ic_values, n_folds=5, embargo=5)

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
        ic_tstat=tstat,
        ic_pvalue=pvalue,
        multi_period=multi_period,
        oos_ic=oos_ic,
        walk_forward_ic=walk_forward_ic,
    )
