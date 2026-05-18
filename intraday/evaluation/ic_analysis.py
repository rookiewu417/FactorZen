"""Intraday factor IC analysis.

Computes minute-level cross-sectional Rank IC between factor values and
forward intraday returns. Aggregates by day and by time-of-day segment.

Design: same polars group_by + pl.corr pattern as daily/evaluation/ic_analysis.py,
but groups on (trade_date, trade_time) rather than just trade_date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_MIN_CROSS_SAMPLES = 10  # lower threshold for intraday cross-sections


@dataclass
class IntradayICResult:
    factor_name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_ratio: float
    n_periods: int
    daily_ic: pl.DataFrame  # trade_date, ic_mean (daily aggregate)
    segment_ic: pl.DataFrame  # segment, ic_mean (open/midday/close)

    def summary(self) -> str:
        lines = [
            f"Intraday IC: {self.factor_name}",
            f"  IC Mean: {self.ic_mean:.4f}  |  IC Std: {self.ic_std:.4f}  |  IR: {self.ir:.2f}",
            f"  IC > 0 Ratio: {self.ic_positive_ratio:.1%}  |  Minute-bars: {self.n_periods}",
        ]
        if not self.segment_ic.is_empty():
            lines.append("  Segment IC:")
            for row in self.segment_ic.iter_rows(named=True):
                lines.append(f"    {row['segment']}: IC={row['ic']:.4f}")
        return "\n".join(lines)


def _assign_segment(df: pl.DataFrame, time_col: str = "trade_time") -> pl.DataFrame:
    """Assign each bar to open / midday / close segment based on time-of-day."""
    # Chinese A-share session: 09:30-10:00 open, 14:30-15:00 close, rest midday
    return df.with_columns(
        pl.when(
            pl.col(time_col).dt.hour().eq(9)
            | (pl.col(time_col).dt.hour().eq(10) & pl.col(time_col).dt.minute().lt(1))
        )
        .then(pl.lit("open"))
        .when(pl.col(time_col).dt.hour().eq(14) & pl.col(time_col).dt.minute().ge(30))
        .then(pl.lit("close"))
        .otherwise(pl.lit("midday"))
        .alias("segment")
    )


def compute_intraday_rank_ic(
    factor_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    factor_col: str = "factor_value",
    ret_col: str = "fwd_ret_1bar",
    time_col: str = "trade_time",
    min_samples: int = _MIN_CROSS_SAMPLES,
) -> IntradayICResult:
    """Compute cross-sectional Rank IC for an intraday factor.

    Args:
        factor_df: DataFrame with trade_time, ts_code, {factor_col}.
        ret_df: DataFrame with trade_time, ts_code, {ret_col} (next-bar return).
        factor_col: Factor column name.
        ret_col: Forward return column name.
        time_col: Timestamp column name.
        min_samples: Minimum cross-section size to keep a bar.

    Returns:
        IntradayICResult with per-bar IC, daily aggregate, and segment breakdown.
    """
    merged = factor_df.join(ret_df, on=[time_col, "ts_code"], how="inner")

    valid = merged.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    if valid.is_empty():
        empty_daily = pl.DataFrame({"trade_date": [], "ic": []}).cast(
            {"trade_date": pl.Date, "ic": pl.Float64}
        )
        empty_seg = pl.DataFrame({"segment": [], "ic": []}).cast(
            {"segment": pl.Utf8, "ic": pl.Float64}
        )
        return IntradayICResult(
            factor_name=factor_col,
            ic_mean=0.0,
            ic_std=0.0,
            ir=0.0,
            ic_positive_ratio=0.0,
            n_periods=0,
            daily_ic=empty_daily,
            segment_ic=empty_seg,
        )

    # Rank within each (trade_time) cross-section
    ranked = valid.with_columns(
        [
            pl.col(factor_col).rank(method="average").over(time_col).alias("_f_rank"),
            pl.col(ret_col).rank(method="average").over(time_col).alias("_r_rank"),
        ]
    )

    # Per-bar IC
    bar_ic = (
        ranked.group_by(time_col)
        .agg(
            [
                pl.corr("_f_rank", "_r_rank").alias("ic"),
                pl.len().alias("_n"),
            ]
        )
        .filter(pl.col("_n") >= min_samples)
        .drop("_n")
        .sort(time_col)
    )

    ic_vals = bar_ic["ic"].drop_nulls().to_numpy()
    ic_mean = float(np.mean(ic_vals)) if len(ic_vals) > 0 else 0.0
    ic_std = float(np.std(ic_vals, ddof=1)) if len(ic_vals) > 1 else 0.0
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_vals > 0)) if len(ic_vals) > 0 else 0.0

    # Daily aggregate IC: mean of bar ICs within each date
    daily_ic = (
        bar_ic.with_columns(pl.col(time_col).dt.date().alias("trade_date"))
        .group_by("trade_date")
        .agg(pl.col("ic").mean())
        .sort("trade_date")
    )

    # Segment IC: assign open/midday/close
    bar_with_seg = _assign_segment(bar_ic, time_col)
    segment_ic = bar_with_seg.group_by("segment").agg(pl.col("ic").mean()).sort("segment")

    return IntradayICResult(
        factor_name=factor_col,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_positive_ratio=ic_pos,
        n_periods=len(ic_vals),
        daily_ic=daily_ic,
        segment_ic=segment_ic,
    )
