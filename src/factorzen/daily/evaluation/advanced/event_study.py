"""Event Study — 事件前后窗口累计收益分析。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EventStudyResult:
    """事件研究结果。

    Attributes:
        windows: 相对事件日的窗口列表，如 [-5, -4, ..., 0, ..., 20]
        avg_cumret: 各窗口日的平均累计收益（shape: len(windows)）
        ci_95: 95% 置信区间半宽（1.96 * std / sqrt(n_events)），shape: len(windows)
        n_events: 事件数量
    """

    windows: list[int]
    avg_cumret: np.ndarray
    ci_95: np.ndarray
    n_events: int


def compute_event_study(
    factor_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    event_threshold: float = 0.95,
    pre_window: int = 5,
    post_window: int = 20,
    factor_col: str = "factor_clean",
) -> EventStudyResult:
    """选 factor top event_threshold 分位作为事件，计算事件前后窗口平均累计收益。

    Args:
        factor_df: 含 trade_date, ts_code, {factor_col} 的因子 DataFrame
        ret_df: 含 trade_date, ts_code, ret_1d 的收益 DataFrame
        event_threshold: 事件阈值分位数（默认 0.95，即 top 5% 为事件）
        pre_window: 事件前窗口天数（默认 5）
        post_window: 事件后窗口天数（默认 20）
        factor_col: 因子列名

    Returns:
        EventStudyResult
    """
    windows = list(range(-pre_window, post_window + 1))
    n_windows = len(windows)

    # 过滤有效因子值
    valid_factor = factor_df.filter(pl.col(factor_col).is_not_null())
    if valid_factor.is_empty():
        return EventStudyResult(
            windows=windows,
            avg_cumret=np.zeros(n_windows),
            ci_95=np.zeros(n_windows),
            n_events=0,
        )

    # 按日期找 top event_threshold 分位的事件
    event_rows = (
        valid_factor.with_columns(
            pl.col(factor_col).rank(method="average").over("trade_date").alias("_rank"),
            pl.len().over("trade_date").alias("_n"),
        )
        .filter(pl.col("_rank") / pl.col("_n") >= event_threshold)
        .select(["trade_date", "ts_code"])
    )

    if event_rows.is_empty():
        return EventStudyResult(
            windows=windows,
            avg_cumret=np.zeros(n_windows),
            ci_95=np.zeros(n_windows),
            n_events=0,
        )

    # 构建日期索引（用于窗口偏移）
    all_dates = sorted(ret_df["trade_date"].unique().to_list())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 对 ret_df 建立 (date, ts_code) -> ret_1d 的查找字典
    ret_lookup: dict[tuple, float] = {}
    for row in ret_df.iter_rows(named=True):
        ret = row["ret_1d"]
        ret_lookup[(row["trade_date"], row["ts_code"])] = np.nan if ret is None else float(ret)

    # 对每个事件，计算窗口内累计收益
    event_cumrets: list[np.ndarray] = []

    for row in event_rows.iter_rows(named=True):
        event_date = row["trade_date"]
        ts_code = row["ts_code"]

        if event_date not in date_to_idx:
            continue
        event_idx = date_to_idx[event_date]

        # 收集各窗口日的日收益
        daily_rets = []
        valid_event = True
        for w in windows:
            target_idx = event_idx + w
            if target_idx < 0 or target_idx >= len(all_dates):
                valid_event = False
                break
            target_date = all_dates[target_idx]
            ret = ret_lookup.get((target_date, ts_code), np.nan)
            daily_rets.append(ret)

        if not valid_event:
            continue

        # 以事件日收盘为基准的累计收益：cumret[w] = price[w]/price[0] - 1。
        # daily_arr[i] 是窗口日 windows[i] 的日收益（前一交易日收盘→当日收盘），
        # 故事件日收益 daily_arr[base_idx] 是"day-1→day0"，只用于把事件前折算到基准，
        # 绝不计入事件后（否则 w=+1 会含 r0，w=0→+1 出现约 2*r0 的假跳变）。
        daily_arr = np.array(daily_rets, dtype=float)

        # 如果缺失数据过多（超过 50%），跳过该事件
        nan_ratio = np.sum(np.isnan(daily_arr)) / len(daily_arr)
        if nan_ratio > 0.5:
            continue

        # w=0 对应 pre_window 索引
        base_idx = pre_window
        cumrets = np.zeros(n_windows)
        for i in range(n_windows):
            if i == base_idx:
                cumrets[i] = 0.0  # 事件日基准
            elif i > base_idx:
                # 事件后：price[+k]/price[0] = prod(1 + r_{1..k})，不含事件日收益
                segment = daily_arr[base_idx + 1 : i + 1]
                cumrets[i] = float(np.nanprod(1.0 + segment)) - 1.0
            else:
                # 事件前：price[-k]/price[0] = 1 / prod(1 + r_{-k+1..0})（含事件日收益折算）
                segment = daily_arr[i + 1 : base_idx + 1]
                denom = float(np.nanprod(1.0 + segment))
                cumrets[i] = (1.0 / denom) - 1.0 if denom != 0 else np.nan

        event_cumrets.append(cumrets)

    if len(event_cumrets) == 0:
        return EventStudyResult(
            windows=windows,
            avg_cumret=np.zeros(n_windows),
            ci_95=np.zeros(n_windows),
            n_events=0,
        )

    cumret_matrix = np.array(event_cumrets)  # shape: (n_events, n_windows)
    avg_cumret = np.mean(cumret_matrix, axis=0)
    n_events = len(event_cumrets)

    if n_events > 1:
        ci_95 = 1.96 * np.std(cumret_matrix, axis=0, ddof=1) / np.sqrt(n_events)
    else:
        ci_95 = np.zeros(n_windows)

    return EventStudyResult(
        windows=windows,
        avg_cumret=avg_cumret,
        ci_95=ci_95,
        n_events=n_events,
    )
