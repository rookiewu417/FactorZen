"""1m bar 重采样与 funding/OI 频率对齐(纯函数,无 IO)。

约定:bar 键列 ``trade_date``;intraday 为 naive-UTC ``Datetime("us")``、bar 起点
标记(Binance open_time 惯例);``daily`` 降为 ``Date``(与现日频路径零回归)。
funding/OI 事件按所在 bar 起点截断归属([bar_start, bar_end))。空窗不造假 bar。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.crypto.frequency import BAR_FREQS, normalize_freq

_SUM_COLS = ("vol", "amount", "taker_buy_volume")


def _bar_key(col: str, freq: str) -> pl.Expr:
    """事件时刻 → 所属 bar 键(起点截断;daily 再降 Date)。"""
    key = pl.col(col).dt.truncate(BAR_FREQS[freq].every)
    return key.cast(pl.Date) if freq == "daily" else key


def resample_bars(bars_1m: pl.DataFrame, freq: str) -> pl.DataFrame:
    f = normalize_freq(freq)
    if bars_1m.is_empty():
        return bars_1m
    aggs = [pl.col("open").first(), pl.col("high").max(),
            pl.col("low").min(), pl.col("close").last()]
    aggs += [pl.col(c).sum() for c in _SUM_COLS if c in bars_1m.columns]
    out = (
        bars_1m.sort(["ts_code", "trade_date"])
        .group_by_dynamic("trade_date", every=BAR_FREQS[f].every, group_by="ts_code",
                          closed="left", label="left")
        .agg(aggs)
    )
    if f == "daily":
        out = out.with_columns(pl.col("trade_date").cast(pl.Date))
    return out.sort(["ts_code", "trade_date"])


def _align_events(events: pl.DataFrame, freq: str, value_col: str, agg: pl.Expr) -> pl.DataFrame:
    f = normalize_freq(freq)
    if events.is_empty():
        key_dtype: pl.DataType = pl.Date() if f == "daily" else pl.Datetime("us")
        return pl.DataFrame(
            schema={"ts_code": pl.String, "trade_date": key_dtype, value_col: pl.Float64})
    return (
        events.sort(["ts_code", "event_time"])
        .with_columns(_bar_key("event_time", f).alias("trade_date"))
        .group_by(["ts_code", "trade_date"])
        .agg(agg)
        .sort(["ts_code", "trade_date"])
    )


def align_funding(events: pl.DataFrame, freq: str) -> pl.DataFrame:
    """funding 事件 → bar 内求和(日频=当日各档和,现行为)。"""
    return _align_events(events, freq, "funding_rate", pl.col("funding_rate").sum())


def align_open_interest(metrics: pl.DataFrame, freq: str) -> pl.DataFrame:
    """OI 5 分钟点 → bar 内最后一笔(日频=当日最后值)。"""
    return _align_events(metrics, freq, "open_interest", pl.col("open_interest").last())
