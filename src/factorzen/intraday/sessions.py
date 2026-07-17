"""A 股日内 session / 分钟频率的单一真源。

口径结论（跨年抽查 2017/2018/2020/2022/2024/2025 无漂移，写死为常量）：

- **bar 标签约定 = bar-end**：常规标签 09:31..11:30（120 根）∪ 13:01..15:00（120 根），
  共 240 根；标签 t 的 bar 覆盖 (t-1min, t]。**没有 13:00 标签、有 11:30 标签**。
- 另有特殊 **09:30 竞价 bar**（开盘集合竞价；SH 股票 OHLC 四价相同，SZ 含开盘瞬间成交）。
- **15:00 bar 含收盘集合竞价**（14:57-14:59 标签 bar 常为 vol=0 价格延续）。
- **15:01..15:30 标签 bar 仅在北交所 920 前缀代码上出现**（2024 起），量极小 →
  政策：**一律剔除 >15:00 的 bar**（``AFTER_HOURS_POLICY = "drop"``）。
- 分钟 vol=股、amount=元；日线 vol=手、amount=千元；分钟按日求和与日线精确闭合（×100 / ×1000）。
- 覆盖：2017 全年、2018 十个月、**2019 整年缺失**、2020-2025 全、2026 到 04-10。

实证审计见 ``docs/plans/20260715-minute-upgrade.md``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

BAR_LABEL_CONVENTION: Final[str] = "end"
FIRST_BAR_INCLUDES_AUCTION: Final[bool] = True
AFTER_HOURS_POLICY: Final[str] = "drop"

_MORNING_END_IDX: Final[int] = 120  # 11:30 对应 session index


@dataclass(frozen=True)
class AShareBarFreq:
    """A 股分钟 bar 频率描述。"""

    minutes: int
    bars_per_day: int


ASHARE_BAR_FREQS: dict[str, AShareBarFreq] = {
    "1min": AShareBarFreq(1, 240),
    "5min": AShareBarFreq(5, 48),
    "15min": AShareBarFreq(15, 16),
    "30min": AShareBarFreq(30, 8),
    "60min": AShareBarFreq(60, 4),
}

def normalize_freq(freq: str) -> str:
    """规范化频率字符串；只接受 ``ASHARE_BAR_FREQS`` 精确键。

    Raises:
        ValueError: 未知频率。
    """
    if freq not in ASHARE_BAR_FREQS:
        raise ValueError(
            f"未知频率: {freq!r}，支持 {sorted(ASHARE_BAR_FREQS)}"
        )
    return freq


def _time_of_day_minutes(time_col: str) -> pl.Expr:
    """``trade_time`` → 当日从 00:00 起的分钟数（忽略秒）。

    注意：polars ``dt.hour()`` / ``dt.minute()`` 返回 Int8，必须先 cast 到 Int32，
    否则 ``hour*60`` 会 i8 溢出（如 15*60 → -124）。
    """
    return (
        pl.col(time_col).dt.hour().cast(pl.Int32) * 60
        + pl.col(time_col).dt.minute().cast(pl.Int32)
    )


def _canonical_mask(time_col: str) -> pl.Expr:
    """canonical 时间掩码：{09:30} ∪ [09:31,11:30] ∪ [13:01,15:00]。"""
    tod = _time_of_day_minutes(time_col)
    return (
        (tod == 570)  # 09:30 竞价
        | ((tod >= 571) & (tod <= 690))  # 09:31..11:30
        | ((tod >= 781) & (tod <= 900))  # 13:01..15:00
    )


def canonicalize_minute(lf: pl.LazyFrame, *, time_col: str = "trade_time") -> pl.LazyFrame:
    """掩码收口：只保留 canonical session 时间戳的行。

    保留 time-of-day ∈ {09:30} ∪ [09:31, 11:30] ∪ [13:01, 15:00]；
    其余（>15:00 盘后、任何意外时间戳）一律 drop。对 LazyFrame 纯谓词操作。
    """
    return lf.filter(_canonical_mask(time_col))


def session_bar_index(time_col: str = "trade_time") -> pl.Expr:
    """返回 Int32 表达式：session bar 索引。

    - 09:30 → 0
    - 09:31..11:30 → 1..120（= hour*60+minute-570）
    - 13:01..15:00 → 121..240（= 120+(hour*60+minute-780)）

    前置条件：输入已 canonicalize；canonical 集合之外的时间返回 null。
    """
    tod = _time_of_day_minutes(time_col)
    return (
        pl.when(tod == 570)
        .then(pl.lit(0, dtype=pl.Int32))
        .when((tod >= 571) & (tod <= 690))
        .then((tod - 570).cast(pl.Int32))
        .when((tod >= 781) & (tod <= 900))
        .then((120 + (tod - 780)).cast(pl.Int32))
        .otherwise(pl.lit(None, dtype=pl.Int32))
    )


def _empty_bars() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ts_code": pl.String,
            "trade_time": pl.Datetime("us"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "vol": pl.Int64,
            "amount": pl.Float64,
        }
    )


def _bucket_end_minutes(end_idx: pl.Expr) -> pl.Expr:
    """session end_idx → 当日从 00:00 起的分钟数（bar-end 标签）。"""
    return (
        pl.when(end_idx <= _MORNING_END_IDX)
        .then(570 + end_idx)  # 09:30 + end_idx 分钟
        .otherwise(780 + (end_idx - _MORNING_END_IDX))  # 13:00 + (end_idx-120) 分钟
    )


def resample_intraday(
    bars: pl.DataFrame,
    freq: str,
    *,
    time_col: str = "trade_time",
    already_canonical: bool = False,
) -> pl.DataFrame:
    """将 1min bars 重采样为指定频率（bar-end 标签，不跨午休）。

    语义：
    1. ``normalize_freq``；``k = minutes``；默认做 canonicalize 同款过滤（防御性）；
       若调用方已 canonicalize，传 ``already_canonical=True`` 跳过重复过滤。
    2. ``idx = session_bar_index()``；``bucket = 0 if idx==0 else (idx-1)//k``。
    3. 按 ``(ts_code, trade_time.dt.date(), bucket)`` 分组聚合：
       open=时间最早 bar 的 open，high=max，low=min，close=时间最晚 bar 的 close，
       vol=sum，amount=sum（``arg_min``/``arg_max``，**不依赖输入行序**）。
    4. 输出标签 = 桶末 bar-end：``end_idx=(bucket+1)*k``；
       ``end_idx<=120`` → 09:30+end_idx 分钟；否则 13:00+(end_idx-120) 分钟。
    5. 输出 schema：ts_code, trade_time, open/high/low/close, vol(Int64), amount(Float64)。
       **行序无契约**（调用方需序时自行 ``sort(["ts_code", time_col])``）；
       引擎 / discovery 下游均会再 sort。``freq="1min"`` 时竞价 bar 并入 09:31 桶。
    6. 空帧返回空帧（schema 正确）；某桶只有部分 bar（临停/缺数据）也正常聚合。
    """
    freq = normalize_freq(freq)
    k = ASHARE_BAR_FREQS[freq].minutes

    if bars.is_empty():
        return _empty_bars()

    # 调用方已 canonicalize 时跳过全表再滤（引擎/discovery 主路径）
    filtered = bars if already_canonical else bars.filter(_canonical_mask(time_col))
    if filtered.is_empty():
        return _empty_bars()

    # 无预排序：open/close 用 arg_min/arg_max，避免 DataFrame.sort 热点
    tod = (
        pl.col(time_col).dt.hour().cast(pl.Int32) * 60
        + pl.col(time_col).dt.minute().cast(pl.Int32)
    )
    idx = (
        pl.when(tod == 570)
        .then(pl.lit(0, dtype=pl.Int32))
        .when((tod >= 571) & (tod <= 690))
        .then((tod - 570).cast(pl.Int32))
        .when((tod >= 781) & (tod <= 900))
        .then((120 + (tod - 780)).cast(pl.Int32))
        .otherwise(None)
    )
    bucket = (
        pl.when(idx == 0)
        .then(pl.lit(0, dtype=pl.Int32))
        .otherwise(((idx - 1) // k).cast(pl.Int32))
    )
    end_min = _bucket_end_minutes((pl.col("_bucket") + 1) * k)

    lf = (
        filtered.lazy()
        .select(
            [
                "ts_code",
                time_col,
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ]
        )
        .with_columns(
            idx.alias("_idx"),
            pl.col(time_col).cast(pl.Date).alias("_date"),
            bucket.alias("_bucket"),
        )
        .filter(pl.col("_idx").is_not_null())
        .group_by(["ts_code", "_date", "_bucket"])
        .agg(
            pl.col("open").get(pl.col(time_col).arg_min()).alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").get(pl.col(time_col).arg_max()).alias("close"),
            pl.col("vol").sum().cast(pl.Int64).alias("vol"),
            pl.col("amount").sum().cast(pl.Float64).alias("amount"),
        )
        .with_columns(
            (
                pl.col("_date").cast(pl.Datetime("us"))
                + pl.duration(minutes=end_min)
            ).alias(time_col)
        )
        .select(["ts_code", time_col, "open", "high", "low", "close", "vol", "amount"])
    )
    # streaming 在整月 ~24M 行 group_by 上更快；不支持时回退
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect()
    except pl.exceptions.InvalidOperationError:
        return lf.collect()
