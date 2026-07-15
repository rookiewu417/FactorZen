"""A 股分钟 bar 口径审计（纯函数层，不做 IO；输入 DataFrame 便于离线单测）。

分层对齐 ``factorzen.core.data_audit``：census / 闭合核对 / 标签推断 / 覆盖报告。
覆盖报告是唯一涉及分区扫描的入口；其余函数仅操作内存帧。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import DATA_RAW_MINUTE

_REL_TOL: float = 1e-4


def _rel_match(left: pl.Expr, right: pl.Expr, *, tol: float = _REL_TOL) -> pl.Expr:
    """相对容差匹配：|a-b| / max(|a|,|b|) <= tol；两侧均为 0 视为匹配。"""
    denom = pl.max_horizontal(left.abs(), right.abs())
    return (
        pl.when((left == 0) & (right == 0))
        .then(pl.lit(True))
        .when(denom == 0)
        .then(pl.lit(False))
        .otherwise((left - right).abs() / denom <= tol)
    )


def timestamp_census(
    minute: pl.DataFrame,
    *,
    time_col: str = "trade_time",
) -> pl.DataFrame:
    """按 (year, board) 做时间戳普查。

    board = ``ts_code`` 前 3 字符（如 000 / 600 / 688 / 300 / 920）。

    输出列：year, board, n_rows, n_distinct_times, min_time, max_time,
    n_after_1500, vol_after_1500。
    """
    if minute.is_empty():
        return pl.DataFrame(
            schema={
                "year": pl.Int32,
                "board": pl.String,
                "n_rows": pl.UInt32,
                "n_distinct_times": pl.UInt32,
                "min_time": pl.Time,
                "max_time": pl.Time,
                "n_after_1500": pl.UInt32,
                "vol_after_1500": pl.Int64,
            }
        )

    # dt.hour/minute 返回 Int8，必须 cast 再乘，避免 i8 溢出
    tod_minutes = (
        pl.col(time_col).dt.hour().cast(pl.Int32) * 60
        + pl.col(time_col).dt.minute().cast(pl.Int32)
    )
    after = tod_minutes > 900  # > 15:00

    return (
        minute.with_columns(
            pl.col(time_col).dt.year().cast(pl.Int32).alias("year"),
            pl.col("ts_code").str.slice(0, 3).alias("board"),
            pl.col(time_col).dt.time().alias("_tod"),
            after.alias("_after"),
        )
        .group_by(["year", "board"])
        .agg(
            pl.len().alias("n_rows"),
            pl.col(time_col).n_unique().alias("n_distinct_times"),
            pl.col("_tod").min().alias("min_time"),
            pl.col("_tod").max().alias("max_time"),
            pl.col("_after").sum().cast(pl.UInt32).alias("n_after_1500"),
            pl.when(pl.col("_after"))
            .then(pl.col("vol"))
            .otherwise(pl.lit(0, dtype=pl.Int64))
            .sum()
            .cast(pl.Int64)
            .alias("vol_after_1500"),
        )
        .sort(["year", "board"])
    )


def reconcile_with_daily(minute: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """逐 (ts_code, trade_date) 核对分钟汇总 vs 日线。

    输入：
    - minute: 含 ts_code, trade_time, open/high/low/close, vol, amount
    - daily: 含 trade_date(Date), ts_code, open/high/low/close, vol, amount

    输出：minute_vol_sum(≤15:00)、daily_vol、vol_multiplier、amount_multiplier 同理、
    open_match / close_match / high_match / low_match（相对容差 1e-4）。

    预期健康值：vol_multiplier≈100、amount_multiplier≈1000。
    """
    empty_schema = {
        "ts_code": pl.String,
        "trade_date": pl.Date,
        "minute_vol_sum": pl.Int64,
        "daily_vol": pl.Int64,
        "vol_multiplier": pl.Float64,
        "minute_amount_sum": pl.Float64,
        "daily_amount": pl.Float64,
        "amount_multiplier": pl.Float64,
        "open_match": pl.Boolean,
        "close_match": pl.Boolean,
        "high_match": pl.Boolean,
        "low_match": pl.Boolean,
    }
    if minute.is_empty() or daily.is_empty():
        return pl.DataFrame(schema=empty_schema)

    # 只计 ≤15:00（tod minutes <= 900）；hour/minute 先 cast 防 i8 溢出
    tod = (
        pl.col("trade_time").dt.hour().cast(pl.Int32) * 60
        + pl.col("trade_time").dt.minute().cast(pl.Int32)
    )
    m = (
        minute.filter(tod <= 900)
        .sort(["ts_code", "trade_time"])
        .with_columns(pl.col("trade_time").dt.date().alias("trade_date"))
    )

    is_1500 = (pl.col("trade_time").dt.hour().cast(pl.Int32) == 15) & (
        pl.col("trade_time").dt.minute().cast(pl.Int32) == 0
    )

    m_agg = m.group_by(["ts_code", "trade_date"]).agg(
        pl.col("vol").sum().cast(pl.Int64).alias("minute_vol_sum"),
        pl.col("amount").sum().cast(pl.Float64).alias("minute_amount_sum"),
        pl.col("open").first().alias("minute_open"),
        pl.col("close").filter(is_1500).first().alias("minute_close_1500"),
        pl.col("high").max().alias("minute_high"),
        pl.col("low").min().alias("minute_low"),
    )

    d = daily.select(
        pl.col("ts_code").cast(pl.String),
        pl.col("trade_date").cast(pl.Date),
        pl.col("open").cast(pl.Float64).alias("daily_open"),
        pl.col("high").cast(pl.Float64).alias("daily_high"),
        pl.col("low").cast(pl.Float64).alias("daily_low"),
        pl.col("close").cast(pl.Float64).alias("daily_close"),
        pl.col("vol").cast(pl.Int64).alias("daily_vol"),
        pl.col("amount").cast(pl.Float64).alias("daily_amount"),
    )

    joined = m_agg.join(d, on=["ts_code", "trade_date"], how="inner")

    if joined.is_empty():
        return pl.DataFrame(schema=empty_schema)

    return (
        joined.with_columns(
            (pl.col("minute_vol_sum") / pl.col("daily_vol")).alias("vol_multiplier"),
            (pl.col("minute_amount_sum") / pl.col("daily_amount")).alias(
                "amount_multiplier"
            ),
            _rel_match(pl.col("minute_open"), pl.col("daily_open")).alias("open_match"),
            _rel_match(pl.col("minute_close_1500"), pl.col("daily_close")).alias(
                "close_match"
            ),
            _rel_match(pl.col("minute_high"), pl.col("daily_high")).alias("high_match"),
            _rel_match(pl.col("minute_low"), pl.col("daily_low")).alias("low_match"),
        )
        .select(
            "ts_code",
            "trade_date",
            "minute_vol_sum",
            "daily_vol",
            "vol_multiplier",
            "minute_amount_sum",
            "daily_amount",
            "amount_multiplier",
            "open_match",
            "close_match",
            "high_match",
            "low_match",
        )
        .sort(["ts_code", "trade_date"])
    )


def infer_label_convention(
    minute: pl.DataFrame,
    *,
    time_col: str = "trade_time",
) -> dict[str, object]:
    """从时间戳分布推断 bar 标签约定。

    - has_1130 且无 1300 → ``"end"``
    - 有 1300 且无 1130 → ``"start"``
    - 否则 ``"ambiguous"``

    返回 ``label_convention / first_time / last_time / has_0930 / has_after_1500``。
    """
    if minute.is_empty() or time_col not in minute.columns:
        return {
            "label_convention": "ambiguous",
            "first_time": None,
            "last_time": None,
            "has_0930": False,
            "has_after_1500": False,
        }

    tod = (
        minute.select(
            (
                pl.col(time_col).dt.hour().cast(pl.Int32) * 60
                + pl.col(time_col).dt.minute().cast(pl.Int32)
            ).alias("_tod")
        )["_tod"]
        .unique()
        .to_list()
    )
    tod_set = set(int(x) for x in tod if x is not None)

    has_1130 = 690 in tod_set  # 11:30
    has_1300 = 780 in tod_set  # 13:00
    has_0930 = 570 in tod_set  # 09:30
    has_after_1500 = any(t > 900 for t in tod_set)

    if has_1130 and not has_1300:
        label = "end"
    elif has_1300 and not has_1130:
        label = "start"
    else:
        label = "ambiguous"

    first_t = min(tod_set) if tod_set else None
    last_t = max(tod_set) if tod_set else None

    def _fmt(mins: int | None) -> str | None:
        if mins is None:
            return None
        return f"{mins // 60:02d}:{mins % 60:02d}"

    return {
        "label_convention": label,
        "first_time": _fmt(first_t),
        "last_time": _fmt(last_t),
        "has_0930": has_0930,
        "has_after_1500": has_after_1500,
    }


def _month_iter(start: date, end: date) -> list[tuple[int, int]]:
    """返回 [start, end] 覆盖的 (year, month) 列表。"""
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return months


def _merge_missing_ranges(
    expected: list[date], present: set[date]
) -> list[list[str]]:
    """把 expected 中连续缺失的交易日合并成 [start_iso, end_iso] 区间。"""
    missing = [d for d in expected if d not in present]
    if not missing:
        return []

    missing_set = set(missing)
    ranges: list[list[str]] = []
    i = 0
    while i < len(expected):
        if expected[i] not in missing_set:
            i += 1
            continue
        start_d = expected[i]
        end_d = expected[i]
        while i < len(expected) and expected[i] in missing_set:
            end_d = expected[i]
            i += 1
        ranges.append([start_d.isoformat(), end_d.isoformat()])
    return ranges


def coverage_report(
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    trade_dates: list[date] | None = None,
) -> dict[str, Any]:
    """对交易日历审计分钟湖覆盖。

    ``trade_dates=None`` 时用 ``core.calendar.get_trade_dates``；
    测试可注入 ``trade_dates`` 离线跑。

    逐月惰性扫描（``pl.scan_parquet`` 单月分区文件），只 collect 聚合
    （distinct dates、行数、股票数），**绝不 collect 原始大帧**。
    """
    base = DATA_RAW_MINUTE if base_dir is None else Path(base_dir)
    start_d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()

    if trade_dates is None:
        from factorzen.core.calendar import get_trade_dates

        expected = get_trade_dates(start, end)
    else:
        expected = sorted(d for d in trade_dates if start_d <= d <= end_d)

    present_dates: set[date] = set()
    months_present: list[str] = []
    per_month_rows: dict[str, int] = {}

    for year, month in _month_iter(start_d, end_d):
        part = base / f"year={year}" / f"month={month:02d}" / "data.parquet"
        if not part.exists():
            continue

        month_key = f"{year}-{month:02d}"
        # 惰性聚合：只 collect 汇总
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)

        win_lo = max(start_d, month_start)
        win_hi = min(end_d, month_end)

        lf = pl.scan_parquet(str(part)).filter(
            (pl.col("trade_time").dt.date() >= win_lo)
            & (pl.col("trade_time").dt.date() <= win_hi)
        )
        # 分组聚合：只 materialize 汇总，不 collect 原始大帧
        agg = (
            lf.with_columns(pl.col("trade_time").dt.date().alias("_d"))
            .group_by(pl.lit(1).alias("_g"))
            .agg(
                pl.len().alias("n_rows"),
                pl.col("ts_code").n_unique().alias("n_codes"),
                pl.col("_d").unique().alias("dates"),
            )
            .collect()
        )
        if agg.is_empty():
            continue

        n_rows = int(agg["n_rows"][0])
        if n_rows <= 0:
            continue

        months_present.append(month_key)
        per_month_rows[month_key] = n_rows
        dates_list = agg["dates"][0]
        if dates_list is not None:
            for d in dates_list:
                if isinstance(d, date):
                    present_dates.add(d)

    expected_set = set(expected)
    present_in_window = present_dates & expected_set
    missing_ranges = _merge_missing_ranges(expected, present_in_window)

    return {
        "start": start,
        "end": end,
        "n_expected_days": len(expected),
        "n_present_days": len(present_in_window),
        "missing_ranges": missing_ranges,
        "months_present": months_present,
        "per_month_rows": per_month_rows,
    }
