"""Normalize heterogeneous A-share minute parquet files into the production lake."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_RAW
from factorzen.core.storage import save_parquet

MINUTE_DATA_TYPE = "minute_1min"
MINUTE_COLUMNS = (
    "ts_code",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)


@dataclass(frozen=True)
class MinuteIngestReport:
    """Observable outcome of one idempotent ingest run."""

    source_files: int
    rows_by_month: dict[str, int]

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_month.values())


def discover_parquet_files(source: Path | str) -> list[Path]:
    """Return deterministic parquet inputs from a file or recursive directory."""
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"分钟源不存在或不是文件/目录: {path}")
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise ValueError(f"分钟源目录没有 parquet 文件: {path}")
    return files


def _normalised_scan(files: list[Path]) -> pl.LazyFrame:
    scan = pl.scan_parquet([str(path) for path in files])
    schema = scan.collect_schema()
    code_col = "ts_code" if "ts_code" in schema else "code" if "code" in schema else None
    required = {"trade_time", "open", "high", "low", "close", "vol", "amount"}
    missing = sorted(required - set(schema.names()))
    if code_col is None:
        missing.insert(0, "ts_code|code")
    if missing:
        raise ValueError(f"分钟源缺少必要列: {', '.join(missing)}")
    assert code_col is not None  # guarded above; narrows the column name for type checking

    trade_dtype = schema["trade_time"]
    if trade_dtype == pl.String:
        trade_time = pl.col("trade_time").str.to_datetime(strict=False)
    else:
        trade_time = pl.col("trade_time").cast(pl.Datetime("us"), strict=False)

    floats = [
        pl.col(name).cast(pl.Float64, strict=False).fill_nan(None).alias(name)
        for name in ("open", "high", "low", "close", "amount")
    ]
    return (
        scan.select(
            pl.col(code_col).cast(pl.String).alias("ts_code"),
            trade_time.cast(pl.Datetime("us")).alias("trade_time"),
            *floats,
            pl.col("vol")
            .cast(pl.Float64, strict=False)
            .fill_nan(None)
            .round(0)
            .cast(pl.Int64, strict=False)
            .alias("vol"),
        )
        .select(MINUTE_COLUMNS)
        .filter(pl.col("ts_code").is_not_null() & pl.col("trade_time").is_not_null())
    )


def _validate_months(months: Iterable[str]) -> list[str]:
    values = sorted(set(months))
    for month in values:
        try:
            datetime.strptime(month, "%Y%m")
        except ValueError as exc:
            raise ValueError(f"非法月份 {month!r}，期望 YYYYMM") from exc
    return values


def _available_months(scan: pl.LazyFrame) -> list[str]:
    return sorted(
        scan.select(pl.col("trade_time").dt.strftime("%Y%m").alias("month"))
        .unique()
        .collect()["month"]
        .drop_nulls()
        .to_list()
    )


def ingest_minute_files(
    files: Iterable[Path | str],
    *,
    base_dir: Path = DATA_RAW,
    months: Iterable[str] | None = None,
) -> MinuteIngestReport:
    """Ingest either by-day or by-symbol files through one schema contract.

    Existing partitions are always merged by ``(trade_time, ts_code)``.  A partition's
    mere existence is never treated as proof of source coverage, so gap fills can repair
    partial months and repeated runs remain idempotent.
    """
    paths = sorted({Path(path) for path in files})
    if not paths:
        raise ValueError("至少需要一个分钟 parquet 源文件")
    missing_files = [str(path) for path in paths if not path.is_file()]
    if missing_files:
        raise ValueError(f"分钟源文件不存在: {missing_files[0]}")

    scan = _normalised_scan(paths)
    selected_months = (
        _validate_months(months) if months is not None else _available_months(scan)
    )
    rows_by_month: dict[str, int] = {}
    for month in selected_months:
        start = datetime.strptime(month, "%Y%m")
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
        frame = (
            scan.filter(
                (pl.col("trade_time") >= start) & (pl.col("trade_time") < end)
            )
            .sort(["trade_time", "ts_code"])
            .collect()
        )
        if frame.is_empty():
            continue
        save_parquet(
            frame,
            data_type=MINUTE_DATA_TYPE,
            date_col="trade_time",
            base_dir=base_dir,
            mode="append",
        )
        rows_by_month[month] = frame.height

    return MinuteIngestReport(source_files=len(paths), rows_by_month=rows_by_month)
