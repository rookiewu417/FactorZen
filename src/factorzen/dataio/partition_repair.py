"""Repair a partitioned raw dataset from a legacy snapshot without overwrites."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_RAW
from factorzen.core.storage import save_parquet


@dataclass(frozen=True)
class PartitionRepairReport:
    source_rows: int
    merged_rows: int
    rows_by_year: dict[int, int]
    min_date: object | None
    max_date: object | None
    n_symbols: int


def _parquet_glob(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"分区源目录不存在: {root}")
    if not any(root.rglob("*.parquet")):
        raise ValueError(f"分区源目录没有 parquet: {root}")
    return str(root / "**" / "*.parquet")


def merge_missing_partition_rows(
    source_root: Path | str,
    *,
    target_data_type: str,
    base_dir: Path = DATA_RAW,
    key_cols: Sequence[str] = ("trade_date", "ts_code"),
    date_col: str = "trade_date",
) -> PartitionRepairReport:
    """Append only source keys absent from the target dataset.

    The target schema owns the interface.  Legacy sources may omit newer fields; they
    are filled with typed nulls.  Existing target keys always win, so a stale backup
    cannot overwrite corrected or newly fetched values.
    """
    source_path = Path(source_root)
    target_root = base_dir / target_data_type
    source = pl.scan_parquet(_parquet_glob(source_path))
    target = pl.scan_parquet(_parquet_glob(target_root))
    source_schema = source.collect_schema()
    target_schema = target.collect_schema()

    required = set(key_cols) | {date_col}
    missing_source = sorted(required - set(source_schema.names()))
    missing_target = sorted(required - set(target_schema.names()))
    if missing_source or missing_target:
        raise ValueError(
            "修复键/日期列缺失: "
            f"source={missing_source or 'ok'}, target={missing_target or 'ok'}"
        )

    additions = [
        pl.lit(None, dtype=target_schema[name]).alias(name)
        for name in target_schema.names()
        if name not in source_schema
    ]
    aligned = source.with_columns(additions).select(target_schema.names())
    target_keys = target.select(key_cols).unique()
    missing = (
        aligned.join(target_keys, on=key_cols, how="anti")
        .unique(subset=key_cols, keep="last")
        .collect()
    )
    source_rows = source.select(pl.len()).collect().item()
    if missing.is_empty():
        return PartitionRepairReport(source_rows, 0, {}, None, None, 0)

    save_parquet(
        missing,
        data_type=target_data_type,
        date_col=date_col,
        base_dir=base_dir,
        mode="append",
    )
    by_year = (
        missing.with_columns(pl.col(date_col).dt.year().alias("year"))
        .group_by("year")
        .len()
        .sort("year")
    )
    return PartitionRepairReport(
        source_rows=source_rows,
        merged_rows=missing.height,
        rows_by_year={int(row["year"]): int(row["len"]) for row in by_year.to_dicts()},
        min_date=missing[date_col].min(),
        max_date=missing[date_col].max(),
        n_symbols=missing["ts_code"].n_unique() if "ts_code" in missing.columns else 0,
    )
