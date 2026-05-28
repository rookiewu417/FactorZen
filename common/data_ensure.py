"""Data cache audit and gap filling helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from common.calendar import get_trade_dates
from common.loader import _retry, init_tushare
from common.storage import load_parquet, save_parquet
from config.settings import DATA_RAW


class DataEnsureError(RuntimeError):
    """Raised when cache gaps remain after a refill attempt."""


@dataclass(frozen=True)
class DataAuditResult:
    data_type: str
    start: str
    end: str
    expected_dates: list[str]
    present_dates: list[str]
    missing_dates: list[str]
    duplicate_key_count: int = 0
    row_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing_dates and self.duplicate_key_count == 0


@dataclass(frozen=True)
class DataRequirement:
    data_type: str
    params: dict[str, str] = field(default_factory=dict)


DAILY_COLUMNS = [
    "trade_date",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
ADJ_FACTOR_COLUMNS = ["ts_code", "trade_date", "adj_factor"]
DAILY_BASIC_COLUMNS = [
    "trade_date",
    "ts_code",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]
INDEX_DAILY_COLUMNS = [
    "trade_date",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def audit_daily_like(
    data_type: str,
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    date_col: str = "trade_date",
    key_cols: list[str] | None = None,
) -> DataAuditResult:
    expected_dates = [_date_to_str(d) for d in get_trade_dates(start, end)]
    base = DATA_RAW if base_dir is None else base_dir
    root = base / data_type

    if not root.exists():
        return DataAuditResult(
            data_type=data_type,
            start=start,
            end=end,
            expected_dates=expected_dates,
            present_dates=[],
            missing_dates=expected_dates,
        )

    try:
        df = load_parquet(data_type, start=start, end=end, date_col=date_col, base_dir=base).collect()
    except Exception:
        df = pl.DataFrame()

    if df.is_empty():
        present_dates: list[str] = []
        duplicate_key_count = 0
    else:
        df = _ensure_date_col(df, date_col)
        present_dates = sorted(
            df.select(pl.col(date_col).dt.strftime("%Y%m%d"))
            .to_series()
            .unique()
            .to_list()
        )
        duplicate_key_count = _duplicate_count(df, key_cols or [date_col, "ts_code"])

    present_set = set(present_dates)
    missing_dates = [d for d in expected_dates if d not in present_set]
    return DataAuditResult(
        data_type=data_type,
        start=start,
        end=end,
        expected_dates=expected_dates,
        present_dates=present_dates,
        missing_dates=missing_dates,
        duplicate_key_count=duplicate_key_count,
        row_count=df.height if not df.is_empty() else 0,
    )


def ensure_daily(
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    strict: bool = True,
) -> DataAuditResult:
    return _ensure_by_trade_date(
        data_type="daily",
        start=start,
        end=end,
        api_getter=lambda pro: pro.daily,
        standard_columns=DAILY_COLUMNS,
        base_dir=base_dir,
        strict=strict,
    )


def ensure_adj_factor(
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    strict: bool = True,
) -> DataAuditResult:
    return _ensure_by_trade_date(
        data_type="adj_factor",
        start=start,
        end=end,
        api_getter=lambda pro: pro.adj_factor,
        standard_columns=ADJ_FACTOR_COLUMNS,
        base_dir=base_dir,
        strict=strict,
    )


def ensure_daily_basic(
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    strict: bool = True,
) -> DataAuditResult:
    return _ensure_by_trade_date(
        data_type="daily_basic",
        start=start,
        end=end,
        api_getter=lambda pro: pro.daily_basic,
        standard_columns=DAILY_BASIC_COLUMNS,
        base_dir=base_dir,
        strict=strict,
    )


def ensure_index_daily(
    index_code: str,
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    strict: bool = True,
) -> DataAuditResult:
    data_type = f"index_daily_{index_code.replace('.', '_')}"
    before = audit_daily_like(data_type, start, end, base_dir=base_dir)
    if before.ok:
        return before

    pro = init_tushare()
    api = pro.index_daily
    df_pd = _retry(
        api,
        ts_code=index_code,
        start_date=min(before.missing_dates),
        end_date=max(before.missing_dates),
    )
    if df_pd is not None and not df_pd.empty:
        save_parquet(
            _normalize_tushare_frame(df_pd, INDEX_DAILY_COLUMNS),
            data_type=data_type,
            base_dir=base_dir,
        )

    after = audit_daily_like(data_type, start, end, base_dir=base_dir)
    if strict and not after.ok:
        raise DataEnsureError(
            f"{data_type} still missing dates after refill: {after.missing_dates[:10]}"
        )
    return after


def ensure_qlib_provider(
    start: str,
    end: str,
    *,
    provider_uri: str | None = None,
    strict: bool = True,
) -> bool:
    uri = provider_uri or os.getenv(
        "QLIB_PROVIDER_URI", str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    )
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=uri, region=REG_CN)
    cal = D.calendar(freq="day")
    if len(cal) == 0:
        if strict:
            raise DataEnsureError(f"qlib provider has empty calendar: {uri}")
        return False

    first = cal[0].strftime("%Y%m%d")
    last = cal[-1].strftime("%Y%m%d")
    ok = first <= start and last >= end
    if strict and not ok:
        raise DataEnsureError(
            f"qlib provider calendar {first}~{last} does not cover requested {start}~{end}"
        )
    return ok


def ensure_data_for_daily_run(
    *,
    required_data: list[str],
    start: str,
    end: str,
    universe: str | None = None,
    benchmark: str | None = None,
    needs_size_neutralization: bool = False,
    is_qlib_factor: bool = False,
    strict: bool = True,
) -> dict[str, DataAuditResult | bool]:
    results: dict[str, DataAuditResult | bool] = {}

    if "daily" in required_data:
        results["daily"] = ensure_daily(start, end, strict=strict)
        results["adj_factor"] = ensure_adj_factor(start, end, strict=strict)

    if "daily_basic" in required_data or needs_size_neutralization:
        results["daily_basic"] = ensure_daily_basic(start, end, strict=strict)

    if benchmark:
        results[f"index_daily:{benchmark}"] = ensure_index_daily(
            benchmark, start, end, strict=strict
        )

    if is_qlib_factor:
        results["qlib_provider"] = ensure_qlib_provider(start, end, strict=strict)

    # Universe construction currently fetches index members through common.universe.
    # Calling it here warms the cache and fails early for invalid universe names.
    if universe:
        from common.universe import get_universe

        get_universe(end, universe)

    return results


def _ensure_by_trade_date(
    *,
    data_type: str,
    start: str,
    end: str,
    api_getter: Callable[[Any], Any],
    standard_columns: list[str],
    base_dir: Path | None,
    strict: bool,
) -> DataAuditResult:
    before = audit_daily_like(data_type, start, end, base_dir=base_dir)
    if before.ok:
        return before

    pro = init_tushare()
    api = api_getter(pro)
    parts: list[pl.DataFrame] = []
    for trade_date in before.missing_dates:
        df_pd = _retry(api, trade_date=trade_date)
        if df_pd is not None and not df_pd.empty:
            parts.append(_normalize_tushare_frame(df_pd, standard_columns))

    if parts:
        save_parquet(pl.concat(parts), data_type=data_type, base_dir=base_dir)

    after = audit_daily_like(data_type, start, end, base_dir=base_dir)
    if strict and not after.ok:
        raise DataEnsureError(
            f"{data_type} still missing dates after refill: {after.missing_dates[:10]}"
        )
    return after


def _normalize_tushare_frame(df_pd: pd.DataFrame, standard_columns: list[str]) -> pl.DataFrame:
    df = pl.from_pandas(df_pd)
    if "trade_date" in df.columns and df.schema["trade_date"] == pl.Utf8:
        df = df.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
    return df.select([c for c in standard_columns if c in df.columns])


def _ensure_date_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    dtype = df.schema[col]
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(col).dt.date().alias(col))
    if dtype == pl.Utf8:
        return df.with_columns(pl.col(col).str.strptime(pl.Date, "%Y%m%d", strict=False))
    return df


def _duplicate_count(df: pl.DataFrame, key_cols: list[str]) -> int:
    existing_keys = [c for c in key_cols if c in df.columns]
    if len(existing_keys) != len(key_cols):
        return 0
    duplicate_rows = df.group_by(existing_keys).len().filter(pl.col("len") > 1)
    return duplicate_rows.height


def _date_to_str(value: date | datetime) -> str:
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%Y%m%d")
