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

from factorzen.config.settings import DATA_RAW
from factorzen.core.calendar import get_trade_dates
from factorzen.core.loader import (
    DAILY_BASIC_COLS,
    DAILY_STD_COLS,
    _retry,
    init_tushare,
)
from factorzen.core.logger import get_logger
from factorzen.core.storage import load_parquet, save_parquet

logger = get_logger(__name__)


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


# 列集单一真源 = loader（全量拉取写湖的一方）。增量补拉若自带窄副本，
# 追加帧与既有分区列数不一致会在 concat 处直接炸（2026-07-23 daily_basic
# 11 列副本 vs 湖 17 列实锤），窄集静默写入还会丢 turnover 类字段。
DAILY_COLUMNS = DAILY_STD_COLS
ADJ_FACTOR_COLUMNS = ["ts_code", "trade_date", "adj_factor"]
DAILY_BASIC_COLUMNS = DAILY_BASIC_COLS
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
FETCH_SAVE_BATCH_SIZE = 20


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
        df = load_parquet(
            data_type, start=start, end=end, date_col=date_col, base_dir=base
        ).collect()
    except Exception:
        df = pl.DataFrame()

    if df.is_empty():
        present_dates: list[str] = []
        duplicate_key_count = 0
    else:
        df = _ensure_date_col(df, date_col)
        present_dates = sorted(
            df.select(pl.col(date_col).dt.strftime("%Y%m%d")).to_series().unique().to_list()
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
    logger.info(f"[data ensure] {data_type} 审计开始: {start}~{end}")
    before = audit_daily_like(data_type, start, end, base_dir=base_dir)
    if before.ok:
        _log_audit_result(data_type, before)
        return before

    logger.info(
        f"[data ensure] {data_type} 需要补齐: "
        f"missing={len(before.missing_dates)}, "
        f"range={min(before.missing_dates)}~{max(before.missing_dates)}"
    )
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
    _log_audit_result(data_type, after)
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
    logger.info(f"[data ensure] qlib provider 检查开始: uri={uri}, range={start}~{end}")
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
    logger.info(f"[data ensure] qlib provider 检查完成: ok={ok}, calendar={first}~{last}")
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
    required_label = ", ".join(required_data) if required_data else "<none>"
    logger.info(
        "[data ensure] 开始: "
        f"range={start}~{end}, required=[{required_label}], "
        f"universe={universe or '<none>'}, benchmark={benchmark or '<none>'}, "
        f"size_neutralization={needs_size_neutralization}, "
        f"qlib={is_qlib_factor}, strict={strict}"
    )

    if "daily" in required_data:
        logger.info("[data ensure] daily 检查/补齐开始")
        results["daily"] = ensure_daily(start, end, strict=strict)
        logger.info("[data ensure] adj_factor 检查/补齐开始")
        results["adj_factor"] = ensure_adj_factor(start, end, strict=strict)

    if "daily_basic" in required_data or needs_size_neutralization:
        logger.info("[data ensure] daily_basic 检查/补齐开始")
        results["daily_basic"] = ensure_daily_basic(start, end, strict=strict)

    if benchmark:
        logger.info(f"[data ensure] benchmark 检查/补齐开始: {benchmark}")
        results[f"index_daily:{benchmark}"] = ensure_index_daily(
            benchmark, start, end, strict=strict
        )

    if is_qlib_factor:
        results["qlib_provider"] = ensure_qlib_provider(start, end, strict=strict)

    # Universe construction currently fetches index members through common.universe.
    # Calling it here warms the cache and fails early for invalid universe names.
    if universe:
        from factorzen.core.universe import get_universe

        logger.info(f"[data ensure] 股票池缓存预热开始: universe={universe}, date={end}")
        warmed_universe = get_universe(end, universe)
        logger.info(
            f"[data ensure] 股票池缓存预热完成: universe={universe}, stocks={warmed_universe.height}"
        )

    logger.info(f"[data ensure] 完成: checked={list(results)}")
    return results


def _log_audit_result(label: str, result: DataAuditResult | bool) -> None:
    if isinstance(result, bool):
        logger.info(f"[data ensure] {label} 完成: ok={result}")
        return

    message = (
        f"[data ensure] {label} 完成: ok={result.ok}, "
        f"rows={result.row_count}, dates={len(result.present_dates)}/{len(result.expected_dates)}, "
        f"missing={len(result.missing_dates)}, duplicate_keys={result.duplicate_key_count}"
    )
    if result.ok:
        logger.info(message)
    else:
        logger.warning(message)


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
    logger.info(f"[data ensure] {data_type} 审计开始: {start}~{end}")
    before = audit_daily_like(data_type, start, end, base_dir=base_dir)
    if before.ok:
        _log_audit_result(data_type, before)
        return before

    if before.duplicate_key_count:
        logger.info(
            f"[data ensure] {data_type} 发现重复键，开始修复: "
            f"duplicate_keys={before.duplicate_key_count}"
        )
        _repair_duplicate_keys(
            data_type,
            start,
            end,
            base_dir=base_dir,
            key_cols=["trade_date", "ts_code"],
        )
        before = audit_daily_like(data_type, start, end, base_dir=base_dir)
        if before.ok:
            logger.info(f"[data ensure] {data_type} 重复键修复完成")
            _log_audit_result(data_type, before)
            return before

    parts: list[pl.DataFrame] = []
    if before.missing_dates:
        logger.info(
            f"[data ensure] {data_type} 需要补齐: missing={len(before.missing_dates)}, "
            f"range={before.missing_dates[0]}~{before.missing_dates[-1]}"
        )
        pro = init_tushare()
        api = api_getter(pro)
        total_missing = len(before.missing_dates)
        try:
            for idx, trade_date in enumerate(before.missing_dates, start=1):
                if _should_log_fetch_progress(idx, total_missing):
                    logger.info(
                        f"[data ensure] {data_type} 拉取进度: "
                        f"{idx}/{total_missing}, trade_date={trade_date}"
                    )
                df_pd = _retry(api, trade_date=trade_date)
                if df_pd is not None and not df_pd.empty:
                    parts.append(_normalize_tushare_frame(df_pd, standard_columns))
                if len(parts) >= FETCH_SAVE_BATCH_SIZE:
                    _flush_fetch_parts(data_type, parts, base_dir=base_dir)
        except Exception:
            _flush_fetch_parts(data_type, parts, base_dir=base_dir)
            raise

    _flush_fetch_parts(data_type, parts, base_dir=base_dir)

    after = audit_daily_like(data_type, start, end, base_dir=base_dir)
    _log_audit_result(data_type, after)
    if strict and not after.ok:
        raise DataEnsureError(_format_audit_failure(data_type, after))
    return after


def _should_log_fetch_progress(idx: int, total: int) -> bool:
    return total <= 20 or idx == 1 or idx == total or idx % 20 == 0


def _flush_fetch_parts(
    data_type: str,
    parts: list[pl.DataFrame],
    *,
    base_dir: Path | None,
) -> None:
    if not parts:
        return
    logger.info(f"[data ensure] {data_type} 写入新增分区: parts={len(parts)}")
    save_parquet(pl.concat(parts), data_type=data_type, base_dir=base_dir)
    parts.clear()


def _repair_duplicate_keys(
    data_type: str,
    start: str,
    end: str,
    *,
    base_dir: Path | None,
    key_cols: list[str],
) -> None:
    base = DATA_RAW if base_dir is None else base_dir
    root = base / data_type
    start_date = datetime.strptime(start, "%Y%m%d").date()
    end_date = datetime.strptime(end, "%Y%m%d").date()

    for path in root.glob("**/*.parquet"):
        df = pl.read_parquet(path)
        if not set(key_cols).issubset(df.columns):
            continue
        df = _ensure_date_col(df, key_cols[0])
        in_range = df.filter(
            (pl.col(key_cols[0]) >= start_date) & (pl.col(key_cols[0]) <= end_date)
        )
        if _duplicate_count(in_range, key_cols) == 0:
            continue

        logger.info(f"[data ensure] {data_type} 修复重复键文件: {path}")
        repaired = df.unique(subset=key_cols, keep="last", maintain_order=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            repaired.write_parquet(temp_path)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _format_audit_failure(data_type: str, result: DataAuditResult) -> str:
    if result.missing_dates:
        return (
            f"{data_type} still missing dates after refill: "
            f"{result.missing_dates[:10]}; "
            f"duplicate_key_count={result.duplicate_key_count}"
        )
    return (
        f"{data_type} still has duplicate keys after repair: "
        f"{result.duplicate_key_count}; missing_dates=[]"
    )


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
