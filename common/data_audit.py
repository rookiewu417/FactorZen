"""原始数据分区完整性审计。

与 data_quality.py 的区别：
- data_quality.py  审计单次 run 级别的数据质量（流经因子管线的数据）
- data_audit.py    审计 data/raw/ 分区的横向完整性（日期缺口、股票覆盖、字段空值率）

不调用 Tushare，只读本地落盘数据，可在无网络/无 token 时运行。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from common.calendar import get_trade_dates
from common.storage import load_parquet

_DAILY_BASIC_KEY_FIELDS = ["pe", "pb", "total_mv", "circ_mv"]
_FINANCE_KEY_FIELDS = ["revenue", "n_income", "total_assets", "total_equity", "roe"]

# finance ann_date 超过此天数视为陈旧（约 18 个月，覆盖 4 个季报周期）
_FINANCE_STALENESS_DAYS = 548

_SUPPORTED_TYPES = ("daily", "daily_basic", "finance")


def build_raw_data_audit(
    *,
    data_type: str,
    start: str,
    end: str,
    universe_codes: list[str] | None = None,
) -> dict[str, Any]:
    """审计 data/raw/ 分区的横向完整性。

    Args:
        data_type: "daily" | "daily_basic" | "finance"
        start:     起始日期 "YYYYMMDD"
        end:       截止日期 "YYYYMMDD"
        universe_codes: 目标股票池，用于覆盖率计算；None 时跳过覆盖率检查

    Returns:
        与 build_daily_quality_report 格式对齐的 dict：
        {status: "ok"|"warning"|"error", checks: {...}, warnings: [...], errors: [...]}
    """
    errors: list[str] = []
    warnings: list[str] = []

    if data_type not in _SUPPORTED_TYPES:
        errors.append(f"unsupported data_type '{data_type}'; expected one of {_SUPPORTED_TYPES}")
        return {"status": "error", "checks": {}, "warnings": warnings, "errors": errors}

    date_col = "end_date" if data_type == "finance" else "trade_date"

    try:
        df = load_parquet(data_type, start=start, end=end, date_col=date_col).collect()
    except Exception as exc:
        errors.append(f"failed to load {data_type} partition: {exc}")
        return {"status": "error", "checks": {}, "warnings": warnings, "errors": errors}

    if df.is_empty():
        errors.append(f"{data_type} partition is empty for {start}~{end}")
        return {"status": "error", "checks": {}, "warnings": warnings, "errors": errors}

    checks: dict[str, Any] = {
        "data_type": data_type,
        "period": {"start": start, "end": end},
        "total_rows": df.height,
    }

    _check_date_coverage(df, date_col, data_type, start, end, checks, warnings)
    _check_stock_coverage(df, universe_codes, data_type, checks, warnings)
    _check_field_nulls(df, data_type, checks, warnings)

    status = "error" if errors else ("warning" if warnings else "ok")
    return {"status": status, "checks": checks, "warnings": warnings, "errors": errors}


# ── 内部检查函数 ───────────────────────────────────────────────────────────


def _check_date_coverage(
    df: pl.DataFrame,
    date_col: str,
    data_type: str,
    start: str,
    end: str,
    checks: dict[str, Any],
    warnings: list[str],
) -> None:
    if data_type == "finance":
        checks["date_coverage"] = {"unique_end_dates": df[date_col].n_unique()}
        return

    expected: set[date] = set(get_trade_dates(start, end))
    actual: set[date] = set(df[date_col].to_list())
    missing = sorted(expected - actual)

    checks["date_coverage"] = {
        "expected": len(expected),
        "actual": len(actual),
        "missing_count": len(missing),
        "missing_dates": [d.strftime("%Y%m%d") for d in missing[:20]],
    }
    if missing:
        warnings.append(f"{data_type}: {len(missing)} missing trade dates")


def _check_stock_coverage(
    df: pl.DataFrame,
    universe_codes: list[str] | None,
    data_type: str,
    checks: dict[str, Any],
    warnings: list[str],
) -> None:
    if "ts_code" not in df.columns:
        return

    actual_codes: set[str] = set(df["ts_code"].unique().to_list())

    if universe_codes is None:
        checks["stock_coverage"] = {"actual_codes": len(actual_codes)}
        return

    universe_set = set(universe_codes)
    covered = actual_codes & universe_set
    coverage = len(covered) / len(universe_set) if universe_set else 0.0
    missing_sample = sorted(universe_set - actual_codes)[:10]

    checks["stock_coverage"] = {
        "universe_size": len(universe_set),
        "actual_codes": len(actual_codes),
        "covered": len(covered),
        "coverage": coverage,
        "missing_codes_sample": missing_sample,
    }
    if coverage < 0.9:
        warnings.append(f"{data_type}: stock coverage {coverage:.1%} < 90%")


def _check_field_nulls(
    df: pl.DataFrame,
    data_type: str,
    checks: dict[str, Any],
    warnings: list[str],
) -> None:
    if data_type == "daily_basic":
        key_fields = _DAILY_BASIC_KEY_FIELDS
        low_threshold = 0.8
    elif data_type == "finance":
        key_fields = _FINANCE_KEY_FIELDS
        low_threshold = 0.7
    else:
        return

    field_nulls: dict[str, Any] = {}
    for col in key_fields:
        if col not in df.columns:
            field_nulls[col] = {"missing_column": True}
            warnings.append(f"{data_type}: expected column '{col}' not found")
            continue
        null_count = df[col].null_count()
        coverage = 1.0 - null_count / df.height if df.height else 0.0
        field_nulls[col] = {"null_count": null_count, "coverage": coverage}
        if coverage < low_threshold:
            warnings.append(f"{data_type}.{col}: coverage {coverage:.1%} < {low_threshold:.0%}")

    checks["field_null_rates"] = field_nulls

    if data_type == "finance":
        _check_finance_pit(df, checks, warnings)


def _check_finance_pit(
    df: pl.DataFrame,
    checks: dict[str, Any],
    warnings: list[str],
) -> None:
    """检查每只股票最近一期 ann_date 是否陈旧。"""
    if "ann_date" not in df.columns or "ts_code" not in df.columns:
        return

    threshold_date = (datetime.now() - timedelta(days=_FINANCE_STALENESS_DAYS)).strftime("%Y%m%d")

    df_ann = df.select(["ts_code", "ann_date"]).filter(pl.col("ann_date").is_not_null())
    if df_ann.is_empty():
        checks["pit_staleness"] = {"threshold_date": threshold_date, "stale_count": 0, "stale_sample": []}
        return

    ann_col = df_ann["ann_date"]
    if ann_col.dtype == pl.Date:
        latest_df = (
            df_ann.group_by("ts_code")
            .agg(pl.col("ann_date").max().alias("latest_ann"))
            .with_columns(pl.col("latest_ann").dt.strftime("%Y%m%d"))
        )
    else:
        latest_df = (
            df_ann.group_by("ts_code")
            .agg(pl.col("ann_date").cast(pl.Utf8).max().alias("latest_ann"))
        )

    stale = latest_df.filter(pl.col("latest_ann") < threshold_date)
    stale_list = stale["ts_code"].to_list()

    checks["pit_staleness"] = {
        "threshold_date": threshold_date,
        "stale_count": len(stale_list),
        "stale_sample": stale_list[:10],
    }
    if stale_list:
        warnings.append(
            f"finance: {len(stale_list)} stocks have stale ann_date (>{_FINANCE_STALENESS_DAYS}d old)"
        )
