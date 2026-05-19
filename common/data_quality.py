"""Daily research data quality checks."""

from __future__ import annotations

from typing import Any

import polars as pl


class QualityCheckError(RuntimeError):
    """Raised when a quality issue makes research output unreliable."""


def build_daily_quality_report(
    *,
    daily_df: pl.DataFrame,
    factor_df: pl.DataFrame,
    clean_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    universe_codes: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable daily research quality report.

    Fatal issues raise QualityCheckError. Non-fatal issues are returned as warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if daily_df.is_empty():
        errors.append("daily data is empty")
    if factor_df.is_empty():
        errors.append("factor data is empty")
    if clean_df.is_empty():
        errors.append("clean factor data is empty")
    if ret_df.is_empty():
        errors.append("forward return data is empty")

    _append_duplicate_error(errors, daily_df, "daily", ["trade_date", "ts_code"])
    _append_duplicate_error(errors, factor_df, "factor", ["trade_date", "ts_code"])
    _append_duplicate_error(errors, clean_df, "clean factor", ["trade_date", "ts_code"])
    _append_duplicate_error(errors, ret_df, "return", ["trade_date", "ts_code"])

    checks: dict[str, Any] = {
        "daily": _frame_stats(daily_df),
        "factor_value": _value_stats(factor_df, "factor_value"),
        "factor_clean": _value_stats(clean_df, "factor_clean"),
        "forward_return": _value_stats(ret_df, "fwd_ret_1d"),
        "universe": _universe_stats(daily_df, universe_codes),
    }

    if checks["factor_clean"]["valid_count"] == 0:
        errors.append("factor_clean has no valid values")
    if checks["forward_return"]["valid_count"] == 0:
        errors.append("fwd_ret_1d has no valid values")

    _append_low_coverage_warning(warnings, "factor_value", checks["factor_value"]["coverage"])
    _append_low_coverage_warning(warnings, "factor_clean", checks["factor_clean"]["coverage"])
    _append_low_coverage_warning(warnings, "fwd_ret_1d", checks["forward_return"]["coverage"])
    _append_low_coverage_warning(warnings, "universe", checks["universe"]["coverage"])

    _append_non_positive_price_warning(warnings, daily_df)
    _append_extreme_return_warning(warnings, ret_df)

    if errors:
        raise QualityCheckError("; ".join(errors))

    return {
        "status": "warning" if warnings else "ok",
        "checks": checks,
        "warnings": warnings,
        "errors": [],
    }


def _append_duplicate_error(
    errors: list[str], df: pl.DataFrame, label: str, keys: list[str]
) -> None:
    if df.is_empty() or not all(k in df.columns for k in keys):
        return
    duplicate_count = df.group_by(keys).len().filter(pl.col("len") > 1).height
    if duplicate_count > 0:
        errors.append(f"duplicate {label} keys: {duplicate_count}")


def _frame_stats(df: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": df.height,
        "columns": list(df.columns),
    }


def _value_stats(df: pl.DataFrame, col: str) -> dict[str, Any]:
    if df.is_empty() or col not in df.columns:
        return {
            "rows": df.height,
            "valid_count": 0,
            "null_count": df.height,
            "inf_count": 0,
            "coverage": 0.0,
        }

    valid = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite()).height
    inf_count = df.filter(pl.col(col).is_infinite()).height
    null_count = df[col].null_count()
    coverage = valid / df.height if df.height else 0.0
    return {
        "rows": df.height,
        "valid_count": valid,
        "null_count": null_count,
        "inf_count": inf_count,
        "coverage": coverage,
    }


def _universe_stats(df: pl.DataFrame, universe_codes: list[str]) -> dict[str, Any]:
    unique_universe = set(universe_codes)
    if not unique_universe:
        return {"size": 0, "covered": 0, "coverage": 0.0}
    covered = set(df["ts_code"].unique().to_list()) if "ts_code" in df.columns else set()
    covered_count = len(unique_universe & covered)
    return {
        "size": len(unique_universe),
        "covered": covered_count,
        "coverage": covered_count / len(unique_universe),
    }


def _append_low_coverage_warning(warnings: list[str], label: str, coverage: float) -> None:
    if coverage < 0.8:
        warnings.append(f"{label} coverage is low: {coverage:.1%}")


def _append_non_positive_price_warning(warnings: list[str], df: pl.DataFrame) -> None:
    price_cols = [c for c in ("open", "close", "pre_close") if c in df.columns]
    if not price_cols:
        return
    bad = df.filter(pl.any_horizontal([pl.col(c) <= 0 for c in price_cols])).height
    if bad > 0:
        warnings.append(f"non-positive price rows: {bad}")


def _append_extreme_return_warning(warnings: list[str], df: pl.DataFrame) -> None:
    if "fwd_ret_1d" not in df.columns:
        return
    extreme = df.filter(pl.col("fwd_ret_1d").abs() > 0.5).height
    if extreme > 0:
        warnings.append(f"extreme fwd_ret_1d rows: {extreme}")
