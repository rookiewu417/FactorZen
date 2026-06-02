"""Build compact factor result snapshots for low-token LLM prompts."""

from __future__ import annotations

from typing import Any


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is not None and hasattr(obj, attr):
        return getattr(obj, attr)
    return default


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _compact_metric_map(data: dict[Any, Any] | None) -> dict[str, Any]:
    if not data:
        return {}
    compact: dict[str, Any] = {}
    for key, value in data.items():
        label = f"{key}d" if isinstance(key, int) else str(key)
        if isinstance(value, dict):
            compact[label] = {
                str(k): _safe_float(v) for k, v in value.items() if _safe_float(v) is not None
            }
        else:
            compact[label] = _safe_float(value)
    return compact


def build_factor_snapshot(
    *,
    factor_name: str,
    factor_description: str | None,
    frequency: str,
    date_range: str,
    universe: str,
    ic_result: Any,
    bt_result: Any,
    to_result: Any,
    walk_forward_summary: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
    backtest_direction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a small, JSON-serializable summary for LLM input."""

    bt_stats = _safe_attr(bt_result, "summary_stats", {}) or {}
    ls_stats = bt_stats.get("long_short", {}) if isinstance(bt_stats, dict) else {}
    oos_ic = _safe_attr(ic_result, "oos_ic", {}) or {}
    direction = backtest_direction or {}

    return {
        "factor": {
            "name": factor_name,
            "description": factor_description or "",
            "frequency": frequency,
        },
        "sample": {
            "date_range": date_range,
            "universe": universe,
            "n_periods": _safe_attr(ic_result, "n_periods", 0) or 0,
        },
        "ic": {
            "mean": _safe_float(_safe_attr(ic_result, "ic_mean")),
            "std": _safe_float(_safe_attr(ic_result, "ic_std")),
            "ir": _safe_float(_safe_attr(ic_result, "ir")),
            "positive_ratio": _safe_float(_safe_attr(ic_result, "ic_positive_ratio")),
            "tstat": _safe_float(_safe_attr(ic_result, "ic_tstat")),
            "pvalue": _safe_float(_safe_attr(ic_result, "ic_pvalue")),
        },
        "multi_period": _compact_metric_map(_safe_attr(ic_result, "multi_period", {}))
        or _compact_metric_map(_safe_attr(ic_result, "decay", {})),
        "oos": {
            "train_ic": _safe_float(oos_ic.get("train")),
            "test_ic": _safe_float(oos_ic.get("test")),
        },
        "backtest": {
            "strategy": _safe_attr(bt_result, "strategy_name", ""),
            "ls_ann_ret": _safe_float(ls_stats.get("ann_ret")),
            "ls_sharpe": _safe_float(ls_stats.get("sharpe")),
            "ls_max_dd": _safe_float(ls_stats.get("max_dd")),
        },
        "turnover": {
            "avg_turnover": _safe_float(_safe_attr(to_result, "avg_turnover")),
        },
        "walk_forward": {
            k: v
            for k, v in (walk_forward_summary or {}).items()
            if k in {"status", "n_folds", "oos_sharpe_mean", "oos_return_mean", "error"}
        },
        "quality": {
            "warnings": list((quality_report or {}).get("warnings", []))[:5],
        },
        "direction": {
            "reversed": direction.get("direction") == "reversed",
            "reason": direction.get("reason", ""),
        },
    }
