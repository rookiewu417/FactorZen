"""report 流程的产物持久化：meta / quality。

评估 run（``factors/<market>/<name>/evaluations/{run_id}/``）**不写任何 parquet**。
因子数值面板唯一落点：``factors/<market>/<name>/factor.parquet``（4 列）；
本模块**不再**覆盖写 store 面板，仅记录已有路径。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.core.logger import get_logger
from factorzen.daily.evaluation.backtest import BacktestResult
from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.daily.evaluation.turnover import TurnoverResult
from factorzen.experiments.run_paths import artifact_path

logger = get_logger(__name__)


def _meta_path(run_dir: Path) -> Path:
    return artifact_path(run_dir, "meta")


def _quality_path(run_dir: Path) -> Path:
    return artifact_path(run_dir, "quality_report")


def _existing_store_panel_path(
    factor_name: str, *, market: str = "ashare"
) -> str | None:
    """若 factors 下已有该因子 store parquet，返回路径，否则 None。"""
    try:
        from factorzen.discovery.factor_store import DEFAULT_ROOT, asset_dir

        path = asset_dir(market, factor_name, root=DEFAULT_ROOT) / "factor.parquet"
        return str(path) if path.exists() else None
    except Exception:
        return None


def _save_results(
    run_dir: Path,
    factor_name: str,
    start: str,
    end: str,
    clean_df: pl.DataFrame,
    ic_result: ICAnalysisResult,
    bt_result: BacktestResult,
    to_result: TurnoverResult,
    quality_report: dict | None = None,
    quality_path: Path | None = None,
    walk_forward_summary: dict | None = None,
    backtest_direction: dict[str, Any] | None = None,
    *,
    market: str = "ashare",
) -> None:
    """写 meta/quality 到 run_dir；不覆盖写 store 面板。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    # clean_df 保留形参以兼容调用方；评估不再用子集 clobber store
    _ = clean_df
    store_panel_path = _existing_store_panel_path(factor_name, market=market)

    meta = {
        "factor_name": ic_result.factor_name,
        "frequency": ic_result.frequency,
        "ic_mean": ic_result.ic_mean,
        "ic_std": ic_result.ic_std,
        "ir": ic_result.ir,
        "ic_positive_ratio": ic_result.ic_positive_ratio,
        "n_periods": ic_result.n_periods,
        "ic_tstat": ic_result.ic_tstat,
        "ic_pvalue": ic_result.ic_pvalue,
        "decay": {str(k): v for k, v in ic_result.decay.items()},
        "multi_period": {str(k): v for k, v in ic_result.multi_period.items()},
        "oos_ic": ic_result.oos_ic,
        "bt_factor_name": bt_result.factor_name,
        "bt_strategy_name": bt_result.strategy_name,
        "bt_n_groups": bt_result.n_groups,
        "bt_summary_stats": {str(k): v for k, v in bt_result.summary_stats.items()},
        "bt_frequency": bt_result.frequency,
        "bt_config": bt_result.config,
        "bt_ret_definition": bt_result.ret_definition,
        "to_factor_name": to_result.factor_name,
        "to_avg_turnover": to_result.avg_turnover,
        "to_frequency": to_result.frequency,
        "quality_status": (quality_report or {}).get("status"),
        "quality_warnings": (quality_report or {}).get("warnings", []),
        "quality_report_path": str(quality_path) if quality_path is not None else None,
        "walk_forward_summary": walk_forward_summary or {"status": "not_run", "n_folds": 0},
        "backtest_direction": backtest_direction
        or {"direction": "normal", "should_reverse": False, "reason": "未记录"},
        "start": start,
        "end": end,
        "factor_name_arg": factor_name,
        "store_panel": store_panel_path,
    }
    _meta_path(run_dir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"meta/quality 已落盘 run_dir={run_dir}; store_panel={store_panel_path}")


def _save_quality_report(
    run_dir: Path,
    report: dict,
) -> Path:
    path = _quality_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_walk_forward_summary(run_dir: Path) -> dict | None:
    mp = _meta_path(run_dir)
    if not mp.exists():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return meta.get("walk_forward_summary")


def _existing_report_outputs(run_dir: Path) -> dict[str, str]:
    candidates = {
        "report": artifact_path(run_dir, "report"),
        "meta": _meta_path(run_dir),
        "quality_report": _quality_path(run_dir),
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}
