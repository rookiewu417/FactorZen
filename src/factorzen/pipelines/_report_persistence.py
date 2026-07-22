"""report 流程的产物持久化：meta / quality / 因子与评价 parquet。

``_save_results`` 落盘 bt/nav/positions/trades/turnover 等评价产物，
供下游审计与报告索引；不再提供 ``--reuse`` 缓存回读。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import (
    daily_factor_output_dir,
    daily_report_output_dir,
    daily_result_output_dir,
)
from factorzen.core.logger import get_logger
from factorzen.daily.evaluation.backtest import BacktestResult
from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.daily.evaluation.turnover import TurnoverResult

logger = get_logger(__name__)


def _meta_path(factor_name: str, start: str, end: str) -> Path:
    result_dir = daily_result_output_dir(factor_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{factor_name}_{start}_{end}_meta.json"


def _save_results(
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
) -> None:
    """落盘因子与评价产物（bt/nav/positions/trades/turnover + meta）。"""
    factor_dir = daily_factor_output_dir(factor_name)
    result_dir = daily_result_output_dir(factor_name)
    factor_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{factor_name}_{start}_{end}"

    clean_df.write_parquet(str(factor_dir / f"{prefix}.parquet"))
    ic_result.ic_series.write_parquet(str(result_dir / f"{prefix}_ic.parquet"))
    bt_result.returns.write_parquet(str(result_dir / f"{prefix}_bt_returns.parquet"))
    bt_result.nav.write_parquet(str(result_dir / f"{prefix}_bt_nav.parquet"))
    bt_result.positions.write_parquet(str(result_dir / f"{prefix}_bt_positions.parquet"))
    bt_result.trades.write_parquet(str(result_dir / f"{prefix}_bt_trades.parquet"))
    to_result.daily_turnover.write_parquet(str(result_dir / f"{prefix}_to_daily.parquet"))
    to_result.migration_matrix.write_parquet(str(result_dir / f"{prefix}_to_matrix.parquet"))

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
    }
    _meta_path(factor_name, start, end).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"中间结果已落盘: {result_dir / (prefix + '_*.parquet')}")


def _quality_path(factor_name: str, start: str, end: str) -> Path:
    result_dir = daily_result_output_dir(factor_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{factor_name}_{start}_{end}_quality.json"


def _save_quality_report(
    factor_name: str,
    start: str,
    end: str,
    report: dict,
) -> Path:
    path = _quality_path(factor_name, start, end)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_walk_forward_summary(factor_name: str, start: str, end: str) -> dict | None:
    mp = _meta_path(factor_name, start, end)
    if not mp.exists():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return meta.get("walk_forward_summary")


def _existing_report_outputs(factor_name: str, start: str, end: str) -> dict[str, str]:
    candidates = {
        "report": daily_report_output_dir(factor_name) / f"{factor_name}_{start}_{end}.html",
        "meta": _meta_path(factor_name, start, end),
        "quality_report": _quality_path(factor_name, start, end),
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}
