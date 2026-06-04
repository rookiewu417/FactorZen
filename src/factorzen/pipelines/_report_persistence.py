"""report 流程的产物持久化与回读：meta / quality / 因子与评价 parquet。"""

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
from factorzen.daily.evaluation.backtest import (
    BacktestResult,
    trim_backtest_to_first_trade,
)
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
    """Persist factor and evaluation artifacts for reuse."""
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


def _load_results(
    factor_name: str, start: str, end: str
) -> tuple[pl.DataFrame, ICAnalysisResult, BacktestResult, TurnoverResult] | None:
    """从磁盘加载已有的评价结果。若文件不存在返回 None。"""
    mp = _meta_path(factor_name, start, end)
    if not mp.exists():
        return None

    prefix = f"{factor_name}_{start}_{end}"
    factor_dir = daily_factor_output_dir(factor_name)
    result_dir = daily_result_output_dir(factor_name)
    ic_path = result_dir / f"{prefix}_ic.parquet"
    bt_ret_path = result_dir / f"{prefix}_bt_returns.parquet"
    bt_nav_path = result_dir / f"{prefix}_bt_nav.parquet"
    bt_pos_path = result_dir / f"{prefix}_bt_positions.parquet"
    bt_trades_path = result_dir / f"{prefix}_bt_trades.parquet"
    to_daily_path = result_dir / f"{prefix}_to_daily.parquet"
    to_mat_path = result_dir / f"{prefix}_to_matrix.parquet"
    factor_path = factor_dir / f"{prefix}.parquet"

    for p in [
        ic_path,
        bt_ret_path,
        bt_nav_path,
        bt_pos_path,
        bt_trades_path,
        to_daily_path,
        to_mat_path,
        factor_path,
    ]:
        if not p.exists():
            logger.warning(f"--reuse: 缺少文件 {p.name}，退回重新计算")
            return None

    meta = json.loads(mp.read_text(encoding="utf-8"))

    clean_df = pl.read_parquet(str(factor_path))
    ic_result = ICAnalysisResult(
        factor_name=meta["factor_name"],
        ic_mean=meta["ic_mean"],
        ic_std=meta["ic_std"],
        ir=meta["ir"],
        ic_positive_ratio=meta["ic_positive_ratio"],
        n_periods=meta["n_periods"],
        ic_series=pl.read_parquet(str(ic_path)),
        decay={int(k): v for k, v in meta["decay"].items()},
        frequency=meta["frequency"],
        ic_tstat=meta["ic_tstat"],
        ic_pvalue=meta["ic_pvalue"],
        multi_period={int(k): v for k, v in meta["multi_period"].items()},
        oos_ic=meta["oos_ic"],
    )
    bt_result = BacktestResult(
        factor_name=meta["bt_factor_name"],
        strategy_name=meta.get("bt_strategy_name", "quantile_long_short"),
        n_groups=meta["bt_n_groups"],
        returns=pl.read_parquet(str(bt_ret_path)),
        nav=pl.read_parquet(str(bt_nav_path)),
        positions=pl.read_parquet(str(bt_pos_path)),
        trades=pl.read_parquet(str(bt_trades_path)),
        summary_stats={
            (int(k) if k.isdigit() else k): v for k, v in meta["bt_summary_stats"].items()
        },
        config=meta.get("bt_config", {}),
        frequency=meta["bt_frequency"],
        ret_definition=meta.get("bt_ret_definition", "open_to_close_with_overnight_carry"),
    )
    bt_result = trim_backtest_to_first_trade(bt_result)
    to_result = TurnoverResult(
        factor_name=meta["to_factor_name"],
        avg_turnover=meta["to_avg_turnover"],
        migration_matrix=pl.read_parquet(str(to_mat_path)),
        daily_turnover=pl.read_parquet(str(to_daily_path)),
        frequency=meta["to_frequency"],
    )
    logger.info(f"--reuse: 从磁盘加载 {prefix} 评价结果")
    return clean_df, ic_result, bt_result, to_result


def _existing_report_outputs(factor_name: str, start: str, end: str) -> dict[str, str]:
    candidates = {
        "report": daily_report_output_dir(factor_name) / f"{factor_name}_{start}_{end}.html",
        "meta": _meta_path(factor_name, start, end),
        "quality_report": _quality_path(factor_name, start, end),
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}
