#!/usr/bin/env python
"""因子 Tear Sheet 报告生成器。

整合因子计算、基础评价、高级评价与 HTML 报告输出。

用法:
  pixi run report -- --factor momentum_20d --start 20250101 --end 20250513
  pixi run report -- --factor momentum_20d --start 20250101 --end 20250513 --reuse
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import polars as pl

from factorzen.config.settings import (
    daily_factor_output_dir,
    daily_report_output_dir,
    daily_result_output_dir,
)
from factorzen.core.calendar import get_trade_dates
from factorzen.core.config_loader import (
    RunConfig,
    build_backtest_strategies,
    build_cost_model,
    build_preprocessing_pipeline,
    build_runtime_backtest_config,
    default_benchmark_for_universe,
    load_run_config,
    with_default_all_strategies,
)
from factorzen.core.data_quality import QualityCheckError, build_daily_quality_report
from factorzen.core.experiment import record_experiment_output, run_experiment
from factorzen.core.loader import fetch_daily
from factorzen.core.logger import get_logger, setup_logging
from factorzen.core.storage import load_parquet
from factorzen.core.universe import get_universe
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.evaluation.backtest import (
    BacktestResult,
    run_strategy_backtest,
    trim_backtest_to_first_trade,
)
from factorzen.daily.evaluation.ic_analysis import (
    BothIcResult,
    ICAnalysisResult,
    IcStats,
    compute_fwd_returns,
    compute_rank_ic,
)
from factorzen.daily.evaluation.turnover import TurnoverResult, compute_turnover
from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary
from factorzen.daily.factors.registry import get_factor
from factorzen.experiments.run_paths import copy_outputs_to_run_dir
from factorzen.llm import generate_llm_explanation
from factorzen.reports.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)

def _merge_report_config_args(args: argparse.Namespace, run_config: RunConfig | None):
    """Merge YAML config into report CLI args without overriding explicit CLI values."""
    cli_benchmark = args.benchmark is not None
    cli_ic_method = getattr(args, "ic_method", None) is not None
    cli_neutralized_ic = getattr(args, "neutralized_ic", None) is not None
    cli_event_study = getattr(args, "event_study", None) is not None

    if run_config is not None:
        for field in ("factor", "start", "end", "universe"):
            if getattr(args, field, None) is None:
                setattr(args, field, getattr(run_config, field))
        if args.benchmark is None and run_config.benchmark is not None:
            args.benchmark = run_config.benchmark
        if getattr(args, "ic_method", None) is None:
            args.ic_method = run_config.ic_method
        if getattr(args, "neutralized_ic", None) is None:
            args.neutralized_ic = run_config.neutralized_ic
        if getattr(args, "event_study", None) is None:
            args.event_study = run_config.event_study

    if args.universe is None:
        args.universe = "csi300"
    if args.benchmark is None:
        args.benchmark = default_benchmark_for_universe(args.universe)

    if getattr(args, "all", False):
        if not cli_benchmark:
            args.benchmark = default_benchmark_for_universe(args.universe)
        if not cli_ic_method:
            args.ic_method = "both"
        if not cli_neutralized_ic:
            args.neutralized_ic = True
        if not cli_event_study:
            args.event_study = True
        args.llm_explain = True

    if getattr(args, "ic_method", None) is None:
        args.ic_method = "rank"
    if getattr(args, "neutralized_ic", None) is None:
        args.neutralized_ic = False
    if getattr(args, "event_study", None) is None:
        args.event_study = False

    missing = [field for field in ("factor", "start", "end") if getattr(args, field, None) is None]
    if missing:
        raise ValueError(f"缺少必填参数: {', '.join(missing)}（可通过 CLI 或 --config 提供）")
    return args


def _effective_report_config(args: argparse.Namespace, run_config: RunConfig | None) -> RunConfig:
    base = run_config or RunConfig(factor=args.factor, start=args.start, end=args.end)
    config = base.model_copy(
        update={
            "factor": args.factor,
            "start": args.start,
            "end": args.end,
            "universe": args.universe,
            "benchmark": args.benchmark or base.benchmark,
            "ic_method": args.ic_method,
            "neutralized_ic": args.neutralized_ic,
            "event_study": args.event_study,
        }
    )
    if run_config is None:
        config = with_default_all_strategies(config)
    return config


def _decide_backtest_direction(ic_result: ICAnalysisResult) -> dict[str, Any]:
    """Decide whether the report backtest should invert factor direction."""
    ic_mean = float(getattr(ic_result, "ic_mean", 0.0) or 0.0)
    ic_tstat = float(getattr(ic_result, "ic_tstat", 0.0) or 0.0)
    ic_pvalue = float(getattr(ic_result, "ic_pvalue", 1.0) or 1.0)
    oos_ic = getattr(ic_result, "oos_ic", {}) or {}
    oos_train = oos_ic.get("train")
    oos_test = oos_ic.get("test")
    oos_both_negative = (
        oos_train is not None and oos_test is not None and oos_train < 0 and oos_test < 0
    )

    statistically_negative = ic_mean < 0 and ic_pvalue <= 0.10
    if statistically_negative or (ic_mean < 0 and oos_both_negative):
        reason = (
            "IC 均值为负且 p 值小于等于 0.10"
            if statistically_negative
            else "IC 均值为负，且历史观察期/未来验证期 IC 均为负"
        )
        return {
            "direction": "reversed",
            "should_reverse": True,
            "reason": reason,
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "ic_pvalue": ic_pvalue,
            "oos_train_ic": oos_train,
            "oos_test_ic": oos_test,
        }

    reason = (
        "IC 均值为负，但显著性或样本外一致性不足，保持原方向"
        if ic_mean < 0
        else "IC 均值非负，保持原方向"
    )
    return {
        "direction": "normal",
        "should_reverse": False,
        "reason": reason,
        "ic_mean": ic_mean,
        "ic_tstat": ic_tstat,
        "ic_pvalue": ic_pvalue,
        "oos_train_ic": oos_train,
        "oos_test_ic": oos_test,
    }


def _apply_backtest_direction(
    clean_df: pl.DataFrame, decision: dict[str, Any] | None
) -> pl.DataFrame:
    """Flip factor_clean for backtesting when the IC decision requires reverse direction."""
    if not decision or decision.get("direction") != "reversed":
        return clean_df
    return clean_df.with_columns((-pl.col("factor_clean")).alias("factor_clean"))


def _load_backtest_direction(factor_name: str, start: str, end: str) -> dict[str, Any] | None:
    mp = _meta_path(factor_name, start, end)
    if not mp.exists():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    direction = meta.get("backtest_direction")
    return direction if isinstance(direction, dict) else None


# ---------------------------------------------------------------------------
# 持久化辅助
# ---------------------------------------------------------------------------


def _meta_path(factor_name: str, start: str, end: str) -> "Path":
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
) -> "tuple[pl.DataFrame, ICAnalysisResult, BacktestResult, TurnoverResult] | None":
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


def _run_backtest_strategies(
    config: RunConfig,
    clean_df: pl.DataFrame,
    daily: pl.DataFrame,
    *,
    factor_name: str,
    frequency: str,
) -> tuple[BacktestResult, dict[str, BacktestResult]]:
    strategy_results: dict[str, BacktestResult] = {}
    specs = {spec.name: spec for spec in config.backtest.strategy_specs}
    for strategy_name, strategy in build_backtest_strategies(config).items():
        spec = specs[strategy_name]
        result = run_strategy_backtest(
            strategy,
            clean_df,
            daily,
            config=build_runtime_backtest_config(
                config,
                factor_col="factor_clean",
                frequency=frequency,
                strategy_spec=spec,
            ),
            cost_model=build_cost_model(config, spec),
            factor_name=factor_name,
        )
        result = trim_backtest_to_first_trade(result)
        strategy_results[strategy_name] = result
        logger.info(f"\n{result.summary()}")

    primary_name = config.backtest.primary or next(iter(strategy_results))
    return strategy_results[primary_name], strategy_results


# ---------------------------------------------------------------------------
# 高级评价
# ---------------------------------------------------------------------------


def _run_advanced_evaluation(clean_df, ret_df, frequency, start: str = "", end: str = ""):
    """运行高级评价模块，各模块互不依赖，单个失败不影响整体。"""
    advanced: dict = {}

    try:
        from factorzen.daily.evaluation.advanced import compute_ic_decay

        advanced["decay_results"] = compute_ic_decay(clean_df, ret_df, factor_col="factor_clean")
        logger.info(f"IC Decay: {len(advanced['decay_results'])} horizons")
    except ImportError as e:
        logger.warning(f"IC Decay 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"IC Decay 失败: {e}")

    try:
        from factorzen.daily.evaluation.advanced import compute_monotonicity

        mono_df = clean_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        advanced["mono"] = compute_monotonicity(
            mono_df, factor_col="factor_clean", ret_col="fwd_ret_1d"
        )
        logger.info(f"单调性: score={advanced['mono'].monotonicity_score:.3f}")
    except ImportError as e:
        logger.warning(f"Monotonicity 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"单调性分析失败: {e}")

    try:
        from factorzen.daily.evaluation.advanced import compute_rank_autocorr

        advanced["autocorr"] = compute_rank_autocorr(
            clean_df, factor_col="factor_clean", lags=[1, 5, 10]
        )
        logger.info(
            f"排名自相关: mean={advanced['autocorr'].mean_autocorr:.3f}, "
            f"half_life={advanced['autocorr'].half_life_est:.1f}"
        )
    except ImportError as e:
        logger.warning(f"Rank Autocorr 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"排名自相关失败: {e}")

    try:
        from factorzen.daily.evaluation.advanced import compute_market_regime_ic

        advanced["regime"] = compute_market_regime_ic(
            clean_df.join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"],
                how="inner",
            ),
            factor_col="factor_clean",
            ret_col="fwd_ret_1d",
            regime_type="direction",
            return_object=True,
        )
        logger.info(f"市场状态 IC: {advanced['regime'].regime_type}")
    except ImportError as e:
        logger.warning(f"Market Regime 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"市场状态 IC 失败: {e}")

    # ── 行业分层 IC ──
    try:
        from factorzen.core.loader import fetch_stock_basic
        from factorzen.daily.evaluation.advanced import compute_sector_ic

        stock_basic = (
            fetch_stock_basic().select(["ts_code", "industry"]).rename({"industry": "sector"})
        )
        sector_df = (
            clean_df.join(stock_basic, on="ts_code", how="left")
            .join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"],
                how="inner",
            )
            .filter(pl.col("sector").is_not_null() & (pl.col("sector") != ""))
        )
        if not sector_df.is_empty():
            advanced["sector"] = compute_sector_ic(
                sector_df,
                factor_col="factor_clean",
                ret_col="fwd_ret_1d",
                sector_col="sector",
                return_object=True,
            )
            logger.info(f"行业 IC: {advanced['sector'].sector_ic_df.height} 个行业")
    except Exception as e:
        logger.warning(f"行业分层 IC 失败: {e}")

    # ── 市值分层 IC ──
    try:
        from factorzen.daily.evaluation.advanced import compute_size_ic

        if start and end:
            db = load_parquet("daily_basic", start=start, end=end).collect()
        else:
            db = load_parquet("daily_basic").collect()
        if db.is_empty() and start and end:
            db = load_parquet("daily_basic").collect()
        cap_df = (
            clean_df.join(
                db.select(["trade_date", "ts_code", "total_mv"]),
                on=["trade_date", "ts_code"],
                how="left",
            )
            .join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"],
                how="inner",
            )
            .filter(pl.col("total_mv").is_not_null())
        )
        if not cap_df.is_empty():
            advanced["size"] = compute_size_ic(
                cap_df,
                factor_col="factor_clean",
                ret_col="fwd_ret_1d",
                cap_col="total_mv",
                n_buckets=3,
                return_object=True,
            )
            logger.info(f"市值分层 IC: {advanced['size'].buckets}")
    except Exception as e:
        logger.warning(f"市值分层 IC 失败: {e}")

    return advanced if advanced else None


def _build_report_deep_results(
    clean_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    args: argparse.Namespace,
) -> tuple[Any | None, Any | None, Any | None]:
    """Compute optional deep report outputs controlled by report CLI flags."""
    pearson_ic_result: IcStats | None = None
    neutralized_ic_result = None
    event_study_result = None

    if args.ic_method in ("pearson", "both"):
        try:
            from factorzen.daily.evaluation.ic_analysis import compute_ic

            merged_simple = clean_df.join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
                    {"fwd_ret_1d": "ret_1d"}
                ),
                on=["trade_date", "ts_code"],
                how="inner",
            )
            if args.ic_method == "both":
                both_ic = compute_ic(
                    merged_simple,
                    factor_col="factor_clean",
                    ret_col="ret_1d",
                    method="both",
                )
                # method="both" 返回 BothIcResult(TypedDict)；取 pearson 分量为 IcStats
                pearson_ic_result = cast(BothIcResult, both_ic)["pearson"]
            else:
                pearson_ic_result = cast(
                    IcStats,
                    compute_ic(
                        merged_simple,
                        factor_col="factor_clean",
                        ret_col="ret_1d",
                        method="pearson",
                    ),
                )
            logger.info(
                f"Pearson IC Mean: {pearson_ic_result.ic_mean:.4f}, "
                f"IR: {pearson_ic_result.ir:.2f}"
            )
        except Exception as e:
            logger.warning(f"Pearson IC 计算失败（跳过）: {e}")

    if args.neutralized_ic:
        try:
            from factorzen.daily.evaluation.advanced import compute_neutralized_ic

            neutral_frame = clean_df.join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
                    {"fwd_ret_1d": "ret_1d"}
                ),
                on=["trade_date", "ts_code"],
                how="inner",
            )
            universe = get_universe(args.end, args.universe)
            if not universe.is_empty() and {"ts_code", "industry"}.issubset(
                set(universe.columns)
            ):
                neutral_frame = neutral_frame.join(
                    universe.select(["ts_code", "industry"]).unique(subset=["ts_code"]),
                    on="ts_code",
                    how="left",
                )
            try:
                daily_basic = load_parquet("daily_basic", start=args.start, end=args.end).collect()
                if not daily_basic.is_empty() and "total_mv" in daily_basic.columns:
                    neutral_frame = neutral_frame.join(
                        daily_basic.select(["trade_date", "ts_code", "total_mv"]),
                        on=["trade_date", "ts_code"],
                        how="left",
                    )
            except Exception as e:
                logger.warning(
                    f"daily_basic 缓存加载失败，中性化 IC 将使用可用暴露: {e}"
                )
            neutralized_ic_result = compute_neutralized_ic(neutral_frame, ret_col="ret_1d")
            logger.info(f"Neutralized IC Mean: {neutralized_ic_result.ic_mean:.4f}")
        except Exception as e:
            logger.warning(f"中性化 IC 计算失败（跳过）: {e}")

    if args.event_study:
        try:
            from factorzen.daily.evaluation.advanced import compute_event_study

            factor_simple = clean_df.select(["trade_date", "ts_code", "factor_clean"])
            ret_simple = ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
                {"fwd_ret_1d": "ret_1d"}
            )
            event_study_result = compute_event_study(factor_simple, ret_simple)
            logger.info(f"事件研究完成: {event_study_result.n_events} 个事件")
        except Exception as e:
            logger.warning(f"事件研究计算失败（跳过）: {e}")

    return pearson_ic_result, neutralized_ic_result, event_study_result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _existing_report_outputs(factor_name: str, start: str, end: str) -> dict[str, str]:
    candidates = {
        "report": daily_report_output_dir(factor_name) / f"{factor_name}_{start}_{end}.html",
        "meta": _meta_path(factor_name, start, end),
        "quality_report": _quality_path(factor_name, start, end),
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}


def _run(args: argparse.Namespace, effective_config: RunConfig) -> dict[str, str]:
    logger.info(f"──── 因子报告生成: {args.factor} | {args.start} ~ {args.end} ────")

    # ── 1. 获取因子类 ──
    try:
        factor_cls = get_factor(args.factor)
    except KeyError as e:
        logger.error(str(e))
        raise RuntimeError(f"unknown factor: {args.factor}") from e
    factor = factor_cls()
    logger.info(f"因子: {factor.name} | {factor.description}")

    walk_forward_summary: dict | None = None
    walk_forward_result = None
    backtest_direction: dict[str, Any] | None = None
    pearson_ic_result = None
    neutralized_ic_result = None
    event_study_result = None
    strategy_results: dict[str, BacktestResult] | None = None

    # ── --reuse 路径 ──
    reused = None
    if args.reuse:
        reused = _load_results(args.factor, args.start, args.end)
        if reused is not None:
            saved_direction = _load_backtest_direction(args.factor, args.start, args.end)
            if saved_direction is not None:
                backtest_direction = saved_direction
            else:
                decision = _decide_backtest_direction(reused[1])
                if decision["should_reverse"]:
                    logger.info("--reuse: 缓存缺少反向回测方向记录，退回完整计算")
                    reused = None
                else:
                    backtest_direction = decision

    if reused is not None:
        clean_df, ic_result, bt_result, to_result = reused
        walk_forward_summary = _load_walk_forward_summary(args.factor, args.start, args.end)
        # 高级评价仍需 ret_df，重新从存储加载（快速路径：只读收盘价）
        try:
            fetch_daily(args.start, args.end)
        except Exception as e:
            logger.warning(f"数据拉取失败（高级评价可能跳过）: {e}")
        daily = load_parquet("daily", start=args.start, end=args.end).collect()
        if not daily.is_empty():
            ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(
                ["ts_code", "trade_date"]
            )
            ret_df = ret_df.with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
            )
            ret_df = compute_fwd_returns(ret_df, ret_col="ret")
            if len(effective_config.backtest.strategy_specs) > 1:
                backtest_df = _apply_backtest_direction(clean_df, backtest_direction)
                bt_result, strategy_results = _run_backtest_strategies(
                    effective_config,
                    backtest_df,
                    daily,
                    factor_name=factor.name,
                    frequency=args.frequency,
                )
            else:
                strategy_results = {bt_result.strategy_name: bt_result}
            advanced_results = _run_advanced_evaluation(
                clean_df, ret_df, args.frequency, args.start, args.end
            )
            pearson_ic_result, neutralized_ic_result, event_study_result = (
                _build_report_deep_results(clean_df, ret_df, args)
            )
        else:
            strategy_results = {bt_result.strategy_name: bt_result}
            logger.warning("日线数据为空，跳过高级评价")
            advanced_results = None
    else:
        if args.reuse:
            logger.info("--reuse: 未找到缓存，退回完整计算")

        # ── 2. 准备数据 ──
        trade_dates = get_trade_dates(args.start, args.end)
        logger.info(f"交易日数: {len(trade_dates)}")
        if len(trade_dates) < 30:
            logger.warning("交易日不足 30 天，IC 分析可能不稳定")

        try:
            fetch_daily(args.start, args.end)
        except Exception as e:
            logger.error(f"数据拉取失败: {e}")
            raise RuntimeError(f"fetch_daily failed: {e}") from e

        # ── 3. 股票池 ──
        universe = get_universe(args.end, args.universe)
        if universe.is_empty():
            logger.error(f"股票池为空: {args.universe} ({args.end})")
            raise RuntimeError(f"empty universe: {args.universe} ({args.end})")
        ts_codes = universe["ts_code"].to_list()
        logger.info(f"股票池: {len(ts_codes)} 只")

        # ── 4. 计算因子 ──
        ctx = FactorDataContext(
            start=args.start,
            end=args.end,
            required_data=factor.required_data,
            lookback_days=factor.lookback_days,
            universe=ts_codes,
            snapshot_mode=args.frequency,
        )
        try:
            factor_df = factor.compute(ctx)
        except Exception as e:
            logger.error(f"因子计算失败: {e}")
            raise RuntimeError(f"factor compute failed: {e}") from e

        validation = factor.validate(factor_df)
        logger.info(f"因子计算完成: {validation}")
        if factor_df.is_empty():
            logger.error("因子计算结果为空，退出")
            raise RuntimeError("empty factor result")
        if validation.get("coverage", 0) < 0.5:
            logger.warning("因子覆盖率不足 50%，结果可能不可靠")

        # ── 5. 预处理 ──
        clean_df = build_preprocessing_pipeline(effective_config).run(factor_df, col="factor_value")
        logger.info("预处理完成 (去极值 → 填充 → 标准化)")

        # ── 6. 前向收益 ──
        daily = load_parquet("daily", start=args.start, end=args.end).collect()
        if daily.is_empty():
            logger.error("日线数据为空，无法计算收益")
            raise RuntimeError("empty daily data")
        ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(["ts_code", "trade_date"])
        ret_df = ret_df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
        )
        ret_df = compute_fwd_returns(ret_df, ret_col="ret")
        logger.info("前向收益计算完成 (horizons: 1/5/10/20d)")

        try:
            quality_report = build_daily_quality_report(
                daily_df=daily,
                factor_df=factor_df,
                clean_df=clean_df,
                ret_df=ret_df,
                universe_codes=ts_codes,
            )
        except QualityCheckError as e:
            logger.error(f"数据质量检查失败: {e}")
            raise RuntimeError(f"quality check failed: {e}") from e
        quality_report_path = _save_quality_report(
            factor.name, args.start, args.end, quality_report
        )
        if quality_report["warnings"]:
            logger.warning(f"数据质量警告: {quality_report['warnings']}")
        logger.info(f"数据质量报告已保存: {quality_report_path}")

        # ── 7. IC 分析 ──
        ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
        ic_result.factor_name = factor.name
        logger.info(f"\n{ic_result.summary()}")
        backtest_direction = _decide_backtest_direction(ic_result)
        logger.info(f"回测方向判定: {backtest_direction['reason']}")
        backtest_df = _apply_backtest_direction(clean_df, backtest_direction)

        # ── 8. 策略回测 ──
        bt_result, strategy_results = _run_backtest_strategies(
            effective_config,
            backtest_df,
            daily,
            factor_name=factor.name,
            frequency=args.frequency,
        )

        # ── 9. 换手率 ──
        to_result = compute_turnover(backtest_df, frequency=args.frequency)
        to_result.factor_name = factor.name
        logger.info(f"\n{to_result.summary()}")

        # ── 10. Walk-forward / OOS 摘要 ──
        try:
            walk_forward_summary, walk_forward_result = run_quantile_walk_forward_summary(
                backtest_df,
                daily,
                effective_config,
                factor_name=factor.name,
                frequency=args.frequency,
            )
            logger.info(f"Walk-forward 摘要: {walk_forward_summary}")
        except Exception as e:
            walk_forward_summary = {"status": "error", "n_folds": 0, "error": str(e)}
            walk_forward_result = None
            logger.warning(f"Walk-forward 计算失败（跳过）: {e}")

        # ── 11. 高级评价 ──
        advanced_results = _run_advanced_evaluation(clean_df, ret_df, args.frequency)
        pearson_ic_result, neutralized_ic_result, event_study_result = (
            _build_report_deep_results(clean_df, ret_df, args)
        )

        # ── 持久化中间结果 ──
        _save_results(
            args.factor,
            args.start,
            args.end,
            clean_df,
            ic_result,
            bt_result,
            to_result,
            quality_report=quality_report,
            quality_path=quality_report_path,
            walk_forward_summary=walk_forward_summary,
            backtest_direction=backtest_direction,
        )

    # ── (Optional) Benchmark 对比 ──
    benchmark_result = None
    if args.benchmark:
        try:
            from factorzen.daily.evaluation.benchmark import compute_excess_return

            benchmark_result = compute_excess_return(
                bt_result.returns, args.benchmark, args.start, args.end
            )
            logger.info(f"Benchmark: {benchmark_result.summary()}")
        except Exception as e:
            logger.warning(f"Benchmark 计算失败（跳过）: {e}")

    # ── 11. 生成 HTML 报告 ──
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    quality_report_for_llm: dict[str, Any] | None = None
    quality_report_path = _quality_path(args.factor, args.start, args.end)
    if quality_report_path.exists():
        try:
            quality_report_for_llm = json.loads(quality_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            quality_report_for_llm = None
    llm_explanation, llm_explanation_path = generate_llm_explanation(
        enabled=args.llm_explain,
        refresh=args.llm_refresh,
        cache_dir=daily_result_output_dir(factor.name),
        factor_name=factor.name,
        factor_description=getattr(factor, "description", ""),
        start=args.start,
        end=args.end,
        frequency=args.frequency,
        date_range=date_range,
        universe=args.universe,
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        walk_forward_summary=walk_forward_summary,
        quality_report=quality_report_for_llm,
        backtest_direction=backtest_direction,
    )
    html = generate_tear_sheet(
        factor_name=factor.name,
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        frequency=args.frequency,
        date_range=date_range,
        advanced_results=advanced_results,
        universe=args.universe,
        benchmark_result=benchmark_result,
        attribution_result=None,  # Brinson requires index constituent data; deferred
        backtest_direction=backtest_direction,
        walk_forward_result=walk_forward_result,
        walk_forward_summary=walk_forward_summary,
        event_study_result=event_study_result,
        pearson_ic_result=pearson_ic_result if args.ic_method in ("pearson", "both") else None,
        neutralized_ic_result=neutralized_ic_result if args.neutralized_ic else None,
        llm_explanation=llm_explanation.to_dict() if llm_explanation is not None else None,
        strategy_results=strategy_results,
        primary_strategy=effective_config.backtest.primary,
        quality_report=quality_report_for_llm,
    )

    # ── 12. 落盘 HTML ──
    report_dir = daily_report_output_dir(factor.name)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{factor.name}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"报告已生成: {report_path}")

    outputs = {
        "report": str(report_path),
        "meta": str(_meta_path(args.factor, args.start, args.end)),
    }
    quality_report_path = _quality_path(args.factor, args.start, args.end)
    if quality_report_path.exists():
        outputs["quality_report"] = str(quality_report_path)
    if llm_explanation_path is not None:
        outputs["llm_explanation"] = str(llm_explanation_path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="因子 Tear Sheet 报告生成")
    parser.add_argument("--factor", default=None, help="因子名称")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default=None, help="股票池")
    parser.add_argument(
        "--frequency", default="daily", choices=["daily", "weekly", "monthly"], help="因子频率"
    )
    parser.add_argument(
        "--reuse", action="store_true", help="复用已有 parquet 结果，跳过重新计算（需先跑过一次）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "启用报告深度预设：reuse、both IC、中性化 IC、事件研究、"
            "按 universe 匹配 benchmark、LLM 解读"
        ),
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="基准指数代码（如 000300.SH），若指定则计算超额收益与 benchmark 对比",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML 运行配置文件路径")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC 计算方法：rank（Spearman，默认）/ pearson / both",
    )
    parser.add_argument(
        "--neutralized-ic",
        action="store_true",
        dest="neutralized_ic",
        default=None,
        help="是否计算中性化后的 Rank IC",
    )
    parser.add_argument(
        "--event-study",
        action="store_true",
        dest="event_study",
        default=None,
        help="是否执行事件研究分析（选 Top 5%% 分位股票为事件）",
    )
    parser.add_argument(
        "--llm-explain",
        action="store_true",
        help="显式启用大模型因子解读；默认关闭，缺少 FACTORZEN_LLM_* 配置时跳过",
    )
    parser.add_argument(
        "--llm-refresh",
        action="store_true",
        help="启用 --llm-explain 时忽略已有 LLM 解读缓存并重新生成",
    )
    args = parser.parse_args()

    run_config = load_run_config(args.config) if args.config else None
    try:
        args = _merge_report_config_args(args, run_config)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)
    effective_config = _effective_report_config(args, run_config)

    try:
        with run_experiment(effective_config, command=sys.argv) as exp_dir:
            try:
                outputs = _run(args, effective_config)
            except Exception:
                for name, path in _existing_report_outputs(args.factor, args.start, args.end).items():
                    record_experiment_output(exp_dir, name, path)
                raise
            for name, path in outputs.items():
                record_experiment_output(exp_dir, name, path)
            for name, path in copy_outputs_to_run_dir(outputs, exp_dir).items():
                record_experiment_output(exp_dir, name, path)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Done.")


if __name__ == "__main__":
    main()
