#!/usr/bin/env python
"""因子交易轨报告生成器（历史模块，CLI 入口已移除）。

.. deprecated::
    `fz report build` 与 `pixi run report` 已删除——因子交易轨报告统一由
    `fz factor backtest`（daily_single track=backtest，与本模块共用
    generate_trading_report 渲染层）产出。本模块暂留仅因其独有测试覆盖
    （ST 涨跌停对齐 / 结果持久化 / 负 IC 反向方向判定守卫），待后续清理；
    勿新增对它的调用。

整合因子计算、基础评价、高级评价与交易轨 HTML 报告输出。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.research import RunConfig, load_run_config
from factorzen.config.settings import daily_report_output_dir
from factorzen.core.calendar import get_trade_dates
from factorzen.core.data_quality import QualityCheckError, build_daily_quality_report
from factorzen.core.experiment import (
    record_experiment_metadata,
    record_experiment_output,
    run_experiment,
)
from factorzen.core.loader import fetch_daily
from factorzen.core.logger import get_logger, setup_logging
from factorzen.core.progress import OverallProgress
from factorzen.core.storage import load_parquet
from factorzen.core.timing import StageTimer
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.evaluation.backtest import (
    BacktestResult,
    run_strategy_backtest,
    trim_backtest_to_first_trade,
)
from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
from factorzen.daily.evaluation.turnover import compute_turnover
from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary
from factorzen.daily.factors.registry import get_factor
from factorzen.daily.runtime import (
    build_backtest_strategies,
    build_cost_model,
    build_runtime_backtest_config,
)
from factorzen.experiments.run_paths import artifact_path
from factorzen.pipelines._report_config import (
    _effective_report_config,
    _merge_report_config_args,
)
from factorzen.pipelines._report_direction import (
    _apply_backtest_direction,
    _decide_backtest_direction,
)
from factorzen.pipelines._report_persistence import (
    _existing_report_outputs,
    _meta_path,
    _quality_path,
    _save_quality_report,
    _save_results,
)
from factorzen.pipelines.daily_single import (
    _build_forward_return_frame,
    _compute_monotonicity_result,
    _load_daily_basic_for_neutralization,
    _preprocess_factor,
    filter_frame_by_membership,
    load_pit_membership,
)
from factorzen.reports.trading_report import generate_trading_report

# setup_logging() 只在 main() 入口调用——模块级会让任何 import 本模块的测试
# 向真实 workspace/runs/logs/ 追加日志（写逃逸）
logger = get_logger(__name__)


def _attach_close_adj(daily: pl.DataFrame, adj: pl.DataFrame) -> pl.DataFrame:
    """join 复权因子派生 close_adj = close * adj_factor（与 DailyContext.daily 同口径）。
    adj 为空/缺 adj_factor 时原样返回，下游 _build_forward_return_frame 会回退未复权 close。"""
    if adj.is_empty() or "adj_factor" not in adj.columns:
        return daily
    return (
        daily.join(
            adj.select(["ts_code", "trade_date", "adj_factor"]),
            on=["ts_code", "trade_date"],
            how="left",
        )
        .with_columns((pl.col("close") * pl.col("adj_factor")).alias("close_adj"))
        .drop("adj_factor")
    )


def _load_daily_with_close_adj(start: str, end: str) -> pl.DataFrame:
    """load 日线并 join 复权因子派生 close_adj，供前向收益/IC 标签使用。

    fz report build 历史上用未复权 close 构造前向收益，与 fz factor eval/backtest（走
    DailyContext.daily，优先 close_adj）口径分叉，且 A 股除权除息日 close 跳空
    会污染 IC/单调性/分层 IC。这里补上 close_adj；adj_factor 缺失时优雅回退。
    """
    daily = load_parquet("daily", start=start, end=end).collect()
    if daily.is_empty():
        return daily
    try:
        adj = load_parquet("adj_factor", start=start, end=end).collect()
    except Exception as e:  # adj_factor 分区不存在等 → 回退未复权 close
        logger.warning("adj_factor 加载失败，前向收益回退未复权 close：%s", e)
        return daily
    return _attach_close_adj(daily, adj)


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
    # PIT ST 涨跌停阈值（4.8% 而非 9.8%）：与 daily_single 一致构建 is_st_by_date 传入，
    # 否则本路径把回测期内曾 ST 的股票按主板 9.8% 判涨跌停，与 fz factor backtest 双路径漂移。
    from factorzen.core.universe import build_is_st_by_date
    codes = daily["ts_code"].unique().to_list()
    trade_dates_list = sorted(daily["trade_date"].unique().to_list())
    is_st_by_date = build_is_st_by_date(codes, trade_dates_list)
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
            is_st_by_date=is_st_by_date,
        )
        result = trim_backtest_to_first_trade(result)
        strategy_results[strategy_name] = result
        logger.info(f"\n{result.summary()}")

    primary_name = config.backtest.primary or next(iter(strategy_results))
    return strategy_results[primary_name], strategy_results


# ---------------------------------------------------------------------------
# 高级评价
# ---------------------------------------------------------------------------


def _run_advanced_evaluation(
    backtest_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    frequency: str = "daily",
    start: str = "",
    end: str = "",
    *,
    n_groups: int = 5,
):
    """只算单调性；单一实现在 daily_single._compute_monotonicity_result（双路径共用）。

    frequency/start/end 保留形参以兼容调用方；失败返回 None。
    """
    del frequency, start, end  # 单调性不依赖这些参数
    return _compute_monotonicity_result(backtest_df, ret_df, n_groups=n_groups)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _run(
    args: argparse.Namespace,
    effective_config: RunConfig,
    run_dir: Path,
    timer: StageTimer | None = None,
) -> dict[str, str]:
    timer = timer or StageTimer()
    progress = OverallProgress(4, label="Report run").start()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"──── 因子报告生成: {args.factor} | {args.start} ~ {args.end} ────")

    # ── 1. 获取因子类 ──
    try:
        factor_cls = get_factor(args.factor)
    except KeyError as e:
        logger.error(str(e))
        raise RuntimeError(f"unknown factor: {args.factor}") from e
    factor = factor_cls()
    progress.advance("init")
    logger.info(f"因子: {factor.name} | {factor.description}")

    walk_forward_summary: dict | None = None
    backtest_direction: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None

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

    # ── 3. 股票池（逐日 PIT membership；union 拉取 + 评估截面按日过滤）──
    try:
        membership, ts_codes, universe_meta = load_pit_membership(
            args.start, args.end, args.universe
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"股票池 membership 失败: {e}")
        raise
    if not ts_codes and args.universe != "all_a":
        logger.error(f"股票池为空: {args.universe} [{args.start},{args.end}]")
        raise RuntimeError(f"empty universe membership: {args.universe}")
    logger.info(
        f"股票池(PIT membership): union={len(ts_codes)} 只, "
        f"membership_rows={membership.height}"
    )

    # ── 4. 计算因子（与 daily_single 一致：ensure store 面板，失败回落直算）──
    # 面板缓存是日频口径；weekly/monthly 必须直算（与 daily_single 同门）
    from factorzen.pipelines.factor_panel_cache import is_daily_frequency

    _freq_daily = is_daily_frequency(args, factor)
    use_cache = not bool(getattr(args, "no_factor_cache", False)) and _freq_daily
    if not _freq_daily:
        logger.info("非日频评估（factor/args frequency != daily），跳过面板缓存直算")
    factor_df = None
    if use_cache:
        from datetime import datetime as _dt

        from factorzen.pipelines.factor_panel_cache import ensure_factor_store_panel

        full_panel = ensure_factor_store_panel(
            factor,
            args.start,
            args.end,
            market=getattr(args, "market", None) or "ashare",
            benchmark=getattr(args, "benchmark", None),
        )
        if full_panel is not None:
            def _parse_ymd(s: str):
                k = str(s).replace("-", "").replace("/", "")[:8]
                return _dt.strptime(k, "%Y%m%d").date()

            win_start = _parse_ymd(args.start)
            win_end = _parse_ymd(args.end)
            factor_df = full_panel.filter(
                (pl.col("trade_date") >= win_start) & (pl.col("trade_date") <= win_end)
            )
            if ts_codes:
                factor_df = factor_df.filter(pl.col("ts_code").is_in(ts_codes))
            if "factor_value" in factor_df.columns:
                factor_df = factor_df.select(
                    [
                        c
                        for c in ("trade_date", "ts_code", "factor_value")
                        if c in factor_df.columns
                    ]
                )

    ctx = FactorDataContext(
        start=args.start,
        end=args.end,
        required_data=factor.required_data,
        lookback_days=factor.lookback_days,
        universe=ts_codes if ts_codes else None,
        snapshot_mode=args.frequency,
    )
    if factor_df is None:
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

    # ── 5. 预处理（与 daily_single 同口径：中性化 side data 齐备）+ 逐日 PIT 过滤 ──
    # 不带 side data 直调 pipeline 会让 industry+size 中性化静默跳过（仅 warning），
    # 与 fz factor eval/backtest 的因子值口径漂移——双路径登记簿。
    daily_basic_for_neutralize = None
    if (
        effective_config.preprocessing.neutralize
        and effective_config.preprocessing.neutralize_by in ("size", "industry+size")
    ):
        try:
            daily_basic_for_neutralize = _load_daily_basic_for_neutralization(
                args.start, args.end
            )
        except Exception as e:
            logger.error(f"daily_basic 本地缓存读取失败，无法执行市值中性化: {e}")
            raise RuntimeError(
                f"load daily_basic cache failed for neutralization: {e}"
            ) from e
    clean_df = _preprocess_factor(
        factor_df,
        effective_config,
        universe=universe_meta,
        daily_basic=daily_basic_for_neutralize,
    )
    clean_df = filter_frame_by_membership(clean_df, membership)
    if clean_df.is_empty():
        logger.error("PIT membership 过滤后因子截面为空")
        raise RuntimeError("empty factor cross-section after PIT membership filter")
    logger.info(
        f"预处理完成 (去极值 → 填充 → 标准化 → 逐日 PIT 过滤, n={clean_df.height})"
    )

    # ── 6. 前向收益（用复权价，与 fz factor eval/backtest 口径一致，避免除权跳空污染 IC）──
    daily = _load_daily_with_close_adj(args.start, args.end)
    if daily.is_empty():
        logger.error("日线数据为空，无法计算收益")
        raise RuntimeError("empty daily data")
    ret_df = _build_forward_return_frame(daily)
    logger.info("前向收益计算完成 (horizons: 1/5/10/20d，复权价)")

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
    quality_report_path = _save_quality_report(run_dir, quality_report)
    if quality_report["warnings"]:
        logger.warning(f"数据质量警告: {quality_report['warnings']}")
    logger.info(f"数据质量报告已保存: {quality_report_path}")

    # ── 7. IC 分析 ──
    with timer.stage("IC 分析"):
        ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
    ic_result.factor_name = factor.name
    logger.info(f"\n{ic_result.summary()}")
    backtest_direction = _decide_backtest_direction(ic_result)
    logger.info(f"回测方向判定: {backtest_direction['reason']}")
    backtest_df = _apply_backtest_direction(clean_df, backtest_direction)

    # ── 8. 策略回测 ──
    with timer.stage("策略回测"):
        bt_result, _ = _run_backtest_strategies(
            effective_config,
            backtest_df,
            daily,
            factor_name=factor.name,
            frequency=args.frequency,
        )

    # ── 9. 换手率 ──
    with timer.stage("换手率"):
        to_result = compute_turnover(backtest_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")

    # ── 10. Walk-forward / OOS 摘要 ──
    if effective_config.walk_forward.enabled:
        try:
            walk_forward_summary, _ = run_quantile_walk_forward_summary(
                backtest_df,
                daily,
                effective_config,
                factor_name=factor.name,
                frequency=args.frequency,
            )
            logger.info(f"Walk-forward 摘要: {walk_forward_summary}")
        except Exception as e:
            walk_forward_summary = {"status": "error", "n_folds": 0, "error": str(e)}
            logger.warning(f"Walk-forward 计算失败（跳过）: {e}")
    else:
        walk_forward_summary = {"status": "disabled", "n_folds": 0}
        logger.info("Walk-forward 已关闭，跳过")

    # ── 11. 单调性（与回测同一信号口径；日志侧车，交易报告不消费）──
    _ = _run_advanced_evaluation(
        backtest_df, ret_df, args.frequency, args.start, args.end, n_groups=5
    )

    # ── 持久化中间结果（只写 run_dir）──
    _save_results(
        run_dir,
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
    progress.advance("results")

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

    # ── 生成 HTML 报告 ──
    # 完整计算路径在 §6 必然已写入 quality_report（失败则 raise），无需 JSON 回读。
    progress.advance("benchmark")
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    with timer.stage("报告生成"):
        html = generate_trading_report(
            factor.name,
            bt_result,
            date_range=date_range,
            universe=args.universe,
            strategy_name=str(getattr(bt_result, "strategy_name", "") or ""),
            backtest_direction=backtest_direction,
            benchmark_result=benchmark_result,
            walk_forward_summary=walk_forward_summary,
            quality_report=quality_report,
        )

    # ── 12. 落盘 HTML（run 内 + reports/daily 镜像）──
    report_path = artifact_path(run_dir, "report")
    report_path.write_text(html, encoding="utf-8")
    report_dir = daily_report_output_dir(factor.name)
    report_dir.mkdir(parents=True, exist_ok=True)
    mirror = report_dir / f"{factor.name}_{args.start}_{args.end}.html"
    mirror.write_text(html, encoding="utf-8")
    progress.advance("report")
    logger.info(f"报告已生成: {report_path} (镜像 {mirror})")

    outputs = {
        "report": str(report_path),
        "meta": str(_meta_path(run_dir)),
    }
    quality_report_path = _quality_path(run_dir)
    if quality_report_path.exists():
        outputs["quality_report"] = str(quality_report_path)
    # factor 面板在 factors/，路径记在 meta.store_panel
    progress.close()
    return outputs


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="因子交易轨报告生成")
    parser.add_argument("--factor", default=None, help="因子名称")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default=None, help="股票池")
    parser.add_argument(
        "--frequency", default="daily", choices=["daily", "weekly", "monthly"], help="因子频率"
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="基准指数代码（如 000300.SH），若指定则计算超额收益与 benchmark 对比",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML 运行配置文件路径")
    parser.add_argument(
        "--no-factor-cache",
        action="store_true",
        dest="no_factor_cache",
        help="禁用 factors 物化缓存（默认启用；miss 时回落直算）",
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
            timer = StageTimer()
            try:
                outputs = _run(args, effective_config, exp_dir, timer=timer)
            except Exception:
                record_experiment_metadata(exp_dir, "stage_timings", timer.timings)
                for name, path in _existing_report_outputs(exp_dir).items():
                    record_experiment_output(exp_dir, name, path)
                raise
            record_experiment_metadata(exp_dir, "stage_timings", timer.timings)
            for name, path in outputs.items():
                record_experiment_output(exp_dir, name, path)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Done.")


if __name__ == "__main__":
    main()
