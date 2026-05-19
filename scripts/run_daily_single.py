"""日频单因子完整评估。用法: python scripts/run_daily_single.py --factor momentum_20d --start 20250101 --end 20250513"""

import argparse
import json
import sys

import polars as pl

from common.calendar import get_trade_dates
from common.config_loader import (
    RunConfig,
    build_cost_model,
    build_preprocessing_pipeline,
    build_runtime_backtest_config,
)
from common.data_quality import QualityCheckError, build_daily_quality_report
from common.experiment import record_experiment_output, run_experiment
from common.loader import fetch_daily
from common.logger import get_logger, setup_logging
from common.universe import get_universe
from config.settings import OUTPUT_DAILY_FACTORS, OUTPUT_DAILY_REPORTS, OUTPUT_DAILY_RESULTS
from daily.data.context import FactorDataContext
from daily.evaluation.backtest import run_stratified_backtest
from daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from daily.evaluation.turnover import compute_turnover
from daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary
from daily.factors.registry import get_factor
from reporting.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)


def _merge_run_config_args(args: argparse.Namespace, run_config: RunConfig | None):
    """Merge YAML config into argparse args without overriding explicit CLI values."""
    if run_config is not None:
        for field in ("factor", "start", "end", "universe", "benchmark", "seed"):
            if getattr(args, field, None) is None:
                setattr(args, field, getattr(run_config, field))
        if args.ic_method is None:
            args.ic_method = run_config.ic_method
        if args.neutralized_ic is None:
            args.neutralized_ic = run_config.neutralized_ic
        if args.event_study is None:
            args.event_study = run_config.event_study

    if args.universe is None:
        args.universe = "csi300"
    if args.ic_method is None:
        args.ic_method = "rank"
    if args.neutralized_ic is None:
        args.neutralized_ic = False
    if args.event_study is None:
        args.event_study = False

    missing = [field for field in ("factor", "start", "end") if getattr(args, field, None) is None]
    if missing:
        raise ValueError(f"缺少必填参数: {', '.join(missing)}（可通过 CLI 或 --config 提供）")

    return args


def _effective_run_config(args: argparse.Namespace, run_config: RunConfig | None) -> RunConfig:
    """Return a RunConfig that reflects final merged scalar CLI values."""
    base = run_config or RunConfig(factor=args.factor, start=args.start, end=args.end)
    return base.model_copy(
        update={
            "factor": args.factor,
            "start": args.start,
            "end": args.end,
            "universe": args.universe,
            "benchmark": args.benchmark or base.benchmark,
            "seed": args.seed,
            "ic_method": args.ic_method,
            "neutralized_ic": args.neutralized_ic,
            "event_study": args.event_study,
        }
    )


def _existing_run_outputs(factor_name: str, start: str, end: str) -> dict[str, str]:
    prefix = f"{factor_name}_{start}_{end}"
    candidates = {
        "factor": OUTPUT_DAILY_FACTORS / f"{prefix}.parquet",
        "ic": OUTPUT_DAILY_RESULTS / f"{prefix}_ic.parquet",
        "quality_report": OUTPUT_DAILY_RESULTS / f"{prefix}_quality.json",
        "walk_forward_summary": OUTPUT_DAILY_RESULTS / f"{prefix}_walk_forward.json",
        "report": OUTPUT_DAILY_REPORTS / f"{prefix}.html",
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}


def _run(args: argparse.Namespace, effective_config: RunConfig) -> dict[str, str]:
    # ── 0b. 设置全局随机种子（可选）──
    if args.seed is not None:
        from common.seed import set_global_seed

        set_global_seed(args.seed)
        logger.info(f"全局随机种子已设置: {args.seed}")

    # ── 1. 获取因子类 ──
    logger.info(f"──── 单因子评估: {args.factor} | {args.start} ~ {args.end} ────")
    try:
        factor_cls = get_factor(args.factor)
    except KeyError as e:
        logger.error(str(e))
        raise RuntimeError(f"unknown factor: {args.factor}") from e
    factor = factor_cls()
    logger.info(f"因子: {factor.name} | {factor.description}")

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

    # ── 6. 计算前向收益 ──
    daily = ctx.daily.collect()
    if daily.is_empty():
        logger.error("日线数据为空，无法计算收益")
        raise RuntimeError("empty daily data")
    ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(["ts_code", "trade_date"])
    ret_df = ret_df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
    )
    ret_df = compute_fwd_returns(ret_df, ret_col="ret")
    logger.info("前向收益计算完成 (horizons: 1/5/10/20d)")

    # ── 6b. 数据质量审计 ──
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
    OUTPUT_DAILY_RESULTS.mkdir(parents=True, exist_ok=True)
    quality_path = (
        OUTPUT_DAILY_RESULTS / f"{factor.name}_{args.start}_{args.end}_quality.json"
    )
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if quality_report["warnings"]:
        logger.warning(f"数据质量警告: {quality_report['warnings']}")
    logger.info(f"数据质量报告已保存: {quality_path}")

    # ── 7. IC 分析 ──
    ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
    ic_result.factor_name = factor.name
    logger.info(f"\n{ic_result.summary()}")

    # 可选：Pearson IC / Both IC
    pearson_ic_result = None
    if args.ic_method in ("pearson", "both"):
        from daily.evaluation.ic_analysis import compute_ic

        # 构建含 ret_1d 列的简化 DataFrame
        merged_simple = clean_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
                {"fwd_ret_1d": "ret_1d"}
            ),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        if args.ic_method == "both":
            both_ic = compute_ic(merged_simple, factor_col="factor_clean", ret_col="ret_1d", method="both")
            pearson_ic_result = both_ic["pearson"]
            logger.info(f"Pearson IC Mean: {pearson_ic_result.ic_mean:.4f}, IR: {pearson_ic_result.ir:.2f}")
        else:
            pearson_ic_result = compute_ic(merged_simple, factor_col="factor_clean", ret_col="ret_1d", method="pearson")
            logger.info(f"Pearson IC Mean: {pearson_ic_result.ic_mean:.4f}, IR: {pearson_ic_result.ir:.2f}")

    # 可选：中性化 IC
    neutralized_ic_result = None
    if args.neutralized_ic:
        from daily.evaluation.advanced import compute_neutralized_ic

        # 尝试构建含 ret_1d 的因子 DataFrame
        merged_neutral = clean_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
                {"fwd_ret_1d": "ret_1d"}
            ),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        try:
            neutralized_ic_result = compute_neutralized_ic(merged_neutral, ret_col="ret_1d")
            logger.info(f"Neutralized IC Mean: {neutralized_ic_result.ic_mean:.4f}")
        except Exception as e:
            logger.warning(f"中性化 IC 计算失败（跳过）: {e}")

    # ── 8. 分层回测 ──
    bt_result = run_stratified_backtest(
        clean_df,
        daily,
        n_groups=effective_config.backtest.quantiles,
        frequency=args.frequency,
        factor_name=factor.name,
        cost_model=build_cost_model(effective_config),
        config=build_runtime_backtest_config(
            effective_config, factor_col="factor_clean", frequency=args.frequency
        ),
    )
    logger.info(f"\n{bt_result.summary()}")

    # ── 9. 换手率 ──
    to_result = compute_turnover(clean_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")

    # ── 10. Walk-forward / OOS 摘要 ──
    try:
        walk_forward_summary, walk_forward_result = run_quantile_walk_forward_summary(
            clean_df,
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

    # ── 11. 落盘 ──
    OUTPUT_DAILY_FACTORS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DAILY_RESULTS.mkdir(parents=True, exist_ok=True)

    factor_path = OUTPUT_DAILY_FACTORS / f"{factor.name}_{args.start}_{args.end}.parquet"
    clean_df.write_parquet(str(factor_path))
    logger.info(f"因子已保存: {factor_path}")

    walk_forward_path = (
        OUTPUT_DAILY_RESULTS / f"{factor.name}_{args.start}_{args.end}_walk_forward.json"
    )
    walk_forward_path.write_text(
        json.dumps(walk_forward_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Walk-forward 摘要已保存: {walk_forward_path}")

    ic_result.ic_series.write_parquet(
        str(OUTPUT_DAILY_RESULTS / f"{factor.name}_{args.start}_{args.end}_ic.parquet")
    )

    # ── 11b. 事件研究（可选）──
    event_study_result = None
    if args.event_study:
        from daily.evaluation.advanced import compute_event_study

        factor_simple = clean_df.select(["trade_date", "ts_code", "factor_clean"])
        ret_simple = ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
            {"fwd_ret_1d": "ret_1d"}
        )
        try:
            event_study_result = compute_event_study(factor_simple, ret_simple)
            logger.info(f"事件研究完成: {event_study_result.n_events} 个事件")
        except Exception as e:
            logger.warning(f"事件研究计算失败（跳过）: {e}")

    # ── 12. Benchmark 对比（可选）──
    benchmark_result = None
    if args.benchmark:
        try:
            from daily.evaluation.benchmark import compute_excess_return

            benchmark_result = compute_excess_return(
                bt_result.returns, args.benchmark, args.start, args.end
            )
            logger.info(f"Benchmark: {benchmark_result.summary()}")
        except Exception as e:
            logger.warning(f"Benchmark 计算失败（跳过）: {e}")

    # ── 13. HTML 报告（当 --benchmark 提供时生成，或始终生成）──
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    html = generate_tear_sheet(
        factor_name=factor.name,
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        frequency=args.frequency,
        date_range=date_range,
        universe=args.universe,
        benchmark_result=benchmark_result,
        attribution_result=None,
        event_study_result=event_study_result,
        walk_forward_result=walk_forward_result,
        walk_forward_summary=walk_forward_summary,
        pearson_ic_result=pearson_ic_result if args.ic_method in ("pearson", "both") else None,
        neutralized_ic_result=neutralized_ic_result if args.neutralized_ic else None,
    )
    OUTPUT_DAILY_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DAILY_REPORTS / f"{factor.name}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"报告已生成: {report_path}")

    return {
        "factor": str(factor_path),
        "ic": str(OUTPUT_DAILY_RESULTS / f"{factor.name}_{args.start}_{args.end}_ic.parquet"),
        "quality_report": str(quality_path),
        "walk_forward_summary": str(walk_forward_path),
        "report": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description="日频单因子评估")
    parser.add_argument("--factor", default=None, help="因子名称")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", type=str, default=None, help="股票池")
    parser.add_argument(
        "--frequency", default="daily", choices=["daily", "weekly", "monthly"], help="因子频率"
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="基准指数代码（如 000300.SH），若指定则计算超额收益并生成 HTML 报告",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML 运行配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子")
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
        help="是否计算中性化后的 Rank IC（需要因子 DataFrame 含 industry 或 log_mktcap 列）",
    )
    parser.add_argument(
        "--event-study",
        action="store_true",
        dest="event_study",
        default=None,
        help="是否执行事件研究分析（选 Top 5%% 分位股票为事件）",
    )
    args = parser.parse_args()

    # ── 0. 加载 YAML 配置（可选），CLI 参数优先级更高 ──
    run_config = None
    if args.config:
        from common.config_loader import load_run_config

        run_config = load_run_config(args.config)

    try:
        args = _merge_run_config_args(args, run_config)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)

    effective_config = _effective_run_config(args, run_config)

    try:
        with run_experiment(effective_config, command=sys.argv) as exp_dir:
            try:
                outputs = _run(args, effective_config)
            except Exception:
                for name, path in _existing_run_outputs(args.factor, args.start, args.end).items():
                    record_experiment_output(exp_dir, name, path)
                raise
            for name, path in outputs.items():
                record_experiment_output(exp_dir, name, path)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Done.")


if __name__ == "__main__":
    main()
