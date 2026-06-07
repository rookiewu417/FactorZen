"""日频单因子完整评估。用法: python factorzen.pipelines.daily_single --factor momentum_20d --start 20250101 --end 20250513"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import polars as pl
from pydantic import ValidationError

from factorzen.config.settings import (
    ROOT,
    daily_factor_output_dir,
    daily_report_output_dir,
    daily_result_output_dir,
)
from factorzen.core.calendar import get_trade_dates
from factorzen.core.config_loader import (
    RunConfig,
    build_backtest_strategies,
    build_cost_model,
    build_default_daily_research_config,
    build_preprocessing_pipeline,
    build_runtime_backtest_config,
    default_benchmark_for_universe,
)
from factorzen.core.data_ensure import ensure_data_for_daily_run
from factorzen.core.data_quality import QualityCheckError, build_daily_quality_report
from factorzen.core.experiment import (
    record_experiment_metadata,
    record_experiment_output,
    run_experiment,
)
from factorzen.core.logger import get_logger, setup_logging
from factorzen.core.progress import OverallProgress
from factorzen.core.storage import load_parquet
from factorzen.core.timing import StageTimer
from factorzen.core.universe import get_universe
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.evaluation.backtest import run_strategy_backtest, trim_backtest_to_first_trade
from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from factorzen.daily.evaluation.turnover import compute_turnover
from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary
from factorzen.daily.factors.registry import get_factor
from factorzen.experiments.run_paths import copy_outputs_to_run_dir
from factorzen.llm import generate_llm_explanation
from factorzen.reports.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)


def _find_default_run_config_path(
    factor_name: str,
    frequency: str,
    *,
    configs_root: Path | None = None,
) -> Path | None:
    """Find one default YAML config whose factor field matches factor_name."""
    root = configs_root or ROOT / "workspace" / "configs"
    config_dir = root / frequency
    if not config_dir.exists():
        return None

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML 未安装，无法自动读取默认 YAML 配置。") from exc

    matches: list[Path] = []
    for path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and data.get("factor") == factor_name:
            matches.append(path)

    exact_matches = [path for path in matches if path.stem == factor_name]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        matches = exact_matches

    primary_matches = [path for path in matches if path.name.startswith("single_factor_")]
    if len(primary_matches) == 1:
        return primary_matches[0]
    if len(primary_matches) > 1:
        matches = primary_matches

    if len(matches) > 1:
        match_text = ", ".join(str(path) for path in matches)
        raise ValueError(f"找到多个默认配置匹配因子 {factor_name}: {match_text}")
    if not matches:
        return None
    return matches[0]


def _merge_run_config_args(args: argparse.Namespace, run_config: RunConfig | None):
    """Merge YAML config into argparse args without overriding explicit CLI values."""
    cli_benchmark = args.benchmark is not None
    cli_ic_method = args.ic_method is not None
    cli_neutralized_ic = args.neutralized_ic is not None
    cli_event_study = args.event_study is not None
    using_builtin_default = run_config is None or bool(
        getattr(args, "_uses_builtin_default_config", False)
    )

    if run_config is None and args.factor and args.start and args.end:
        run_config = build_default_daily_research_config(
            factor=args.factor,
            start=args.start,
            end=args.end,
            universe=args.universe,
            benchmark=args.benchmark,
            seed=args.seed,
        )

    if run_config is not None:
        for field in ("factor", "start", "end", "universe", "seed"):
            if getattr(args, field, None) is None:
                setattr(args, field, getattr(run_config, field))
        if args.benchmark is None and run_config.benchmark is not None:
            args.benchmark = run_config.benchmark
        if args.ic_method is None:
            args.ic_method = run_config.ic_method
        if args.neutralized_ic is None:
            args.neutralized_ic = run_config.neutralized_ic
        if args.event_study is None:
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

    if using_builtin_default:
        args.llm_explain = True

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
    base = run_config or build_default_daily_research_config(
        factor=args.factor,
        start=args.start,
        end=args.end,
        universe=args.universe,
        benchmark=args.benchmark,
        seed=args.seed,
    )
    config = base.model_copy(
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
    return config


def _write_run_metrics(path: str, ic_result: Any, bt_result: Any) -> None:
    """把 IC 与主策略组合级回测指标写出为 JSON（供 factor sweep 汇总，内部接口）。"""
    try:
        portfolio = bt_result.summary_stats.get("portfolio", {})
    except Exception:
        portfolio = {}
    metrics = {
        "ic_mean": ic_result.ic_mean,
        "ir": ic_result.ir,
        "t": ic_result.ic_tstat,
        "ic_pos": ic_result.ic_positive_ratio,
        "n": ic_result.n_periods,
        "sharpe": portfolio.get("sharpe"),
        "ann_ret": portfolio.get("ann_ret"),
        "avg_turnover": portfolio.get("avg_turnover"),
        "max_dd": portfolio.get("max_dd"),
    }
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


def _build_dry_run_payload(
    config: RunConfig,
    *,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    payload = {
        "config": config.model_dump(),
        "output_dir": (ROOT / "workspace" / "factor_evaluations" / "<run_id>").as_posix(),
    }
    if args is not None:
        payload["execution"] = {
            "llm_explain": bool(getattr(args, "llm_explain", False)),
            "llm_refresh": bool(getattr(args, "llm_refresh", False)),
        }
    return payload


def _existing_run_outputs(factor_name: str, start: str, end: str) -> dict[str, str]:
    prefix = f"{factor_name}_{start}_{end}"
    factor_dir = daily_factor_output_dir(factor_name)
    result_dir = daily_result_output_dir(factor_name)
    report_dir = daily_report_output_dir(factor_name)
    candidates = {
        "factor": factor_dir / f"{prefix}.parquet",
        "ic": result_dir / f"{prefix}_ic.parquet",
        "quality_report": result_dir / f"{prefix}_quality.json",
        "walk_forward_summary": result_dir / f"{prefix}_walk_forward.json",
        "report": report_dir / f"{prefix}.html",
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}


def _preprocess_factor(
    factor_df: pl.DataFrame,
    effective_config: RunConfig,
    *,
    universe: pl.DataFrame,
    daily_basic: pl.DataFrame | None,
) -> pl.DataFrame:
    """Run configured preprocessing with the side data required by neutralization."""
    stock_basic = None
    daily_basic_input = None
    if effective_config.preprocessing.neutralize:
        neutralize_by = effective_config.preprocessing.neutralize_by
        if neutralize_by in ("industry", "industry+size"):
            stock_basic = universe
        if neutralize_by in ("size", "industry+size"):
            daily_basic_input = daily_basic
    return build_preprocessing_pipeline(effective_config).run(
        factor_df,
        col="factor_value",
        stock_basic=stock_basic,
        daily_basic=daily_basic_input,
    )


def _load_daily_basic_for_neutralization(start: str, end: str) -> pl.DataFrame:
    """Read daily_basic after the data assurance step has filled any gaps."""
    return load_parquet("daily_basic", start=start, end=end).collect()


def _date_expr(column: str) -> pl.Expr:
    parsed_dash = pl.col(column).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    parsed_plain = pl.col(column).str.strptime(pl.Date, "%Y%m%d", strict=False)
    return parsed_dash.fill_null(parsed_plain).alias(column)


def _ensure_date_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in df.columns:
        return df
    dtype = df.schema[column]
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(column).dt.date().alias(column))
    if dtype == pl.Utf8:
        return df.with_columns(_date_expr(column))
    return df


def _sector_lookup(universe: pl.DataFrame) -> pl.DataFrame:
    if universe.is_empty() or not {"ts_code", "industry"}.issubset(set(universe.columns)):
        return pl.DataFrame(schema={"ts_code": pl.Utf8, "sector": pl.Utf8})
    return (
        universe.select(["ts_code", "industry"])
        .rename({"industry": "sector"})
        .filter(pl.col("sector").is_not_null() & (pl.col("sector") != ""))
        .unique(subset=["ts_code"])
    )


def _build_neutralized_ic_frame(
    clean_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    *,
    universe: pl.DataFrame,
    daily_basic: pl.DataFrame | None,
) -> pl.DataFrame:
    frame = _ensure_date_column(clean_df, "trade_date").join(
        _ensure_date_column(ret_df, "trade_date")
        .select(["trade_date", "ts_code", "fwd_ret_1d"])
        .rename({"fwd_ret_1d": "ret_1d"}),
        on=["trade_date", "ts_code"],
        how="inner",
    )

    if not universe.is_empty() and {"ts_code", "industry"}.issubset(set(universe.columns)):
        industry_lut = (
            universe.select(["ts_code", "industry"])
            .filter(pl.col("industry").is_not_null() & (pl.col("industry") != ""))
            .unique(subset=["ts_code"])
        )
        frame = frame.join(industry_lut, on="ts_code", how="left")

    if daily_basic is not None and not daily_basic.is_empty() and "total_mv" in daily_basic.columns:
        size_lut = _ensure_date_column(daily_basic, "trade_date").select(
            ["trade_date", "ts_code", "total_mv"]
        )
        frame = frame.join(size_lut, on=["trade_date", "ts_code"], how="left")

    return frame


def _build_forward_return_frame(daily: pl.DataFrame) -> pl.DataFrame:
    """Build IC forward-return labels, preferring adjusted close when available."""
    if "close_adj" not in daily.columns:
        price_col = "close"
        ret_df = daily.select(["trade_date", "ts_code", price_col]).sort(["ts_code", "trade_date"])
        ret_df = ret_df.with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1).over("ts_code") - 1).alias("ret")
        )
        return compute_fwd_returns(ret_df, ret_col="ret", price_col=price_col)

    price_col = "_label_price"
    valid_adj = (
        (pl.col("close_adj").is_not_null() & pl.col("close_adj").is_finite())
        .fill_null(False)
        .all()
        .over("ts_code")
    )
    ret_df = (
        daily.select(["trade_date", "ts_code", "close", "close_adj"])
        .with_columns(
            pl.when(valid_adj).then(pl.col("close_adj")).otherwise(pl.col("close")).alias(price_col)
        )
        .select(["trade_date", "ts_code", price_col])
        .sort(["ts_code", "trade_date"])
    )
    ret_df = ret_df.with_columns(
        (pl.col(price_col) / pl.col(price_col).shift(1).over("ts_code") - 1).alias("ret")
    )
    return compute_fwd_returns(ret_df, ret_col="ret", price_col=price_col)


def _build_advanced_results(
    clean_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    *,
    universe: pl.DataFrame,
    daily_basic: pl.DataFrame | None,
) -> dict[str, Any] | None:
    advanced: dict[str, Any] = {}
    sector_lut = _sector_lookup(universe)
    if not sector_lut.is_empty():
        try:
            from factorzen.daily.evaluation.advanced import compute_sector_ic

            sector_df = (
                clean_df.join(sector_lut, on="ts_code", how="left")
                .join(
                    ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                    on=["trade_date", "ts_code"],
                    how="inner",
                )
                .filter(pl.col("sector").is_not_null())
            )
            if not sector_df.is_empty():
                advanced["sector"] = compute_sector_ic(
                    sector_df,
                    factor_col="factor_clean",
                    ret_col="fwd_ret_1d",
                    sector_col="sector",
                    return_object=True,
                )
        except Exception as e:
            logger.warning(f"行业分层 IC 计算失败（跳过）: {e}")

    if daily_basic is not None and not daily_basic.is_empty() and "total_mv" in daily_basic.columns:
        try:
            from factorzen.daily.evaluation.advanced import compute_size_ic

            size_df = (
                clean_df.join(
                    daily_basic.select(["trade_date", "ts_code", "total_mv"]),
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
            if not size_df.is_empty():
                advanced["size"] = compute_size_ic(
                    size_df,
                    factor_col="factor_clean",
                    ret_col="fwd_ret_1d",
                    cap_col="total_mv",
                    n_buckets=3,
                    return_object=True,
                )
        except Exception as e:
            logger.warning(f"市值分层 IC 计算失败（跳过）: {e}")

    return advanced or None


def _build_attribution_result(
    bt_result: Any,
    daily: pl.DataFrame,
    universe: pl.DataFrame,
) -> dict[str, Any] | None:
    positions = getattr(bt_result, "positions", None)
    if positions is None or positions.is_empty() or daily.is_empty():
        return None

    sector_lut = _sector_lookup(universe)
    if sector_lut.is_empty() or not {"trade_date", "ts_code", "weight"}.issubset(
        set(positions.columns)
    ):
        return None

    try:
        from factorzen.daily.evaluation.attribution import brinson_attribution

        daily_ret = (
            _ensure_date_column(daily, "trade_date")
            .select(["trade_date", "ts_code", "close"])
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret")
            )
            .select(["trade_date", "ts_code", "ret"])
            .filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())
        )
        long_positions = (
            _ensure_date_column(positions, "trade_date")
            .select(["trade_date", "ts_code", "weight"])
            .filter(pl.col("weight") > 0)
            .join(sector_lut, on="ts_code", how="inner")
            .join(daily_ret, on=["trade_date", "ts_code"], how="inner")
        )
        if long_positions.is_empty():
            return None

        daily_weight = long_positions.group_by("trade_date").agg(
            pl.col("weight").sum().alias("_daily_weight")
        )
        long_positions = (
            long_positions.join(daily_weight, on="trade_date", how="left")
            .filter(pl.col("_daily_weight") > 0)
            .with_columns((pl.col("weight") / pl.col("_daily_weight")).alias("_stock_weight"))
            .with_columns((pl.col("_stock_weight") * pl.col("ret")).alias("_weighted_ret"))
        )
        portfolio = (
            long_positions.group_by(["trade_date", "sector"])
            .agg(
                [
                    pl.col("_stock_weight").sum().alias("port_weight"),
                    pl.col("_weighted_ret").sum().alias("_ret_num"),
                ]
            )
            .with_columns((pl.col("_ret_num") / pl.col("port_weight")).alias("port_ret"))
            .select(["trade_date", "sector", "port_weight", "port_ret"])
        )
        if portfolio.is_empty():
            return None

        portfolio_dates = portfolio.select("trade_date").unique()
        benchmark_base = (
            daily_ret.join(sector_lut, on="ts_code", how="inner")
            .join(portfolio_dates, on="trade_date", how="inner")
            .filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())
        )
        benchmark_counts = benchmark_base.group_by("trade_date").agg(pl.len().alias("_n_total"))
        benchmark = (
            benchmark_base.group_by(["trade_date", "sector"])
            .agg([pl.len().alias("_n_sector"), pl.col("ret").mean().alias("bench_ret")])
            .join(benchmark_counts, on="trade_date", how="left")
            .with_columns((pl.col("_n_sector") / pl.col("_n_total")).alias("bench_weight"))
            .select(["trade_date", "sector", "bench_weight", "bench_ret"])
        )
        if benchmark.is_empty():
            return None

        sector_returns = (
            benchmark.select(["trade_date", "sector", "bench_ret"])
            .join(
                portfolio.select(["trade_date", "sector", "port_ret"]),
                on=["trade_date", "sector"],
                how="left",
            )
            .with_columns(pl.col("port_ret").fill_null(pl.col("bench_ret")))
        )
        brinson = brinson_attribution(
            portfolio.select(["trade_date", "sector", "port_weight"]),
            benchmark.select(["trade_date", "sector", "bench_weight"]),
            sector_returns.select(["trade_date", "sector", "port_ret", "bench_ret"]),
        )
        if brinson.sector_df.is_empty():
            return None
        return {"brinson": brinson}
    except Exception as e:
        logger.warning(f"Brinson 行业归因计算失败（跳过）: {e}")
        return None


def _run_backtest_strategies(
    config: RunConfig,
    clean_df: pl.DataFrame,
    daily: pl.DataFrame,
    *,
    factor_name: str,
    frequency: str,
) -> tuple[Any, dict[str, Any]]:
    strategy_results: dict[str, Any] = {}
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
        if hasattr(result, "summary"):
            logger.info(f"\n{result.summary()}")

    primary_name = config.backtest.primary or next(iter(strategy_results))
    return strategy_results[primary_name], strategy_results


def _run(
    args: argparse.Namespace,
    effective_config: RunConfig,
    timer: StageTimer | None = None,
) -> dict[str, str]:
    timer = timer or StageTimer()
    progress = OverallProgress(16, label="Daily run").start()
    # ── 0b. 设置全局随机种子（可选）──
    if args.seed is not None:
        from factorzen.core.seed import set_global_seed

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
    factor_output_dir = daily_factor_output_dir(factor.name)
    result_output_dir = daily_result_output_dir(factor.name)
    report_output_dir = daily_report_output_dir(factor.name)
    progress.advance("init")

    # ── 2. 准备数据 ──
    trade_dates = get_trade_dates(args.start, args.end)
    logger.info(f"交易日数: {len(trade_dates)}")
    if len(trade_dates) < 30:
        logger.warning("交易日不足 30 天，IC 分析可能不稳定")

    try:
        ensure_data_for_daily_run(
            required_data=factor.required_data,
            start=args.start,
            end=args.end,
            universe=args.universe,
            benchmark=args.benchmark,
            needs_size_neutralization=(
                effective_config.preprocessing.neutralize
                and effective_config.preprocessing.neutralize_by in ("size", "industry+size")
            )
            or bool(getattr(args, "neutralized_ic", False)),
            is_qlib_factor=getattr(factor, "category", "") == "qlib"
            or factor.name.startswith("qlib_"),
        )
    except Exception as e:
        logger.error(f"数据保障失败: {e}")
        raise RuntimeError(f"ensure_data_for_daily_run failed: {e}") from e
    progress.advance("data")

    # ── 3. 股票池 ──
    universe = get_universe(args.end, args.universe)
    if universe.is_empty():
        logger.error(f"股票池为空: {args.universe} ({args.end})")
        raise RuntimeError(f"empty universe: {args.universe} ({args.end})")
    ts_codes = universe["ts_code"].to_list()
    logger.info(f"股票池: {len(ts_codes)} 只")

    # ── 3b. 保存 universe 快照（供复现和审计）──
    result_output_dir.mkdir(parents=True, exist_ok=True)
    universe_snapshot_path = (
        result_output_dir / f"{factor.name}_{args.start}_{args.end}_universe.parquet"
    )
    universe.write_parquet(str(universe_snapshot_path))
    logger.info(f"Universe 快照已保存: {universe_snapshot_path} ({len(ts_codes)} 只)")
    progress.advance("universe")

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
    progress.advance("factor")

    # ── 5. 预处理 ──
    daily_basic_for_neutralize = None
    if (
        effective_config.preprocessing.neutralize
        and effective_config.preprocessing.neutralize_by in ("size", "industry+size")
    ):
        try:
            daily_basic_for_neutralize = _load_daily_basic_for_neutralization(args.start, args.end)
        except Exception as e:
            logger.error(f"daily_basic 本地缓存读取失败，无法执行市值中性化: {e}")
            raise RuntimeError(f"load daily_basic cache failed for neutralization: {e}") from e

    clean_df = _preprocess_factor(
        factor_df,
        effective_config,
        universe=universe,
        daily_basic=daily_basic_for_neutralize,
    )
    logger.info("预处理完成 (去极值 → 填充 → 标准化)")
    progress.advance("preprocess")

    # ── 6. 计算前向收益 ──
    daily = ctx.daily.collect()
    if daily.is_empty():
        logger.error("日线数据为空，无法计算收益")
        raise RuntimeError("empty daily data")
    ret_df = _build_forward_return_frame(daily)
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
    result_output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = result_output_dir / f"{factor.name}_{args.start}_{args.end}_quality.json"
    quality_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if quality_report["warnings"]:
        logger.warning(f"数据质量警告: {quality_report['warnings']}")
    logger.info(f"数据质量报告已保存: {quality_path}")
    progress.advance("returns-quality")

    # ── 7. IC 分析 ──
    with timer.stage("IC 分析"):
        ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
    ic_result.factor_name = factor.name
    logger.info(f"\n{ic_result.summary()}")
    progress.advance("ic")

    # 可选：Pearson IC / Both IC
    pearson_ic_result = None
    if args.ic_method in ("pearson", "both"):
        from factorzen.daily.evaluation.ic_analysis import BothIcResult, IcStats, compute_ic

        # 构建含 ret_1d 列的简化 DataFrame
        merged_simple = clean_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
                {"fwd_ret_1d": "ret_1d"}
            ),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        if args.ic_method == "both":
            both_ic = cast(
                BothIcResult,
                compute_ic(
                    merged_simple,
                    factor_col="factor_clean",
                    ret_col="ret_1d",
                    method="both",
                ),
            )
            pearson_ic_result = both_ic["pearson"]
            logger.info(
                f"Pearson IC Mean: {pearson_ic_result.ic_mean:.4f}, "
                f"IR: {pearson_ic_result.ir:.2f}"
            )
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

    # 可选：中性化 IC
    neutralized_ic_result = None
    if args.neutralized_ic:
        from factorzen.daily.evaluation.advanced import compute_neutralized_ic

        # 尝试构建含 ret_1d 的因子 DataFrame
        if daily_basic_for_neutralize is None:
            try:
                daily_basic_for_neutralize = load_parquet(
                    "daily_basic", start=args.start, end=args.end
                ).collect()
            except Exception as e:
                logger.warning(
                    "daily_basic cache load failed; "
                    f"neutralized IC will use available exposures: {e}"
                )
        merged_neutral = _build_neutralized_ic_frame(
            clean_df,
            ret_df,
            universe=universe,
            daily_basic=daily_basic_for_neutralize,
        )
        try:
            neutralized_ic_result = compute_neutralized_ic(merged_neutral, ret_col="ret_1d")
            logger.info(f"Neutralized IC Mean: {neutralized_ic_result.ic_mean:.4f}")
        except Exception as e:
            logger.warning(f"中性化 IC 计算失败（跳过）: {e}")
    progress.advance("optional-ic")

    # ── 8. 策略回测 ──
    with timer.stage("策略回测"):
        bt_result, strategy_results = _run_backtest_strategies(
            effective_config,
            clean_df,
            daily,
            factor_name=factor.name,
            frequency=args.frequency,
        )
    progress.advance("backtest")

    # ── 9. 换手率 ──
    with timer.stage("换手率"):
        to_result = compute_turnover(clean_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")
    progress.advance("turnover")

    factor_output_dir.mkdir(parents=True, exist_ok=True)
    result_output_dir.mkdir(parents=True, exist_ok=True)

    factor_path = factor_output_dir / f"{factor.name}_{args.start}_{args.end}.parquet"
    clean_df.write_parquet(str(factor_path))
    logger.info(f"因子已保存: {factor_path}")

    ic_path = result_output_dir / f"{factor.name}_{args.start}_{args.end}_ic.parquet"
    ic_result.ic_series.write_parquet(str(ic_path))
    logger.info(f"IC 序列已保存: {ic_path}")
    progress.advance("save-core")

    # ── 10. Walk-forward / OOS 摘要 ──
    if effective_config.walk_forward.enabled:
        with timer.stage("Walk-forward"):
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
    else:
        walk_forward_summary = {"status": "disabled", "n_folds": 0}
        walk_forward_result = None
        logger.info("Walk-forward 已关闭，跳过")
    progress.advance("walk-forward")

    # ── 11. 落盘 ──
    daily_basic_for_breakdowns = daily_basic_for_neutralize
    try:
        if daily_basic_for_breakdowns is None:
            daily_basic_for_breakdowns = load_parquet(
                "daily_basic", start=args.start, end=args.end
            ).collect()
    except Exception as e:
        logger.warning(f"daily_basic 缓存加载失败，跳过市值分层 IC: {e}")

    advanced_results = _build_advanced_results(
        clean_df,
        ret_df,
        universe=universe,
        daily_basic=daily_basic_for_breakdowns,
    )
    attribution_result = _build_attribution_result(bt_result, daily, universe)

    walk_forward_path = result_output_dir / f"{factor.name}_{args.start}_{args.end}_walk_forward.json"
    walk_forward_path.write_text(
        json.dumps(walk_forward_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Walk-forward 摘要已保存: {walk_forward_path}")

    # ── 11b. 事件研究（可选）──
    event_study_result = None
    if args.event_study:
        from factorzen.daily.evaluation.advanced import compute_event_study

        factor_simple = clean_df.select(["trade_date", "ts_code", "factor_clean"])
        ret_simple = ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename(
            {"fwd_ret_1d": "ret_1d"}
        )
        try:
            event_study_result = compute_event_study(factor_simple, ret_simple)
            logger.info(f"事件研究完成: {event_study_result.n_events} 个事件")
        except Exception as e:
            logger.warning(f"事件研究计算失败（跳过）: {e}")
    progress.advance("advanced")

    # ── 12. Benchmark 对比（可选）──
    benchmark_result = None
    if args.benchmark:
        try:
            from factorzen.daily.evaluation.benchmark import compute_excess_return

            benchmark_data_type = f"index_daily_{args.benchmark.replace('.', '_')}"
            benchmark_data = load_parquet(
                benchmark_data_type, start=args.start, end=args.end
            ).collect()
            benchmark_result = compute_excess_return(
                bt_result.returns,
                args.benchmark,
                args.start,
                args.end,
                benchmark_data=benchmark_data,
            )
            logger.info(f"Benchmark: {benchmark_result.summary()}")
        except Exception as e:
            logger.warning(f"Benchmark 计算失败（跳过）: {e}")
    progress.advance("benchmark")

    # ── 13. HTML 报告（当 --benchmark 提供时生成，或始终生成）──
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    llm_explanation, llm_explanation_path = generate_llm_explanation(
        enabled=args.llm_explain,
        refresh=args.llm_refresh,
        cache_dir=result_output_dir,
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
        quality_report=quality_report,
        backtest_direction=None,
    )
    progress.advance("llm")
    with timer.stage("报告生成"):
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
            attribution_result=attribution_result,
            event_study_result=event_study_result,
            walk_forward_result=walk_forward_result,
            walk_forward_summary=walk_forward_summary,
            pearson_ic_result=pearson_ic_result if args.ic_method in ("pearson", "both") else None,
            neutralized_ic_result=neutralized_ic_result if args.neutralized_ic else None,
            llm_explanation=llm_explanation.to_dict() if llm_explanation is not None else None,
            strategy_results=strategy_results,
            primary_strategy=effective_config.backtest.primary,
            quality_report=quality_report,
        )
    report_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_output_dir / f"{factor.name}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"报告已生成: {report_path}")
    progress.advance("report")

    outputs = {
        "factor": str(factor_path),
        "ic": str(result_output_dir / f"{factor.name}_{args.start}_{args.end}_ic.parquet"),
        "quality_report": str(quality_path),
        "walk_forward_summary": str(walk_forward_path),
        "universe_snapshot": str(universe_snapshot_path),
        "report": str(report_path),
    }
    if llm_explanation_path is not None:
        outputs["llm_explanation"] = str(llm_explanation_path)
    if getattr(args, "metrics_out", None):
        _write_run_metrics(args.metrics_out, ic_result, bt_result)
    progress.close()
    return outputs


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
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印最终配置和输出目录，不执行评估"
    )
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="覆盖任意配置字段（校验前注入），可多次：--set backtest.top_n=30 --set preprocessing.neutralize=true",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        dest="metrics_out",
        help=argparse.SUPPRESS,  # 内部接口：评估完写 IC + 主策略回测指标 JSON，供 factor sweep 读取
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="启用单因子深度评估预设：both IC、neutralized IC、事件研究、按 universe 匹配 benchmark、LLM 解读",
    )
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
    parser.add_argument(
        "--llm-explain",
        action="store_true",
        help="启用大模型因子解读；无 YAML 默认配置会自动启用，缺少 FACTORZEN_LLM_* 配置时跳过",
    )
    parser.add_argument(
        "--llm-refresh",
        action="store_true",
        help="启用 --llm-explain 时忽略已有 LLM 解读缓存并重新生成",
    )
    args = parser.parse_args()

    # ── 0. 加载 YAML 配置（可选），CLI 参数优先级更高 ──
    run_config = None
    if args.config is None and args.all and args.factor:
        try:
            default_config = _find_default_run_config_path(args.factor, args.frequency)
        except (ImportError, ValueError) as e:
            logger.error(str(e))
            sys.exit(2)
        if default_config is not None:
            args.config = str(default_config)
            logger.info(f"自动加载默认运行配置: {default_config}")

    overrides = args.set_overrides or []
    try:
        if args.config:
            from factorzen.core.config_loader import load_run_config

            run_config = load_run_config(args.config, overrides=overrides)
        elif overrides and args.factor and args.start and args.end:
            from factorzen.core.config_loader import build_run_config_from_dict

            base = build_default_daily_research_config(
                factor=args.factor,
                start=args.start,
                end=args.end,
                universe=args.universe,
                benchmark=args.benchmark,
                seed=args.seed,
            ).model_dump()
            if any(item.partition("=")[0].strip() == "backtest.top_n" for item in overrides):
                base["backtest"].pop("primary", None)
                base["backtest"].pop("strategies", None)
            run_config = build_run_config_from_dict(base, overrides=overrides)
            args._uses_builtin_default_config = True
    except (ImportError, ValueError) as e:
        logger.error(str(e))
        sys.exit(2)
    except ValidationError as e:
        logger.error(f"配置校验失败:\n{e}")
        sys.exit(2)

    try:
        args = _merge_run_config_args(args, run_config)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)

    effective_config = _effective_run_config(args, run_config)
    if args.dry_run:
        print(
            json.dumps(
                _build_dry_run_payload(effective_config, args=args), ensure_ascii=False, indent=2
            )
        )
        return

    try:
        with run_experiment(effective_config, command=sys.argv) as exp_dir:
            timer = StageTimer()
            try:
                outputs = _run(args, effective_config, timer=timer)
            except Exception:
                record_experiment_metadata(exp_dir, "stage_timings", timer.timings)
                for name, path in _existing_run_outputs(args.factor, args.start, args.end).items():
                    record_experiment_output(exp_dir, name, path)
                raise
            record_experiment_metadata(exp_dir, "stage_timings", timer.timings)
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
