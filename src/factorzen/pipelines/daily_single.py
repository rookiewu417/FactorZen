"""日频单因子完整评估。用法: python factorzen.pipelines.daily_single --factor momentum_20d --start 20250101 --end 20250513"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from factorzen.config.research import (
    RunConfig,
    build_default_daily_research_config,
    default_benchmark_for_universe,
)
from factorzen.config.settings import (
    ROOT,
    daily_factor_output_dir,
    daily_report_output_dir,
    daily_result_output_dir,
)
from factorzen.core.calendar import get_trade_dates
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
from factorzen.core.universe import build_is_st_by_date, get_universe
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.evaluation.backtest import run_strategy_backtest, trim_backtest_to_first_trade
from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from factorzen.daily.evaluation.signal_backtest import run_signal_backtest
from factorzen.daily.evaluation.turnover import compute_turnover
from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary
from factorzen.daily.factors.registry import get_factor
from factorzen.daily.runtime import (
    build_backtest_strategies,
    build_cost_model,
    build_preprocessing_pipeline,
    build_runtime_backtest_config,
)
from factorzen.experiments.run_paths import copy_outputs_to_run_dir
from factorzen.pipelines._report_direction import (
    _apply_backtest_direction,
    _decide_backtest_direction,
)
from factorzen.pipelines._report_persistence import _meta_path
from factorzen.reports.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)


def filter_frame_by_membership(
    df: pl.DataFrame,
    membership: pl.DataFrame,
) -> pl.DataFrame:
    """按逐日 PIT membership 过滤评估截面：只保留当日成分 ``(trade_date, ts_code)``。

    ``membership`` 列 ``trade_date`` 为 Utf8 YYYYMMDD；``df.trade_date`` 可能是
    Date / Utf8 / Datetime——对齐 dtype 后再 inner join（口径同 factor_mine._attach_in_universe）。
    空 membership → 返回同 schema 空表。
    """
    if df.is_empty():
        return df
    if membership.is_empty():
        return df.clear()

    mem = membership.select(["trade_date", "ts_code"]).unique()
    td_dtype = df.schema.get("trade_date")
    if td_dtype == pl.Date:
        mem = mem.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"))
    elif td_dtype is not None and td_dtype != pl.Utf8:
        mem = mem.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").cast(td_dtype)
        )
    return df.join(mem, on=["trade_date", "ts_code"], how="inner")


def load_pit_membership(
    start: str,
    end: str,
    universe_name: str,
) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    """取评估窗逐日 PIT membership + union ts_codes + 行业元数据。

    Returns
    -------
    membership
        ``[trade_date(Utf8), ts_code]`` 逐日成分。
    ts_codes
        窗口内曾在成分内的并集（供 FactorDataContext 拉取，保证滚动连续）。
    universe_meta
        期末 ``get_universe`` 快照（含 industry，供中性化/分层 IC/归因；
        调出股可能不在此表——行业缺失时下游 left-join 跳过，不回退 end-only 成分过滤）。

    Raises
    ------
    ValueError
        ``get_universe_membership`` 对动态池/未知池抛错时原样上抛（不静默回退期末快照）。
    RuntimeError
        命名指数 membership 为空（成分未回补）；``all_a`` 空池允许（全市场语义）。
    """
    from factorzen.core.universe import get_universe_membership

    try:
        membership = get_universe_membership(start, end, universe_name)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"universe={universe_name!r} 的逐日 PIT membership 构造失败"
            f"（{type(exc).__name__}: {exc}）；"
            f"拒绝回退期末快照（会引入 look-ahead+幸存偏差）。"
            f"请回补指数成分数据，或改用 all_a / csi300 / csi500 / csi800。"
        ) from exc

    ts_codes = membership["ts_code"].unique().to_list() if not membership.is_empty() else []
    if not ts_codes and universe_name != "all_a":
        raise RuntimeError(
            f"universe={universe_name!r} 在 [{start},{end}] 的逐日 PIT membership 为空"
            f"（成分数据未回补）；拒绝用期末快照冒充评估池。"
        )

    # 行业元数据：期末快照（secondary；评估截面过滤以 membership 为准）
    universe_meta = get_universe(end, universe_name)
    return membership, ts_codes, universe_meta


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

    if args.universe is None:
        # 与研究预设默认一致（csi500）；report 侧 _report_config 同款兜底，改一处必查另一处
        args.universe = "csi500"
    if args.benchmark is None:
        args.benchmark = default_benchmark_for_universe(args.universe)

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
    return base.model_copy(
        update={
            "factor": args.factor,
            "start": args.start,
            "end": args.end,
            "universe": args.universe,
            "benchmark": args.benchmark or base.benchmark,
            "seed": args.seed,
        }
    )


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
    # args 保留形参以兼容调用方；execution 深度旗标已移除
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


def _build_forward_return_frame(
    daily: pl.DataFrame,
    *,
    exec_lag: int = 0,
    exec_price_col: str | None = None,
) -> pl.DataFrame:
    """Build IC forward-return labels, preferring adjusted close when available.

    ``exec_lag`` / ``exec_price_col`` 透传 ``compute_fwd_returns``。
    CLI（``fz factor run`` / daily_single）默认可实现口径 1 / open_adj；
    本 helper 内部默认仍 0 / None，便于单测断言旧 close→close 行为。
    """
    # 可实现口径：显式成交价列必须存在——缺列 fail-loudly，禁止静默退回 close→close
    if exec_price_col is not None:
        if exec_price_col not in daily.columns:
            raise ValueError(
                f"_build_forward_return_frame: exec_price_col={exec_price_col!r} 不在 daily 列中；"
                f"实际列为 {list(daily.columns)}。"
                "可实现口径要求该列存在，不会静默回退到 close→close。"
            )
        ret_df = (
            daily.select(["trade_date", "ts_code", exec_price_col])
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (
                    pl.col(exec_price_col) / pl.col(exec_price_col).shift(1).over("ts_code") - 1
                ).alias("ret")
            )
        )
        return compute_fwd_returns(
            ret_df, ret_col="ret", price_col=exec_price_col,
            exec_lag=exec_lag, exec_price_col=None,
        )

    if "close_adj" not in daily.columns:
        price_col = "close"
        ret_df = daily.select(["trade_date", "ts_code", price_col]).sort(["ts_code", "trade_date"])
        ret_df = ret_df.with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1).over("ts_code") - 1).alias("ret")
        )
        return compute_fwd_returns(
            ret_df, ret_col="ret", price_col=price_col,
            exec_lag=exec_lag, exec_price_col=None,
        )

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
    return compute_fwd_returns(
        ret_df, ret_col="ret", price_col=price_col,
        exec_lag=exec_lag, exec_price_col=None,
    )


def _compute_monotonicity_result(
    backtest_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    *,
    n_groups: int = 5,
) -> Any | None:
    """方向对齐后的信号 + fwd_ret_1d 单调性；失败返回 None。"""
    try:
        from factorzen.daily.evaluation.advanced import compute_monotonicity

        mono_df = backtest_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        mono = compute_monotonicity(
            mono_df,
            factor_col="factor_clean",
            ret_col="fwd_ret_1d",
            n_groups=n_groups,
        )
        logger.info(f"单调性: score={mono.monotonicity_score:.3f}")
        return mono
    except Exception as e:
        logger.warning(f"单调性分析失败（跳过）: {e}")
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
    # PIT 收窄 ST 股票涨跌停阈值（4.8% 而非主板 9.8%，见
    # core/universe.py::_get_board_limit）；只构建一次，全程复用。
    codes = daily.select("ts_code").unique()["ts_code"].to_list()
    trade_dates_list = daily.select("trade_date").unique()["trade_date"].to_list()
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
    progress = OverallProgress(14, label="Daily run").start()
    # ── 0b. 设置全局随机种子（可选）──
    if args.seed is not None:
        from factorzen.core.seed import set_global_seed

        set_global_seed(args.seed)
        logger.info(f"全局随机种子已设置: {args.seed}")

    # ── 1. 获取因子类 ──
    logger.info(f"──── 单因子评估: {args.factor} | {args.start} ~ {args.end} ────")
    # ashare daily 管线：注入 factor_library expression 型（库损坏/缺失不崩 run）
    # import 放函数内：daily→discovery 反向依赖禁止在模块级出现（架构环）
    from factorzen.discovery.library_provider import load_library_factors
    try:
        load_library_factors()
    except ValueError as e:
        logger.warning(f"load_library_factors 跳过: {e}")
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
            ),
            is_qlib_factor=getattr(factor, "category", "") == "qlib"
            or factor.name.startswith("qlib_"),
        )
    except Exception as e:
        logger.error(f"数据保障失败: {e}")
        raise RuntimeError(f"ensure_data_for_daily_run failed: {e}") from e
    progress.advance("data")

    # ── 3. 股票池（逐日 PIT membership；union 供拉取，评估截面再按日过滤）──
    try:
        membership, ts_codes, universe = load_pit_membership(
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

    # ── 3b. 保存逐日 membership（供复现和审计；口径=PIT，非期末快照）──
    result_output_dir.mkdir(parents=True, exist_ok=True)
    universe_snapshot_path = (
        result_output_dir / f"{factor.name}_{args.start}_{args.end}_universe.parquet"
    )
    membership.write_parquet(str(universe_snapshot_path))
    logger.info(
        f"Universe membership 已保存: {universe_snapshot_path} "
        f"(rows={membership.height}, union={len(ts_codes)})"
    )
    progress.advance("universe")

    # ── 4. 计算因子 ──
    ctx = FactorDataContext(
        start=args.start,
        end=args.end,
        required_data=factor.required_data,
        lookback_days=factor.lookback_days,
        universe=ts_codes if ts_codes else None,
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

    # ── 5. 预处理（先预处理再按日 PIT 过滤评估截面——预处理可在 union 上做，
    #    截面统计更稳；IC/回测/换手只看当日成分）──
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
    clean_df = filter_frame_by_membership(clean_df, membership)
    if clean_df.is_empty():
        logger.error("PIT membership 过滤后因子截面为空")
        raise RuntimeError("empty factor cross-section after PIT membership filter")
    logger.info(
        f"预处理完成 (去极值 → 填充 → 标准化 → 逐日 PIT 过滤, n={clean_df.height})"
    )
    progress.advance("preprocess")

    # ── 6. 计算前向收益 ──
    daily = ctx.daily.collect()
    if daily.is_empty():
        logger.error("日线数据为空，无法计算收益")
        raise RuntimeError("empty daily data")
    ret_df = _build_forward_return_frame(
        daily,
        exec_lag=int(getattr(args, "exec_lag", 1) if getattr(args, "exec_lag", None) is not None else 1),
        exec_price_col=getattr(args, "exec_price_col", "open_adj"),
    )
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

    # ── 7. IC 分析（始终基于原始 factor_clean，不翻号）──
    with timer.stage("IC 分析"):
        ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
    ic_result.factor_name = factor.name
    logger.info(f"\n{ic_result.summary()}")
    progress.advance("ic")

    # 与 generate_report 对齐：显著负 IC 时回测用反向信号（做多低因子值）
    backtest_direction = _decide_backtest_direction(ic_result)
    logger.info(
        f"回测方向判定: {backtest_direction.get('direction')} | "
        f"{backtest_direction.get('reason')}"
    )
    backtest_df = _apply_backtest_direction(clean_df, backtest_direction)

    # ── 8. 策略回测（IC 对齐方向后的信号）──
    with timer.stage("策略回测"):
        bt_result, _ = _run_backtest_strategies(
            effective_config,
            backtest_df,
            daily,
            factor_name=factor.name,
            frequency=args.frequency,
        )
    progress.advance("backtest")

    # ── 9. 换手率（与回测同一信号口径）──
    with timer.stage("换手率"):
        to_result = compute_turnover(backtest_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")
    progress.advance("turnover")

    factor_output_dir.mkdir(parents=True, exist_ok=True)
    result_output_dir.mkdir(parents=True, exist_ok=True)

    # 落盘原始因子值（语义定义，不翻号）；回测方向写入 meta 供 report --reuse
    factor_path = factor_output_dir / f"{factor.name}_{args.start}_{args.end}.parquet"
    clean_df.write_parquet(str(factor_path))
    logger.info(f"因子已保存: {factor_path}")

    ic_path = result_output_dir / f"{factor.name}_{args.start}_{args.end}_ic.parquet"
    ic_result.ic_series.write_parquet(str(ic_path))
    logger.info(f"IC 序列已保存: {ic_path}")

    meta_path = _meta_path(factor.name, args.start, args.end)
    meta_payload: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_payload = {}
    meta_payload.update(
        {
            "factor_name": factor.name,
            "start": args.start,
            "end": args.end,
            "ic_mean": ic_result.ic_mean,
            "ic_tstat": ic_result.ic_tstat,
            "ic_pvalue": ic_result.ic_pvalue,
            "ir": ic_result.ir,
            "n_periods": ic_result.n_periods,
            "backtest_direction": backtest_direction,
        }
    )
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"回测方向已写入 meta: {meta_path}")
    progress.advance("save-core")

    # ── 10. Walk-forward / OOS 摘要（与回测同一信号口径）──
    if effective_config.walk_forward.enabled:
        with timer.stage("Walk-forward"):
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
    progress.advance("walk-forward")

    # ── 11. 单调性（与回测同一信号口径）──
    mono_result = _compute_monotonicity_result(backtest_df, ret_df, n_groups=5)

    walk_forward_path = result_output_dir / f"{factor.name}_{args.start}_{args.end}_walk_forward.json"
    walk_forward_path.write_text(
        json.dumps(walk_forward_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Walk-forward 摘要已保存: {walk_forward_path}")
    # 回填 meta 中的 walk_forward（方向判定已先写入）
    try:
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_payload["walk_forward_summary"] = walk_forward_summary
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"更新 meta walk_forward 失败（跳过）: {e}")
    progress.advance("mono")

    # ── 11b. 信号层回测 ──
    # 增量产出：失败只 warning 跳过，不拖垮主管线；不改 progress 序列。
    try:
        _sig_exec_lag = int(
            getattr(args, "exec_lag", 1)
            if getattr(args, "exec_lag", None) is not None
            else 1
        )
        _sig_exec_price_col = getattr(args, "exec_price_col", "open_adj")
        signal_result = run_signal_backtest(
            backtest_df,  # 方向对齐后的信号（与回测/换手同口径）
            ret_df,  # 第 6 步已算好的前向收益（exec_lag/exec_price_col 可实现口径，自动继承）
            factor_col="factor_clean",
            n_groups=5,
            frequency=args.frequency,
            factor_name=factor.name,
            meta={
                "exec_lag": _sig_exec_lag,
                "exec_price_col": _sig_exec_price_col,
                "direction": backtest_direction.get("direction"),
            },
        )
        logger.info(f"\n{signal_result.summary()}")
        signal_json_path = (
            result_output_dir / f"{factor.name}_{args.start}_{args.end}_signal.json"
        )
        signal_json_path.write_text(
            json.dumps(
                {
                    "summary_stats": signal_result.summary_stats,
                    "meta": signal_result.meta,
                    "n_groups": signal_result.n_groups,
                    "cost_bps": signal_result.cost_bps,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        signal_nav_path = (
            result_output_dir
            / f"{factor.name}_{args.start}_{args.end}_signal_group_nav.parquet"
        )
        signal_result.group_nav.write_parquet(str(signal_nav_path))
        logger.info(f"信号层回测已保存: {signal_json_path}, {signal_nav_path}")
    except Exception as e:
        logger.warning(f"信号层回测失败（跳过）: {e}")

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

    # ── 13. HTML 报告 ──
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    with timer.stage("报告生成"):
        html = generate_tear_sheet(
            factor.name,
            ic_result,
            bt_result,
            to_result,
            frequency=args.frequency,
            date_range=date_range,
            universe=args.universe,
            mono_result=mono_result,
            benchmark_result=benchmark_result,
            backtest_direction=backtest_direction,
            walk_forward_summary=walk_forward_summary,
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
        "meta": str(meta_path),
    }
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
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子（默认 42）")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="覆盖任意配置字段（校验前注入），可多次：--set backtest.top_n=30 --set preprocessing.neutralize=true",
    )
    parser.add_argument(
        "--exec-lag", dest="exec_lag", type=int, default=1,
        help="成交滞后(交易日)。默认 1=可实现口径；0=旧 close→close（不可实现，仅对照用）",
    )
    parser.add_argument(
        "--exec-price-col", dest="exec_price_col", default="open_adj",
        help="成交价格列。默认 open_adj（可实现口径 open[t+2]/open[t+1]）",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        dest="metrics_out",
        help=argparse.SUPPRESS,  # 内部接口：评估完写 IC + 主策略回测指标 JSON，供 factor sweep 读取
    )
    args = parser.parse_args()

    # ── 0. 加载 YAML 配置（可选），CLI 参数优先级更高 ──
    run_config = None
    overrides = args.set_overrides or []
    try:
        if args.config:
            from factorzen.config.research import load_run_config

            run_config = load_run_config(args.config, overrides=overrides)
        elif overrides and args.factor and args.start and args.end:
            from factorzen.config.research import build_run_config_from_dict

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
            # evidence 链接：评估成功后挂 run_id 到库（非裁决指标；失败只 warning）
            try:
                from datetime import datetime

                from factorzen.discovery.factor_library import link_evaluation_to_library

                linked = link_evaluation_to_library(
                    args.factor,
                    exp_dir.name,
                    datetime.now().date().isoformat(),
                    market="ashare",
                )
                if not linked:
                    logger.warning(
                        "link_evaluation_to_library 未挂上 name=%s run_id=%s",
                        args.factor, exp_dir.name,
                    )
            except Exception as link_exc:
                logger.warning(
                    "link_evaluation_to_library 异常 name=%s: %s: %s",
                    args.factor, type(link_exc).__name__, link_exc,
                )
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Done.")


if __name__ == "__main__":
    main()
