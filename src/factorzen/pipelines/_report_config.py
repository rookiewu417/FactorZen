"""report 流程的配置合并：CLI 参数与 YAML RunConfig 的优先级处理。"""

from __future__ import annotations

import argparse

from factorzen.core.config_loader import (
    RunConfig,
    default_benchmark_for_universe,
    with_default_all_strategies,
)


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
        # --all help 承诺启用 reuse（复用缓存的深度结果）；此前漏置→与 help 漂移、
        # 用户期待复用却全量重算。reuse 有优雅回退（无缓存则完整计算），置 True 安全。
        if not getattr(args, "reuse", False):
            args.reuse = True

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
