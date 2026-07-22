"""report 流程的配置合并：CLI 参数与 YAML RunConfig 的优先级处理。"""

from __future__ import annotations

import argparse

from factorzen.config.research import (
    RunConfig,
    build_default_daily_research_config,
    default_benchmark_for_universe,
)


def _merge_report_config_args(args: argparse.Namespace, run_config: RunConfig | None):
    """Merge YAML config into report CLI args without overriding explicit CLI values."""
    if run_config is not None:
        for field in ("factor", "start", "end", "universe"):
            if getattr(args, field, None) is None:
                setattr(args, field, getattr(run_config, field))
        if args.benchmark is None and run_config.benchmark is not None:
            args.benchmark = run_config.benchmark

    if args.universe is None:
        # 与 fz factor eval/backtest 的无 YAML 研究预设一致（csi500）；此前默认 csi300
        # 会让 report 全量重算路径与 daily_single 用不同股票池（双路径漂移）。
        args.universe = "csi500"
    if args.benchmark is None:
        args.benchmark = default_benchmark_for_universe(args.universe)

    missing = [field for field in ("factor", "start", "end") if getattr(args, field, None) is None]
    if missing:
        raise ValueError(f"缺少必填参数: {', '.join(missing)}（可通过 CLI 或 --config 提供）")
    return args


def _effective_report_config(args: argparse.Namespace, run_config: RunConfig | None) -> RunConfig:
    # 无 YAML 时与 daily_single 用同一份研究预设（quantile_ls_5 + 中性化预处理），
    # 避免 report 与 factor eval/backtest 两条路径口径漂移（双路径登记簿）。
    base = run_config or build_default_daily_research_config(
        factor=args.factor,
        start=args.start,
        end=args.end,
        universe=args.universe,
        benchmark=args.benchmark,
    )
    return base.model_copy(
        update={
            "factor": args.factor,
            "start": args.start,
            "end": args.end,
            "universe": args.universe,
            "benchmark": args.benchmark or base.benchmark,
        }
    )
