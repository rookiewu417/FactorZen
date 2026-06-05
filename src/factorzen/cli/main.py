"""Unified FactorZen command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from factorzen.config.settings import FACTOR_EVALUATIONS_DIR, ROOT
from factorzen.experiments.run_paths import run_dir


def _factor_template(class_name: str, factor_name: str, frequency: str) -> str:
    base = "DailyFactor" if frequency != "intraday" else "IntradayFactor"
    import_path = (
        "factorzen.daily.factors.base"
        if frequency != "intraday"
        else "factorzen.intraday.factors.base"
    )
    context_type = "FactorDataContext" if frequency != "intraday" else "IntradayDataContext"
    context_import = (
        "factorzen.daily.data.context"
        if frequency != "intraday"
        else "factorzen.intraday.data.context"
    )
    time_col = "trade_date" if frequency != "intraday" else "trade_time"
    source = "ctx.daily" if frequency != "intraday" else "ctx.minute"
    return f'''"""User factor: {factor_name}."""

import polars as pl

from {context_import} import {context_type}
from {import_path} import {base}


class {class_name}({base}):
    name = "{factor_name}"
    frequency = "{frequency}"
    description = "{factor_name}"

    def compute(self, ctx: {context_type}) -> pl.DataFrame:
        frame = {source}
        return (
            frame.select(["{time_col}", "ts_code"])
            .with_columns(pl.lit(None, dtype=pl.Float64).alias("factor_value"))
            .collect()
        )
'''


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) + "Factor"


def _cmd_factor_new(args: argparse.Namespace) -> int:
    target = ROOT / "workspace" / "factors" / args.freq / f"{args.name}.py"
    if target.exists() and not args.force:
        print(f"Factor already exists: {target}", file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _factor_template(_class_name(args.name), args.name, args.freq),
        encoding="utf-8",
    )
    print(target)
    return 0


def _cmd_factor_list(args: argparse.Namespace) -> int:
    if args.freq == "intraday":
        from factorzen.intraday.factors.registry import list_factors
    else:
        from factorzen.daily.factors.registry import list_factors

    for name in list_factors():
        print(name)
    return 0


def _cmd_factor_test(args: argparse.Namespace) -> int:
    from factorzen.pipelines import daily_single

    forwarded = [f"fz factor {args.factor_command}"]
    if args.name:
        forwarded.extend(["--factor", args.name])
    if args.start:
        forwarded.extend(["--start", args.start])
    if args.end:
        forwarded.extend(["--end", args.end])
    if args.universe:
        forwarded.extend(["--universe", args.universe])
    forwarded.extend(["--frequency", args.frequency])
    if args.config:
        forwarded.extend(["--config", args.config])
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    if args.benchmark:
        forwarded.extend(["--benchmark", args.benchmark])
    if args.ic_method:
        forwarded.extend(["--ic-method", args.ic_method])
    if args.neutralized_ic:
        forwarded.append("--neutralized-ic")
    if args.event_study:
        forwarded.append("--event-study")
    if args.llm_explain:
        forwarded.append("--llm-explain")
    if args.llm_refresh:
        forwarded.append("--llm-refresh")
    if args.all:
        forwarded.append("--all")
    if args.dry_run:
        forwarded.append("--dry-run")
    for override in getattr(args, "set_overrides", None) or []:
        forwarded.extend(["--set", override])

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        daily_single.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_factor_sweep(args: argparse.Namespace) -> int:
    from datetime import datetime

    from factorzen.config.settings import FACTOR_EVALUATIONS_DIR
    from factorzen.pipelines.factor_sweep import (
        format_sweep_csv,
        format_sweep_table,
        pipeline_runner,
        run_sweep,
    )

    factor = args.name
    start, end, universe = args.start, args.end, args.universe
    if args.config:
        from factorzen.core.config_loader import load_run_config

        cfg = load_run_config(args.config)
        factor = factor or cfg.factor
        start = start or cfg.start
        end = end or cfg.end
        universe = universe or cfg.universe

    if not (factor and start and end):
        print("sweep 需要 factor 与 start/end（经位置参数/--config/CLI 提供）", file=sys.stderr)
        return 2
    if not args.grid:
        print("sweep 需要至少一个 --grid key=v1,v2,...", file=sys.stderr)
        return 2

    runner = pipeline_runner(
        factor=factor,
        start=start,
        end=end,
        config_path=args.config,
        universe=universe,
    )
    rows = run_sweep(
        args.grid,
        runner,
        sort_by=args.sort_by,
        extra_overrides=args.set_overrides,
    )
    print(format_sweep_table(rows))

    out_dir = FACTOR_EVALUATIONS_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_results.csv"
    csv_path.write_text(format_sweep_csv(rows), encoding="utf-8")
    print(f"\n结果已保存: {csv_path}")
    return 0


def _cmd_report_build(args: argparse.Namespace) -> int:
    from factorzen.pipelines import generate_report

    factor_name = args.name or args.factor
    forwarded = [f"fz report {args.report_command}"]
    if factor_name:
        forwarded.extend(["--factor", factor_name])
    if args.start:
        forwarded.extend(["--start", args.start])
    if args.end:
        forwarded.extend(["--end", args.end])
    if args.universe:
        forwarded.extend(["--universe", args.universe])
    forwarded.extend(["--frequency", args.frequency])
    if args.benchmark:
        forwarded.extend(["--benchmark", args.benchmark])
    if args.config:
        forwarded.extend(["--config", args.config])
    if args.reuse:
        forwarded.append("--reuse")
    if args.ic_method:
        forwarded.extend(["--ic-method", args.ic_method])
    if args.neutralized_ic:
        forwarded.append("--neutralized-ic")
    if args.event_study:
        forwarded.append("--event-study")
    if args.llm_explain:
        forwarded.append("--llm-explain")
    if args.llm_refresh:
        forwarded.append("--llm-refresh")
    if args.all:
        forwarded.append("--all")

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        generate_report.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_report_open(args: argparse.Namespace) -> int:
    report = run_dir(args.run_id) / "report.html"
    if not report.exists():
        print(f"Report not found: {report}", file=sys.stderr)
        return 2
    print(report)
    return 0


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    from factorzen.core import loader

    if args.data_type == "daily":
        frame = loader.fetch_daily(args.start, args.end)
    else:
        frame = loader.fetch_daily_basic(args.start, args.end)
    rows = len(frame) if hasattr(frame, "__len__") else "unknown"
    print(f"{args.data_type}: {rows} rows")
    return 0


def _cmd_config_validate(args: argparse.Namespace) -> int:
    from factorzen.core.config_loader import default_benchmark_for_universe, load_run_config

    config = load_run_config(args.path)
    benchmark = config.benchmark or default_benchmark_for_universe(config.universe)
    effective = config.model_copy(update={"benchmark": benchmark})
    payload = {
        "config": effective.model_dump(),
        "output_dir": (ROOT / "workspace" / "factor_evaluations" / "<run_id>").as_posix(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_runs_list(args: argparse.Namespace) -> int:
    index_path = FACTOR_EVALUATIONS_DIR / "experiment_index.jsonl"
    if not index_path.exists():
        print(f"No runs index found: {index_path}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if args.limit:
        rows = rows[-args.limit :]

    print("run_id\tstatus\tfactor\tuniverse\ttimestamp")
    for row in rows:
        print(
            "\t".join(
                str(row.get(key, ""))
                for key in ("run_id", "status", "factor", "universe", "timestamp")
            )
        )
    return 0


def _cmd_runs_show(args: argparse.Namespace) -> int:
    manifest_path = FACTOR_EVALUATIONS_DIR / args.run_id / "manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _add_factor_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="Factor name")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Universe name")
    parser.add_argument(
        "--frequency",
        "--freq",
        dest="frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Factor frequency",
    )
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override any config field, repeatable: --set backtest.top_n=30",
    )
    parser.add_argument("--all", action="store_true", help="Enable deep evaluation preset")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without running")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC method",
    )
    parser.add_argument("--neutralized-ic", action="store_true", dest="neutralized_ic")
    parser.add_argument("--event-study", action="store_true", dest="event_study")
    parser.add_argument("--llm-explain", action="store_true")
    parser.add_argument("--llm-refresh", action="store_true")


def _add_report_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="Factor name")
    parser.add_argument("--factor", default=None, help="Factor name")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Universe name")
    parser.add_argument(
        "--frequency",
        "--freq",
        dest="frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Factor frequency",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse existing artifacts")
    parser.add_argument("--all", action="store_true", help="Enable deep report preset")
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC method",
    )
    parser.add_argument("--neutralized-ic", action="store_true", dest="neutralized_ic")
    parser.add_argument("--event-study", action="store_true", dest="event_study")
    parser.add_argument("--llm-explain", action="store_true")
    parser.add_argument("--llm-refresh", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fz", description="FactorZen research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    factor = sub.add_parser("factor", help="Factor workflows")
    factor_sub = factor.add_subparsers(dest="factor_command", required=True)

    new = factor_sub.add_parser("new", help="Create a user factor template")
    new.add_argument("name")
    new.add_argument(
        "--frequency",
        "--freq",
        dest="freq",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
    )
    new.add_argument("--force", action="store_true")
    new.set_defaults(func=_cmd_factor_new)

    list_cmd = factor_sub.add_parser("list", help="List registered factors")
    list_cmd.add_argument(
        "--frequency",
        "--freq",
        dest="freq",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
    )
    list_cmd.set_defaults(func=_cmd_factor_list)

    run = factor_sub.add_parser("run", help="Run a single factor evaluation")
    _add_factor_run_arguments(run)
    run.set_defaults(func=_cmd_factor_test)

    test = factor_sub.add_parser("test", help="Deprecated alias for 'factor run'")
    _add_factor_run_arguments(test)
    test.set_defaults(func=_cmd_factor_test)

    sweep = factor_sub.add_parser("sweep", help="Parameter grid sweep over --set overrides")
    sweep.add_argument("name", nargs="?", help="Factor name (or supply via --config)")
    sweep.add_argument("--config", default=None, help="Base YAML run config path")
    sweep.add_argument(
        "--grid",
        action="append",
        default=None,
        metavar="KEY=V1,V2,...",
        help="Grid dimension, repeatable: --grid backtest.top_n=30,50,100",
    )
    sweep.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Fixed override applied to every combo",
    )
    sweep.add_argument("--start", default=None, help="Start date YYYYMMDD")
    sweep.add_argument("--end", default=None, help="End date YYYYMMDD")
    sweep.add_argument("--universe", default=None, help="Universe name")
    sweep.add_argument(
        "--sort-by",
        default="ir",
        dest="sort_by",
        help="Metric to rank rows by (ir/ic_mean/ic_pos/t)",
    )
    sweep.set_defaults(func=_cmd_factor_sweep)

    report = sub.add_parser("report", help="Report workflows")
    report_sub = report.add_subparsers(dest="report_command", required=True)

    build_cmd = report_sub.add_parser("build", help="Build a factor report")
    _add_report_build_arguments(build_cmd)
    build_cmd.set_defaults(func=_cmd_report_build)

    path_cmd = report_sub.add_parser("path", help="Print report path for a run")
    path_cmd.add_argument("run_id")
    path_cmd.set_defaults(func=_cmd_report_open)

    open_cmd = report_sub.add_parser("open", help="Deprecated alias for 'report path'")
    open_cmd.add_argument("run_id")
    open_cmd.set_defaults(func=_cmd_report_open)

    data = sub.add_parser("data", help="Data workflows")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="Fetch raw data into cache")
    fetch.add_argument("data_type", choices=["daily", "daily-basic"])
    fetch.add_argument("--start", required=True, help="Start date YYYYMMDD")
    fetch.add_argument("--end", required=True, help="End date YYYYMMDD")
    fetch.set_defaults(func=_cmd_data_fetch)

    config = sub.add_parser("config", help="Config workflows")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate", help="Validate a YAML run config")
    validate.add_argument("path", help="YAML run config path")
    validate.set_defaults(func=_cmd_config_validate)

    runs = sub.add_parser("runs", help="Run history workflows")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    list_cmd = runs_sub.add_parser("list", help="List recorded runs")
    list_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows to print")
    list_cmd.set_defaults(func=_cmd_runs_list)
    show_cmd = runs_sub.add_parser("show", help="Show one run manifest")
    show_cmd.add_argument("run_id")
    show_cmd.set_defaults(func=_cmd_runs_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
