"""Run FactorZen daily reports for all registered qlib factors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from common.config_loader import RunConfig
from common.experiment import record_experiment_output, run_experiment
from config.settings import OUTPUT_DAILY_RESULTS, daily_report_output_dir
from daily.factors.registry import list_factors
from scripts.run_daily_single import _run


def _report_path(factor: str, start: str, end: str) -> Path:
    return daily_report_output_dir(factor) / f"{factor}_{start}_{end}.html"


def _namespace(args: argparse.Namespace, factor: str) -> argparse.Namespace:
    return argparse.Namespace(
        factor=factor,
        start=args.start,
        end=args.end,
        universe=args.universe,
        frequency="daily",
        benchmark=None,
        config=None,
        seed=None,
        ic_method="rank",
        neutralized_ic=False,
        event_study=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reports for all qlib daily factors.")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--universe", default="csi300", help="FactorZen universe")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N qlib factors")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N factors")
    parser.add_argument("--force", action="store_true", help="Re-run factors with existing reports")
    args = parser.parse_args()

    os.environ.setdefault("QLIB_INSTRUMENTS", args.universe)

    factors = [name for name in list_factors() if name.startswith("qlib_")]
    if args.offset:
        factors = factors[args.offset :]
    if args.limit is not None:
        factors = factors[: args.limit]

    OUTPUT_DAILY_RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.offset or args.limit is not None:
        limit_part = "all" if args.limit is None else str(args.limit)
        suffix = f"_offset{args.offset}_limit{limit_part}"
    summary_path = OUTPUT_DAILY_RESULTS / f"qlib_batch_{args.start}_{args.end}{suffix}.json"
    summary: dict[str, object] = {
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "offset": args.offset,
        "limit": args.limit,
        "total": len(factors),
        "ok": [],
        "skipped": [],
        "failed": [],
    }

    for i, factor in enumerate(factors, start=1):
        existing_report = _report_path(factor, args.start, args.end)
        if existing_report.exists() and not args.force:
            summary["skipped"].append(factor)  # type: ignore[index]
            continue

        ns = _namespace(args, factor)
        config = RunConfig(
            factor=factor,
            start=args.start,
            end=args.end,
            universe=args.universe,
            ic_method="rank",
            neutralized_ic=False,
            event_study=False,
        )
        started = perf_counter()
        try:
            with run_experiment(config, command=["scripts/run_qlib_batch_reports.py"]) as exp_dir:
                outputs = _run(ns, config)
                for name, path in outputs.items():
                    record_experiment_output(exp_dir, name, path)
            summary["ok"].append(  # type: ignore[index]
                {"factor": factor, "seconds": round(perf_counter() - started, 3)}
            )
        except Exception as exc:
            summary["failed"].append(  # type: ignore[index]
                {"factor": factor, "error": str(exc), "index": i}
            )

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = summary["failed"]  # type: ignore[index]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
