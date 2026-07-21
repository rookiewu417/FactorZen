"""Ingest external minute parquet files into ``data/raw/minute_1min``.

Examples:
    pixi run python tools/ingest_minute.py data/_gapfill/minute/1min
    pixi run python tools/ingest_minute.py /mnt/e/BaiduNetdiskDownload --month 202001
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from factorzen.config.settings import DATA_RAW, WORKSPACE_OPS_DIR
from factorzen.core.experiment import get_git_sha
from factorzen.dataio.minute_ingest import discover_parquet_files, ingest_minute_files


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一导入按日或按股票布局的 A 股 1min parquet")
    parser.add_argument("source", type=Path, help="parquet 文件或递归源目录")
    parser.add_argument(
        "--month",
        action="append",
        dest="months",
        help="只导入 YYYYMM，可重复；默认导入源内全部月份",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_RAW,
        help=f"生产 raw 根目录（默认 {DATA_RAW}）",
    )
    parser.add_argument("--run-id", default=None, help="manifest/sentinel run id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = datetime.now().astimezone()
    run_id = args.run_id or started.strftime("minute_ingest_%Y%m%d_%H%M%S")
    run_dir = WORKSPACE_OPS_DIR / "data_ingest" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    command_args = list(argv or sys.argv[1:])
    try:
        files = discover_parquet_files(args.source)
        report = ingest_minute_files(
            files,
            base_dir=args.data_root,
            months=args.months,
        )
    except ValueError as exc:
        payload = {
            "run_id": run_id,
            "status": "failed",
            "command": shlex.join(["tools/ingest_minute.py", *command_args]),
            "git_sha": get_git_sha(),
            "started_at": started.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "source": str(args.source.resolve()),
            "target": str((args.data_root / "minute_1min").resolve()),
            "seed": None,
            "universe": "source symbols",
            "window": {"months": args.months},
            "error": str(exc),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "ingest.done").touch()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {
        "run_id": run_id,
        "status": "success",
        "command": shlex.join(["tools/ingest_minute.py", *command_args]),
        "git_sha": get_git_sha(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "source": str(args.source.resolve()),
        "target": str((args.data_root / "minute_1min").resolve()),
        "seed": None,
        "universe": "source symbols",
        "window": {"months": sorted(report.rows_by_month)},
        "result": asdict(report),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "ingest.done").touch()
    print(f"源文件: {report.source_files}")
    for month, rows in report.rows_by_month.items():
        print(f"  {month}: {rows} 行已合并")
    print(f"完成: {report.total_rows} 源行 → {args.data_root / 'minute_1min'}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
