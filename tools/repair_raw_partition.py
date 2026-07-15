"""Merge missing keys from a legacy parquet snapshot into a production raw dataset."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from factorzen.config.settings import DATA_RAW, WORKSPACE_DIR
from factorzen.core.experiment import get_git_sha
from factorzen.dataio.partition_repair import merge_missing_partition_rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只补目标缺失键，不覆盖现有 raw 值")
    parser.add_argument("source", type=Path, help="旧快照/备份的数据类型根目录")
    parser.add_argument("--target-data-type", required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_RAW)
    parser.add_argument("--date-col", default="trade_date")
    parser.add_argument("--key", action="append", dest="keys")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = datetime.now().astimezone()
    run_id = args.run_id or started.strftime("repair_%Y%m%d_%H%M%S")
    run_dir = WORKSPACE_DIR / "data_maintenance" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    keys = tuple(args.keys or [args.date_col, "ts_code"])

    try:
        report = merge_missing_partition_rows(
            args.source,
            target_data_type=args.target_data_type,
            base_dir=args.data_root,
            key_cols=keys,
            date_col=args.date_col,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    finished = datetime.now().astimezone()
    payload = {
        "run_id": run_id,
        "command": shlex.join(["tools/repair_raw_partition.py", *(argv or sys.argv[1:])]),
        "git_sha": get_git_sha(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "source": str(args.source.resolve()),
        "target": str((args.data_root / args.target_data_type).resolve()),
        "date_col": args.date_col,
        "key_cols": list(keys),
        "seed": None,
        "universe": "source snapshot symbols",
        "window": {"start": str(report.min_date), "end": str(report.max_date)},
        "result": asdict(report),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "repair.done").touch()
    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
