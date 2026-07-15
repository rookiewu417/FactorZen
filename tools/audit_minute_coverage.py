"""A 股分钟湖覆盖审计 CLI：交易日历 vs 分区 present days。

用法::

    pixi run -- python tools/audit_minute_coverage.py --start 20170101 --end 20201231
    pixi run -- python tools/audit_minute_coverage.py --start 20190101 --end 20191231 --json

退出码：0=ok；1=error；2=仅 warning（有 missing_ranges）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from factorzen.config.settings import DATA_RAW_MINUTE, REPORTS_DIR
from factorzen.intraday.audit import coverage_report

_DEFAULT_OUT = REPORTS_DIR / "minute_audit"


def run_coverage(
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """跑 coverage_report 并附加 status 字段。"""
    report = coverage_report(start, end, base_dir=base_dir)
    if report["missing_ranges"]:
        status = "warning"
    else:
        status = "ok"
    report["status"] = status
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A 股分钟湖覆盖审计")
    p.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    p.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    p.add_argument("--json", action="store_true", help="同时向 stdout 打印 JSON")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="报告输出目录",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="分钟湖根目录（默认 data/raw/minute_1min）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = args.base_dir if args.base_dir is not None else DATA_RAW_MINUTE
    report = run_coverage(args.start, args.end, base_dir=base)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"minute_coverage_audit_{ts}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    status = report["status"]
    print("=" * 60)
    print("FactorZen 分钟湖覆盖审计")
    print("=" * 60)
    print(f"窗口: {args.start} ~ {args.end}")
    print(
        f"期望交易日 {report['n_expected_days']}  /  "
        f"有数据 {report['n_present_days']}  /  "
        f"缺失区间 {len(report['missing_ranges'])}"
    )
    for lo, hi in report["missing_ranges"]:
        print(f"  · missing: {lo} ~ {hi}")
    print(f"有数据月份: {len(report['months_present'])}")
    print(f"报告: {out_path}")
    print("=" * 60)
    print(f"总体: {status.upper()}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    return {"ok": 0, "error": 1, "warning": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
