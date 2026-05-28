"""原始数据分区完整性审计 CLI。

不调用 Tushare，只读本地落盘数据，可在无网络/无 token 时运行。

用法::

    # 基本审计（无股票池覆盖率）
    pixi run python scripts/audit_raw_data.py --data-type daily_basic --start 20230101 --end 20231231

    # 带股票池覆盖率（需要本地缓存已有指数成分）
    pixi run python scripts/audit_raw_data.py --data-type daily --universe csi300 --start 20230101 --end 20231231

    # 只输出 JSON
    pixi run python scripts/audit_raw_data.py --data-type finance --start 20220101 --end 20231231 --json
"""

from __future__ import annotations

import argparse
import json
import sys


def _load_universe_codes(universe_name: str, end_date: str) -> list[str] | None:
    """尝试从本地缓存加载股票池，失败则返回 None（不发起 Tushare 网络请求）。"""
    try:
        from common.universe import get_universe

        codes = get_universe(end_date, universe_name)
        return codes
    except Exception as exc:
        print(
            f"[warn] 无法加载股票池 '{universe_name}': {exc}，跳过覆盖率计算",
            file=sys.stderr,
        )
        return None


def _print_summary(result: dict, data_type: str, start: str, end: str) -> None:
    status = result["status"]
    icon = {"ok": "✓", "warning": "△", "error": "✗"}.get(status, "?")
    print(f"\n{icon} [{status.upper()}] {data_type} audit  {start} ~ {end}")

    checks = result.get("checks", {})

    if "total_rows" in checks:
        print(f"  总行数:       {checks['total_rows']:,}")

    dc = checks.get("date_coverage")
    if dc:
        if "missing_count" in dc:
            print(
                f"  日期覆盖:     {dc['actual']}/{dc['expected']} 交易日"
                f"  (缺 {dc['missing_count']})"
            )
            if dc.get("missing_dates"):
                sample = dc["missing_dates"][:5]
                suffix = " ..." if dc["missing_count"] > 5 else ""
                print(f"  缺失样本:     {', '.join(sample)}{suffix}")
        else:
            print(f"  唯一 end_date: {dc.get('unique_end_dates')}")

    sc = checks.get("stock_coverage")
    if sc and "coverage" in sc:
        print(
            f"  股票覆盖:     {sc['covered']}/{sc['universe_size']}"
            f"  ({sc['coverage']:.1%})"
        )

    fr = checks.get("field_null_rates")
    if fr:
        print("  字段空值率:")
        for col, stats in fr.items():
            if stats.get("missing_column"):
                print(f"    {col:<14} 列不存在")
            elif "coverage" in stats:
                flag = "" if stats["coverage"] >= 0.8 else "  ← 低"
                print(f"    {col:<14} {stats['coverage']:.1%}{flag}")

    ps = checks.get("pit_staleness")
    if ps:
        print(
            f"  Finance PIT:  {ps['stale_count']} 只陈旧"
            f"  (阈值 {ps['threshold_date']})"
        )

    if result["warnings"]:
        print("\n  ⚠ 警告:")
        for w in result["warnings"]:
            print(f"    - {w}")

    if result["errors"]:
        print("\n  ✗ 错误:")
        for e in result["errors"]:
            print(f"    - {e}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="审计 data/raw/ 分区完整性（无需 Tushare token）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-type",
        required=True,
        choices=["daily", "daily_basic", "finance"],
        help="要审计的数据类型",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="股票池名称（csi300/csi500/all_a 等），用于股票覆盖率计算；省略则跳过",
    )
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="只输出原始 JSON，不输出可读摘要",
    )
    args = parser.parse_args()

    from common.data_audit import build_raw_data_audit

    universe_codes: list[str] | None = None
    if args.universe:
        universe_codes = _load_universe_codes(args.universe, args.end)

    result = build_raw_data_audit(
        data_type=args.data_type,
        start=args.start,
        end=args.end,
        universe_codes=universe_codes,
    )

    if args.json_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    _print_summary(result, args.data_type, args.start, args.end)

    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
