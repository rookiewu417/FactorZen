"""真实数据 smoke：验证 Tushare 连通性 + 本地原始数据分区审计。

与默认 CI 分离（CI 保持离线可重复），作为手动命令运行：

    pixi run smoke-data --start 20230101 --end 20231231
    pixi run smoke-data --skip-tushare            # 仅离线审计本地分区
    pixi run smoke-data --data-type daily --json   # 单类型 + JSON 输出

退出码：0=全部 ok；1=出现 error；2=仅 warning（无 error）。
连通性检查需要 TUSHARE_TOKEN；审计只读本地 data/raw/，可离线运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from factorzen.core.data_audit import build_raw_data_audit

_DEFAULT_TYPES = ("daily", "daily_basic", "finance")


def check_tushare_connectivity() -> tuple[bool, str]:
    """用一次轻量真实调用（trade_cal 5 天窗口）验证 Tushare 可达。

    Returns:
        (ok, message)。token 缺失或网络/权限失败时 ok=False。
    """
    try:
        from factorzen.core.loader import _retry, init_tushare

        pro = init_tushare()
        df = _retry(pro.trade_cal, exchange="SSE", start_date="20240102", end_date="20240108")
    except Exception as exc:
        return False, f"Tushare 连通性失败: {exc}"

    if df is None or getattr(df, "empty", False):
        return False, "Tushare trade_cal 返回空，疑似 token/权限问题"
    return True, f"Tushare 连通正常（trade_cal 返回 {len(df)} 行）"


def run_audits(
    data_types: tuple[str, ...] | list[str],
    start: str,
    end: str,
    universe_codes: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """对每个 data_type 运行本地分区审计，返回 {data_type: audit_result}。"""
    results: dict[str, dict[str, Any]] = {}
    for dt in data_types:
        try:
            results[dt] = build_raw_data_audit(
                data_type=dt, start=start, end=end, universe_codes=universe_codes
            )
        except Exception as exc:
            # 单类型审计异常（如本地无日历缓存且离线）不应中断整个 smoke
            results[dt] = {
                "status": "error",
                "checks": {},
                "warnings": [],
                "errors": [f"审计异常: {exc}"],
            }
    return results


def _worst_status(statuses: list[str]) -> str:
    """error > warning > ok 的优先级取最差。"""
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def summarize(
    connectivity: tuple[bool, str] | None,
    audits: dict[str, dict[str, Any]],
) -> int:
    """打印汇总并返回退出码（0 ok / 1 error / 2 warning）。"""
    print("=" * 60)
    print("FactorZen 数据 smoke")
    print("=" * 60)

    statuses: list[str] = []

    if connectivity is not None:
        ok, msg = connectivity
        print(f"[连通性] {'OK ' if ok else 'FAIL'} — {msg}")
        if not ok:
            statuses.append("error")
    else:
        print("[连通性] 跳过（--skip-tushare）")

    for dt, res in audits.items():
        status = res.get("status", "error")
        statuses.append(status)
        tag = {"ok": "OK  ", "warning": "WARN", "error": "FAIL"}.get(status, "????")
        print(f"[审计:{dt}] {tag} — {res.get('checks', {}).get('total_rows', '?')} 行")
        for w in res.get("warnings", []):
            print(f"         · warning: {w}")
        for e in res.get("errors", []):
            print(f"         · error:   {e}")

    worst = _worst_status(statuses)
    print("=" * 60)
    print(f"总体: {worst.upper()}")
    return {"ok": 0, "error": 1, "warning": 2}[worst]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FactorZen 真实数据 smoke")
    p.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    p.add_argument("--end", default="20231231", help="截止日期 YYYYMMDD")
    p.add_argument(
        "--data-type",
        choices=_DEFAULT_TYPES,
        action="append",
        help="审计的数据类型，可重复；默认审计全部三类",
    )
    p.add_argument(
        "--skip-tushare", action="store_true", help="跳过连通性检查，仅离线审计本地分区"
    )
    p.add_argument("--json", action="store_true", help="同时输出机器可读 JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_types = tuple(args.data_type) if args.data_type else _DEFAULT_TYPES

    connectivity = None if args.skip_tushare else check_tushare_connectivity()
    audits = run_audits(data_types, args.start, args.end)

    code = summarize(connectivity, audits)

    if args.json:
        payload = {
            "connectivity": None
            if connectivity is None
            else {"ok": connectivity[0], "message": connectivity[1]},
            "audits": audits,
            "exit_code": code,
        }
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))

    return code


if __name__ == "__main__":
    sys.exit(main())
