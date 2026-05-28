"""手动真实数据 smoke 测试。

验证：Tushare 连通性 + 已落盘原始数据的完整性审计。

需要：
  - TUSHARE_TOKEN 环境变量，或 ~/.tushare/token 文件
  - data/raw/ 中已有部分数据（无数据时审计会汇报全缺失）

用法（手动触发，不入 CI）:
  pixi run smoke-data

退出码：
  0  全部检查通过或仅有警告
  1  审计状态为 error（critical 缺失）
  2  Tushare 连接失败
"""

from __future__ import annotations

import json
import os
import sys

_MIN_AUDIT_START = "20240101"
_MIN_AUDIT_END = "20240110"
_SMOKE_STOCKS = ["000001.SZ", "600519.SH", "000858.SZ"]


def _check_tushare() -> bool:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        token_file = os.path.expanduser("~/.tushare/token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()
    if not token:
        print("[FAIL] 未找到 TUSHARE_TOKEN，请设置环境变量或写入 ~/.tushare/token")
        return False

    try:
        import tushare as ts  # type: ignore[import]

        pro = ts.pro_api(token)
        df = pro.stock_basic(ts_code="000001.SZ", fields="ts_code,name")
        if df is None or len(df) == 0:
            print("[FAIL] Tushare 连接成功但返回空数据")
            return False
        print(f"[OK]   Tushare 连通，sample: {df.iloc[0].to_dict()}")
        return True
    except Exception as exc:
        print(f"[FAIL] Tushare 连接异常: {exc}")
        return False


def _audit_raw(data_type: str) -> dict:
    from common.data_audit import build_raw_data_audit

    result = build_raw_data_audit(
        data_type=data_type,
        start=_MIN_AUDIT_START,
        end=_MIN_AUDIT_END,
        universe_codes=_SMOKE_STOCKS,
    )
    return result


def main() -> None:
    print("=" * 60)
    print("FactorZen smoke-data: Tushare 连通性 + 原始数据审计")
    print("=" * 60)

    # 1. Tushare 连通性
    print("\n[1] Tushare 连通性检查")
    if not _check_tushare():
        sys.exit(2)

    # 2. 原始数据审计
    overall_ok = True
    for data_type in ("daily", "daily_basic"):
        print(f"\n[2] 审计 data/raw/{data_type}  [{_MIN_AUDIT_START} ~ {_MIN_AUDIT_END}]")
        try:
            result = _audit_raw(data_type)
        except Exception as exc:
            print(f"  [WARN] 审计异常（数据可能尚未拉取）: {exc}")
            continue

        status = result.get("status", "unknown")
        errors = result.get("errors", [])
        warnings = result.get("warnings", [])
        checks = result.get("checks", {})

        print(f"  status   : {status}")
        print(f"  checks   : {json.dumps(checks, ensure_ascii=False, indent=4)}")
        if warnings:
            for w in warnings:
                print(f"  [WARN]   {w}")
        if errors:
            for e in errors:
                print(f"  [ERROR]  {e}")
            overall_ok = False

    print("\n" + "=" * 60)
    if overall_ok:
        print("smoke-data PASSED")
    else:
        print("smoke-data FAILED — 见上方 [ERROR] 输出")
        sys.exit(1)


if __name__ == "__main__":
    main()
