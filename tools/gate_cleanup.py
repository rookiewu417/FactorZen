"""gate_cleanup：把重做后不再满足 library gate 的 active 记录降级为 gate_failed。

用法（从仓库根）:
    pixi run python tools/gate_cleanup.py           # dry-run（默认）：只打印名单
    pixi run python tools/gate_cleanup.py --apply   # 备份 + 写库 + 旁车 manifest

筛选条件（2026-07-28 重做裁决，与 .artifacts/rebuild90_report.md 对齐）:
    status == active 且 eval_start ∈ {20160104, 20170201}（已刷到新基线）且
    ( |ic_train| < 0.01  或  表达式在 OVERFIT_NAMED 点名清单 )

降级词 gate_failed：库消费方全部按 statuses=("active",) 白名单取数，
非 active 词自动退出组合/库池/lift 基线/forward 四条消费链；status 无枚举校验。
幂等：已是 gate_failed 的记录跳过。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

LIB = Path("workspace/factor_library/ashare.jsonl")
OPS_DIR = Path("workspace/_ops/library_cleanup")
NEW_WINDOW_STARTS = {"20160104", "20170201"}
IC_FLOOR = 0.01
# train/holdout 割裂的过拟合嫌疑（train 0.0279 / holdout -0.0005）
OVERFIT_NAMED = {
    "max(delay(volume_ratio, 60), zscore(zscore(neg(amplitude))))",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写库（默认 dry-run）")
    args = ap.parse_args()

    records = [json.loads(line) for line in LIB.read_text().splitlines() if line.strip()]
    targets = []
    for r in records:
        if r.get("status") != "active":
            continue
        if r.get("eval_start") not in NEW_WINDOW_STARTS:
            continue
        ic = r.get("ic_train")
        named = r.get("expression") in OVERFIT_NAMED
        if named or (ic is not None and abs(ic) < IC_FLOOR):
            targets.append(r)

    print(f"[gate-cleanup] 命中 {len(targets)} 条（active → gate_failed）:")
    for r in targets:
        print(f"  ic={r.get('ic_train'):+.4f} holdout={r.get('holdout_ic'):+.4f}  {r['expression'][:90]}")

    if not args.apply:
        print("[gate-cleanup] dry-run，未写库。加 --apply 落库。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = LIB.with_suffix(f".jsonl.bak.{stamp}_gate_cleanup")
    shutil.copy2(LIB, bak)

    hit = {r["expression"] for r in targets}
    out_lines = []
    for r in records:
        if r["expression"] in hit and r.get("status") == "active":
            r["status"] = "gate_failed"
            r["status_reason"] = (
                f"2026-07-28 重做裁决: 2016-2026 all_a 新窗口不满足 library gate"
                f"（|ic_train|<{IC_FLOOR} 或过拟合点名）"
            )
        out_lines.append(json.dumps(r, ensure_ascii=False))
    LIB.write_text("\n".join(out_lines) + "\n")

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "tool": "gate_cleanup",
        "applied_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "backup": str(bak),
        "criteria": {
            "ic_floor": IC_FLOOR,
            "new_window_starts": sorted(NEW_WINDOW_STARTS),
            "overfit_named": sorted(OVERFIT_NAMED),
        },
        "demoted": [
            {"expression": r["expression"], "ic_train": r.get("ic_train"),
             "holdout_ic": r.get("holdout_ic")} for r in targets
        ],
    }
    mpath = OPS_DIR / f"gate_cleanup_{stamp}.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[gate-cleanup] 已落库: 降级 {len(targets)} 条; 备份 {bak}; manifest {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
