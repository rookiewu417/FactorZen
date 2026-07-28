"""存量因子库清算：剔除负裸 IC 记录 + 回填 ``admission_ic``（默认 dry-run）。

数据源是一次「统一窗口逐因子裸 RankIC」审计产出的 CSV
（列 ``name,expression,ic_train_in_lib,holdout_ic_in_lib,rank_ic,ic_std,t,icir,
n_days,negative,skip_reason``）。工具做两件事，且只做这两件：

1. **剔负**：CSV 判 ``negative=true`` 且库内当前 ``status="active"`` 的记录 →
   ``status="naked_negative"``。记录**保留在库里**（可审计），但退出 active 的一切
   消费（等权部署 ``pipelines/factor_combine.py``、挖掘正交参照 / 残差 objective /
   lift baseline ``discovery/factor_library.py`` + ``discovery/lift_test.py``、
   去相关池、store 物化、forward-track）。**可逆**：status 改回 ``active`` 即恢复。
   与 ``correlated``（收录但不参与）同型；刻意**不用 ``probation``**——那是
   forward-track 观察语义，会污染 forward 流程。
2. **回填**：把审计窗口的裸 rank_ic 写进 ``admission_ic``（原为 null 才写），同时补
   ``admission_start`` / ``admission_end``（ISO ``YYYY-MM-DD``，对齐
   ``cli/main.py::_lift_admission_str``）。

已知边界（务必与结论一起转述）：

- ``admission_ic`` 回填**不会**让「裸 IC 同号门」对 single 轨存量生效——single 轨结构上
  不经过 ``lift_admission``（见 ``factor_library.rebuild`` docstring）。回填的价值是
  统一窗口裸 IC 的证据落库 + 可审计。唯一行为面影响是 ``forward_track`` 的方向解析
  从 ``ic_train`` 改用 ``admission_ic``；工具对每条做同号校验，反号即告警并整条跳过。
- ``status="naked_negative"`` **扛不住全量 ``fz factor-library rebuild``**
  （``fresh=True`` 清库 → 重评 → ``_decorrelate`` 写回 ``active``）。定向
  ``rebuild --only``（``preserve_status=True``）则完整存活。跑过全量 rebuild 之后
  必须重跑本工具（幂等，重复跑 0 变更）。
- ``fz combine from-library --statuses naked_negative`` 会被
  ``cli/parser.py`` 的白名单拒（argparse exit 2）——这是期望行为。

用法::

    # dry-run（默认）：只打印变更计划 + 写旁路 <out-dir>/<market>.jsonl.proposed
    pixi run -- python tools/naked_ic_cleanup.py \
        --csv .artifacts/r2_naked_ic_audit/naked_ic_by_factor_full.csv

    # 落库：先备份 <market>.jsonl.bak.<yyyymmdd>_naked_cleanup 再原子覆盖 + 写旁车 manifest
    pixi run -- python tools/naked_ic_cleanup.py --csv <...> --apply

退出码：0=计划生成/落库成功；1=输入错误（库或 CSV 缺失/畸形）；2=有一致性告警
（计划仍然生成，但需人工过目后再 ``--apply``）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from factorzen.config.settings import FACTOR_LIBRARY_DIR
from factorzen.core.experiment import get_git_sha

# 剔负后的目标 status。与 ``correlated`` 同型：收录但不参与 active 消费。
DEFAULT_STATUS = "naked_negative"
MANIFEST_NAME = "naked_ic_cleanup_manifest.json"
_BACKUP_SUFFIX = "_naked_cleanup"
# 回填 admission_ic 的浮点等值容差（幂等判定：已是同一个值就不算变更）。
_EPS = 1e-12


# ── 审计行 ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditRow:
    """审计 CSV 的一行（只保留裁决与回填要用的字段）。"""

    name: str
    expression: str
    rank_ic: float
    t: float | None
    negative: bool
    skip_reason: str


@dataclass(frozen=True)
class Demotion:
    name: str
    expression: str
    rank_ic: float
    t: float | None
    from_status: str
    to_status: str


@dataclass(frozen=True)
class Backfill:
    name: str
    field: str
    old: Any
    new: Any


@dataclass
class Plan:
    """一次清算的完整变更计划。``edits`` 是唯一的落盘真源，其余列表只为报告。"""

    # 记录下标 → {字段: 新值}
    edits: dict[int, dict[str, Any]] = field(default_factory=dict)
    demotions: list[Demotion] = field(default_factory=list)
    backfills: list[Backfill] = field(default_factory=list)
    already_demoted: list[str] = field(default_factory=list)
    missing_in_library: list[str] = field(default_factory=list)
    missing_in_csv: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_records: int = 0

    @property
    def n_changed_records(self) -> int:
        return len(self.edits)

    @property
    def n_changed_fields(self) -> int:
        return sum(len(v) for v in self.edits.values())


# ── IO ──────────────────────────────────────────────────────────────────────

def load_raw_library(path: str | Path) -> list[dict]:
    """按行读 jsonl 为**原始 dict**（不过 ``FactorRecord``）。

    刻意不走 ``load_library`` + ``_save_library``：dataclass 往返会丢弃 jsonl 里的未知
    键、并按 dataclass 字段顺序重排，未变更记录就不再逐字节稳定。原始 dict 最小改字段
    才能保证 diff 只含真实变更（真库已验证 ``json.dumps(d, ensure_ascii=False)+"\\n"``
    重写后 md5 与原文件一致）。
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"因子库不存在: {p}")
    out: list[dict] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{i} 不是合法 JSON: {exc}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"{p}:{i} 不是 JSON 对象")
        out.append(rec)
    return out


def serialize(records: list[dict]) -> str:
    """与 ``factor_library._save_library`` 同口径的 jsonl 文本。"""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def _as_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = v.strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def load_audit_rows(path: str | Path) -> list[AuditRow]:
    """读审计 CSV。缺列 / 畸形行只抛 ``ValueError``（异常契约）。"""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"审计 CSV 不存在: {p}")
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        need = {"name", "expression", "rank_ic", "negative"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"审计 CSV 缺列 {sorted(missing)}: {p}")
        rows: list[AuditRow] = []
        for raw in reader:
            name = (raw.get("name") or "").strip()
            if not name:
                continue
            ic = _as_float(raw.get("rank_ic"))
            rows.append(
                AuditRow(
                    name=name,
                    expression=(raw.get("expression") or "").strip(),
                    rank_ic=ic if ic is not None else float("nan"),
                    t=_as_float(raw.get("t")),
                    negative=(raw.get("negative") or "").strip().lower() == "true",
                    skip_reason=(raw.get("skip_reason") or "").strip(),
                )
            )
    return rows


def file_md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ── 计划 ─────────────────────────────────────────────────────────────────────

def _squash(expr: str) -> str:
    """表达式身份比对：只去空白（CSV 与库的括号内空格排版可能不同）。"""
    return "".join(expr.split())


def _same_sign(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def build_plan(
    records: list[dict],
    rows: list[AuditRow],
    *,
    status_value: str = DEFAULT_STATUS,
    admission_start: str | None = None,
    admission_end: str | None = None,
) -> Plan:
    """比对库与审计结果，产出变更计划（纯函数，不写盘）。

    逐条规则：

    - CSV 行 ``skip_reason`` 非空或 ``rank_ic`` 不可用 → 告警跳过。
    - 名字在库里 0 命中 / >1 命中 / 表达式对不上 → 告警跳过（身份存疑绝不动）。
    - 裸 IC 与库内 ``ic_train`` 反号 → 告警跳过整条（审计前提失效）。
    - ``negative=true`` 且当前 ``active`` → 降级；已是目标态 → 计入 ``already_demoted``；
      其它 status（correlated / probation / no_lift）→ 告警不降级，但仍允许回填。
    - ``admission_ic`` 为 null 才回填；已有值且与审计值不等 → 告警不覆盖。
    - ``admission_start`` / ``admission_end`` 同理，仅在为 null 时填。
    - 库内 ``active`` / 目标态但 CSV 缺 → 告警（库自审计以来变过）。
    """
    plan = Plan(n_records=len(records))
    by_name: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        nm = r.get("name")
        if isinstance(nm, str) and nm:
            by_name.setdefault(nm, []).append(i)

    def _edit(idx: int, key: str, value: Any) -> None:
        plan.edits.setdefault(idx, {})[key] = value

    seen_csv: set[str] = set()
    for row in rows:
        seen_csv.add(row.name)
        if row.skip_reason:
            plan.warnings.append(f"[csv] {row.name}: skip_reason={row.skip_reason!r} → 跳过")
            continue
        if not math.isfinite(row.rank_ic):
            plan.warnings.append(f"[csv] {row.name}: rank_ic 缺失/非有限 → 跳过")
            continue

        idxs = by_name.get(row.name, [])
        if not idxs:
            plan.missing_in_library.append(row.name)
            plan.warnings.append(f"[缺席] CSV 有 {row.name} 但库内无同名记录 → 跳过")
            continue
        if len(idxs) > 1:
            plan.warnings.append(f"[重名] 库内 {row.name} 命中 {len(idxs)} 条 → 跳过")
            continue

        idx = idxs[0]
        rec = records[idx]
        lib_expr = str(rec.get("expression") or "")
        if row.expression and _squash(lib_expr) != _squash(row.expression):
            plan.warnings.append(
                f"[身份] {row.name} 表达式不一致 csv={row.expression!r} lib={lib_expr!r} → 跳过"
            )
            continue

        ic_train = rec.get("ic_train")
        if (
            isinstance(ic_train, (int, float))
            and math.isfinite(float(ic_train))
            and not _same_sign(row.rank_ic, float(ic_train))
        ):
            plan.warnings.append(
                f"[反号] {row.name} 裸 IC={row.rank_ic:+.6f} 与库内 "
                f"ic_train={float(ic_train):+.6f} 反号（审计前提失效）→ 整条跳过"
            )
            continue

        cur_status = str(rec.get("status") or "")
        # ── status ──
        if row.negative:
            if cur_status == status_value:
                plan.already_demoted.append(row.name)
            elif cur_status == "active":
                plan.demotions.append(
                    Demotion(
                        name=row.name, expression=lib_expr, rank_ic=row.rank_ic, t=row.t,
                        from_status=cur_status, to_status=status_value,
                    )
                )
                _edit(idx, "status", status_value)
            else:
                plan.warnings.append(
                    f"[状态] {row.name} CSV 判负但库内 status={cur_status!r}（非 active）→ 不改 status"
                )
        elif cur_status == status_value:
            plan.warnings.append(
                f"[状态] {row.name} 库内已是 {status_value!r} 但 CSV 判正 → 不自动恢复，需人工确认"
            )

        # ── admission_ic / admission_start / admission_end ──
        old_ic = rec.get("admission_ic")
        if old_ic is None:
            plan.backfills.append(Backfill(row.name, "admission_ic", None, row.rank_ic))
            _edit(idx, "admission_ic", row.rank_ic)
        elif not (
            isinstance(old_ic, (int, float))
            and math.isfinite(float(old_ic))
            and abs(float(old_ic) - row.rank_ic) <= _EPS
        ):
            plan.warnings.append(
                f"[回填] {row.name} admission_ic 已有值 {old_ic!r} ≠ 审计值 {row.rank_ic:+.6f} → 不覆盖"
            )

        for key, val in (("admission_start", admission_start), ("admission_end", admission_end)):
            if val is None:
                continue
            old = rec.get(key)
            if old is None:
                plan.backfills.append(Backfill(row.name, key, None, val))
                _edit(idx, key, val)
            elif old != val:
                plan.warnings.append(
                    f"[回填] {row.name} {key} 已有值 {old!r} ≠ 审计窗 {val!r} → 不覆盖"
                )

    # 库内在册但 CSV 缺
    for i, rec in enumerate(records):
        if str(rec.get("status") or "") not in ("active", status_value):
            continue
        nm = rec.get("name")
        if isinstance(nm, str) and nm and nm not in seen_csv:
            plan.missing_in_csv.append(nm)
            plan.warnings.append(
                f"[缺席] 库内 #{i} {nm} status={rec.get('status')!r} 但 CSV 无此因子 → 未处理"
            )
    return plan


def apply_plan(records: list[dict], plan: Plan, *, now: str) -> list[dict]:
    """把计划套到记录副本上，返回新列表（原列表不变）。改动记录同时刷新 ``updated_at``。"""
    out = [dict(r) for r in records]
    for idx, changes in plan.edits.items():
        rec = out[idx]
        rec.update(changes)
        if "updated_at" in rec:
            rec["updated_at"] = now
    return out


# ── 报告 ─────────────────────────────────────────────────────────────────────

def _fmt(v: float | None, nd: int = 5) -> str:
    return "-" if v is None or not math.isfinite(v) else f"{v:+.{nd}f}"


def render_plan(plan: Plan, *, market: str, library_path: Path, status_value: str,
                admission_start: str | None, admission_end: str | None) -> str:
    lines: list[str] = []
    lines.append(f"# 因子库负裸 IC 清算计划 · {market}")
    lines.append("")
    lines.append(f"库文件      : {library_path}")
    lines.append(f"库内记录数  : {plan.n_records}")
    lines.append(f"目标 status : {status_value}")
    lines.append(f"审计窗      : {admission_start or '-'} → {admission_end or '-'}")
    lines.append("")
    lines.append(f"## 1. status 变更（active → {status_value}）：{len(plan.demotions)} 条")
    lines.append("")
    if plan.demotions:
        lines.append("| # | name | 裸 rank_ic | t | expression |")
        lines.append("|---|---|---|---|---|")
        for i, d in enumerate(sorted(plan.demotions, key=lambda x: x.rank_ic), 1):
            lines.append(
                f"| {i} | {d.name} | {_fmt(d.rank_ic)} | {_fmt(d.t, 2)} | `{d.expression}` |"
            )
    else:
        lines.append("_（无）_")
    if plan.already_demoted:
        lines.append("")
        lines.append(
            f"已处于 {status_value} 的记录（本次不重复变更）：{len(plan.already_demoted)} 条 "
            f"— {', '.join(plan.already_demoted)}"
        )

    n_ic = sum(1 for b in plan.backfills if b.field == "admission_ic")
    n_st = sum(1 for b in plan.backfills if b.field == "admission_start")
    n_en = sum(1 for b in plan.backfills if b.field == "admission_end")
    lines.append("")
    lines.append("## 2. admission 字段回填")
    lines.append("")
    lines.append(f"- admission_ic   : {n_ic} 条")
    lines.append(f"- admission_start: {n_st} 条")
    lines.append(f"- admission_end  : {n_en} 条")

    lines.append("")
    lines.append(f"## 3. 一致性告警：{len(plan.warnings)} 条")
    lines.append("")
    if plan.warnings:
        lines.extend(f"- {w}" for w in plan.warnings)
    else:
        lines.append("_（无）_")

    lines.append("")
    lines.append("## 4. diff 摘要")
    lines.append("")
    lines.append(f"- 变更记录数: {plan.n_changed_records} / {plan.n_records}")
    lines.append(f"- 变更字段数: {plan.n_changed_fields}")
    lines.append("- 未变更记录逐字节不变（原始 dict 最小改字段）")
    return "\n".join(lines) + "\n"


# ── 落盘 ─────────────────────────────────────────────────────────────────────

def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _build_manifest(
    *, market: str, library_path: Path, csv_path: Path, plan: Plan, status_value: str,
    admission_start: str | None, admission_end: str | None, now: str,
    md5_before: str, md5_after: str, backup_path: Path, extra_sources: dict[str, str],
) -> dict:
    return {
        "tool": "tools/naked_ic_cleanup.py",
        "market": market,
        "applied_at": now,
        "git_sha": get_git_sha(),
        "status_value": status_value,
        "library_path": str(library_path),
        "library_md5_before": md5_before,
        "library_md5_after": md5_after,
        "backup_path": str(backup_path),
        "n_records": plan.n_records,
        "audit_csv": str(csv_path),
        "audit_csv_md5": file_md5(csv_path),
        "audit_sources": extra_sources,
        "admission_start": admission_start,
        "admission_end": admission_end,
        "n_demoted": len(plan.demotions),
        "demoted": [
            {
                "name": d.name, "expression": d.expression,
                "rank_ic": d.rank_ic, "t": d.t, "from_status": d.from_status,
            }
            for d in plan.demotions
        ],
        "already_demoted": plan.already_demoted,
        "n_backfilled_admission_ic": sum(
            1 for b in plan.backfills if b.field == "admission_ic"
        ),
        "n_backfilled_admission_start": sum(
            1 for b in plan.backfills if b.field == "admission_start"
        ),
        "n_backfilled_admission_end": sum(
            1 for b in plan.backfills if b.field == "admission_end"
        ),
        "backfilled_admission_ic": [
            {"name": b.name, "admission_ic": b.new}
            for b in plan.backfills if b.field == "admission_ic"
        ],
        "n_changed_records": plan.n_changed_records,
        "n_changed_fields": plan.n_changed_fields,
        "warnings": plan.warnings,
        "missing_in_library": plan.missing_in_library,
        "missing_in_csv": plan.missing_in_csv,
        "notes": [
            f"status={status_value} 让记录退出 active 的一切消费但保留在库里；"
            f"改回 status='active' 即完全恢复（本 manifest 的 demoted 列表就是回滚清单）。",
            "全量 `fz factor-library rebuild`(fresh=True) 会清库重评并把本标记冲回 active；"
            "跑过全量 rebuild 后必须重跑本工具（幂等）。定向 `rebuild --only` 不受影响。",
            "admission_ic 为统一窗口裸 RankIC，是证据落库；single 轨结构上不经过 "
            "lift_admission，回填不会激活裸 IC 同号门。",
        ],
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def _resolve_window(
    args: argparse.Namespace, csv_path: Path,
) -> tuple[str | None, str | None]:
    """审计窗：显式旗标优先；否则读 CSV 同目录的 ``result_full.json`` 的 config.start/end。"""
    if args.no_admission_window:
        return None, None
    start, end = args.admission_start, args.admission_end
    if start and end:
        return _iso(start), _iso(end)
    result_json = Path(args.result_json) if args.result_json else csv_path.with_name(
        "result_full.json"
    )
    if not result_json.exists():
        raise ValueError(
            f"未给 --admission-start/--admission-end 且找不到 {result_json}；"
            f"要跳过窗口回填请显式传 --no-admission-window"
        )
    try:
        cfg = json.loads(result_json.read_text(encoding="utf-8")).get("config") or {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{result_json} 不是合法 JSON: {exc}") from exc
    start = start or cfg.get("start")
    end = end or cfg.get("end")
    if not start or not end:
        raise ValueError(f"{result_json} 的 config 缺 start/end")
    return _iso(str(start)), _iso(str(end))


def _iso(s: str) -> str:
    """``YYYYMMDD`` → ``YYYY-MM-DD``（对齐 ``cli/main.py::_lift_admission_str``）。"""
    s = s.strip().replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="因子库负裸 IC 清算 + admission_ic 回填（默认 dry-run）")
    p.add_argument("--market", default="ashare", choices=["ashare", "crypto", "futures", "us"])
    p.add_argument("--library-root", dest="library_root", default=str(FACTOR_LIBRARY_DIR),
                   help="因子库目录（默认 workspace/factor_library）")
    p.add_argument("--csv", required=True, help="审计 CSV（naked_ic_by_factor_full.csv）")
    p.add_argument("--result-json", dest="result_json", default=None,
                   help="审计 result_full.json（默认取 CSV 同目录同名文件），用于取审计窗")
    p.add_argument("--out-dir", dest="out_dir", default=None,
                   help="dry-run 旁路产物目录（默认 <library-root>/_naked_ic_cleanup）")
    p.add_argument("--status", dest="status_value", default=DEFAULT_STATUS,
                   help=f"剔负后的目标 status（默认 {DEFAULT_STATUS}）")
    p.add_argument("--admission-start", dest="admission_start", default=None)
    p.add_argument("--admission-end", dest="admission_end", default=None)
    p.add_argument("--no-admission-window", dest="no_admission_window", action="store_true",
                   help="只回填 admission_ic，不碰 admission_start/end")
    p.add_argument("--now", default=None, help="写入 updated_at / manifest 的日期（默认今天）")
    p.add_argument("--apply", action="store_true",
                   help="真正写库（先备份再原子覆盖 + 写旁车 manifest）；缺省为 dry-run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    now = args.now or date.today().isoformat()
    lib_root = Path(args.library_root)
    lib_path = lib_root / f"{args.market}.jsonl"
    csv_path = Path(args.csv)

    try:
        records = load_raw_library(lib_path)
        rows = load_audit_rows(csv_path)
        adm_start, adm_end = _resolve_window(args, csv_path)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    md5_before = file_md5(lib_path)
    plan = build_plan(
        records, rows, status_value=args.status_value,
        admission_start=adm_start, admission_end=adm_end,
    )
    report = render_plan(
        plan, market=args.market, library_path=lib_path, status_value=args.status_value,
        admission_start=adm_start, admission_end=adm_end,
    )
    print(report)

    new_records = apply_plan(records, plan, now=now)
    payload = serialize(new_records)

    out_dir = Path(args.out_dir) if args.out_dir else lib_root / "_naked_ic_cleanup"
    if not args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)
        proposed = out_dir / f"{args.market}.jsonl.proposed"
        proposed.write_text(payload, encoding="utf-8")
        (out_dir / f"{args.market}_naked_ic_cleanup_plan.md").write_text(report, encoding="utf-8")
        print(f"[dry-run] 未写库。库 md5 {md5_before}（前后一致：{file_md5(lib_path)}）")
        print(f"[dry-run] 提案库 → {proposed}（md5 {hashlib.md5(payload.encode()).hexdigest()}）")
        print(f"[dry-run] 计划全文 → {out_dir / f'{args.market}_naked_ic_cleanup_plan.md'}")
        print("[dry-run] 落库请加 --apply（不可逆的库变更，请先人工过目上面的清单）")
        return 2 if plan.warnings else 0

    if not plan.edits:
        print("无变更，跳过写库（幂等）。")
        return 2 if plan.warnings else 0

    stamp = now.replace("-", "")
    backup = lib_path.with_name(f"{lib_path.name}.bak.{stamp}{_BACKUP_SUFFIX}")
    shutil.copy2(lib_path, backup)
    _atomic_write(lib_path, payload)
    md5_after = file_md5(lib_path)

    extra: dict[str, str] = {}
    for extra_name in ("result_full.json", "REPORT.md"):
        cand = csv_path.with_name(extra_name)
        if cand.exists():
            extra[str(cand)] = file_md5(cand)
    manifest = _build_manifest(
        market=args.market, library_path=lib_path, csv_path=csv_path, plan=plan,
        status_value=args.status_value, admission_start=adm_start, admission_end=adm_end,
        now=now, md5_before=md5_before, md5_after=md5_after, backup_path=backup,
        extra_sources=extra,
    )
    _atomic_write(
        lib_root / MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"[apply] 备份 → {backup}")
    print(f"[apply] 写库 → {lib_path}（md5 {md5_before} → {md5_after}）")
    print(f"[apply] 旁车 manifest → {lib_root / MANIFEST_NAME}")

    # 人类汇总同步（与 factor_library 其它写库路径一致：_save_library 后刷 md）。
    # 注意 render_markdown 的头部只统计 active/correlated/probation，naked_negative
    # 不进三个计数（`no_lift` 今天同样如此），表格行照常渲染。渲染失败不影响 jsonl 真源。
    try:
        from factorzen.discovery.factor_library import render_markdown

        render_markdown(args.market, root=str(lib_root))
        print(f"[apply] 刷新 {args.market}.md / summary.md（头部计数不含 {args.status_value}）")
    except Exception as exc:  # md 是派生产物，绝不因它失败回退 jsonl
        print(f"[apply] 警告: 刷新 md 失败（jsonl 已写成功）: {exc}", file=sys.stderr)
    return 2 if plan.warnings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
