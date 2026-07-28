"""``tools/naked_ic_cleanup.py``：存量库负裸 IC 清算 + admission_ic 回填。

判别力约定：期望值一律由本文件里**独立构造**的常量（合成库字面量 / 合成 CSV 字面量）
推出，绝不拿工具自身的输出当基准。未触及记录用**逐字节**比对锁死「最小 diff」。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "naked_ic_cleanup", _ROOT / "tools" / "naked_ic_cleanup.py"
)
assert _SPEC is not None and _SPEC.loader is not None
nic = importlib.util.module_from_spec(_SPEC)
# dataclasses 解析 `from __future__ import annotations` 下的注解要回查 sys.modules，
# 必须先登记再 exec，否则 @dataclass 直接 AttributeError。
sys.modules[_SPEC.name] = nic
_SPEC.loader.exec_module(nic)


# ── 合成夹具：5 条库 + 3 行 CSV（含一条 CSV 有、库里没有的幽灵行）────────────────

AUDIT_START = "2020-06-05"
AUDIT_END = "2026-06-05"

#  neg_a  active  · CSV 判负 → 应降级 + 回填
#  neg_b  active  · CSV 判负 → 应降级 + 回填
#  pos_c  active  · CSV 判正 → 只回填，status 不动
#  done_d naked_negative · CSV 判负、字段已全填 → 0 变更（幂等）
#  absent_e active · CSV 里没有 → 只告警、0 变更
_LIB_ROWS: list[dict] = [
    {
        "expression": "rank(close)", "market": "ashare", "ic_train": -0.031,
        "status": "active", "admission_track": "single",
        "admission_ic": None, "admission_start": None, "admission_end": None,
        "eval_start": "20200605", "eval_end": "20260605",
        "kind": "expression", "name": "neg_a", "updated_at": "2026-01-01",
    },
    {
        "expression": "ts_mean(vol, 20)", "market": "ashare", "ic_train": -0.009,
        "status": "active", "admission_track": "single",
        "admission_ic": None, "admission_start": None, "admission_end": None,
        "eval_start": "20200605", "eval_end": "20260605",
        "kind": "expression", "name": "neg_b", "updated_at": "2026-01-01",
    },
    {
        "expression": "neg(turnover_rate)", "market": "ashare", "ic_train": 0.018,
        "status": "active", "admission_track": "single",
        "admission_ic": None, "admission_start": None, "admission_end": None,
        "eval_start": "20200605", "eval_end": "20260605",
        "kind": "expression", "name": "pos_c", "updated_at": "2026-01-01",
    },
    {
        "expression": "delay(close, 5)", "market": "ashare", "ic_train": -0.022,
        "status": "naked_negative", "admission_track": "single",
        "admission_ic": -0.0215, "admission_start": AUDIT_START, "admission_end": AUDIT_END,
        "eval_start": "20200605", "eval_end": "20260605",
        "kind": "expression", "name": "done_d", "updated_at": "2026-07-28",
    },
    {
        # hypothesis 带中文：锁死 serialize 的 ensure_ascii=False（真库同款），
        # 否则「未变更即逐字节不变」的承诺在中文行上悄悄失效
        "expression": "abs(ret_1d)", "market": "ashare", "hypothesis": "低波异象·换手拥挤",
        "ic_train": 0.012,
        "status": "active", "admission_track": "single",
        "admission_ic": None, "admission_start": None, "admission_end": None,
        "eval_start": "20200605", "eval_end": "20260605",
        "kind": "expression", "name": "absent_e", "updated_at": "2026-01-01",
    },
]

_CSV = (
    "name,expression,ic_train_in_lib,holdout_ic_in_lib,rank_ic,ic_std,t,icir,"
    "n_days,negative,skip_reason\n"
    'neg_a,"rank(close)",-0.031,-0.03,-0.0300,0.12,-5.0,-0.25,1453,true,\n'
    'neg_b,"ts_mean(vol, 20)",-0.009,-0.008,-0.0100,0.11,-3.2,-0.09,1453,true,\n'
    'pos_c,"neg(turnover_rate)",0.018,0.017,0.0200,0.10,4.0,0.20,1453,false,\n'
    'done_d,"delay(close, 5)",-0.022,-0.02,-0.0215,0.13,-4.1,-0.16,1453,true,\n'
    'ghost_f,"log(amount)",-0.005,-0.004,-0.0050,0.09,-3.1,-0.05,1453,true,\n'
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    lib_root = tmp_path / "factor_library"
    lib_root.mkdir()
    lib_path = lib_root / "ashare.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _LIB_ROWS)
    lib_path.write_text(payload, encoding="utf-8")
    csv_path = tmp_path / "naked_ic_by_factor_full.csv"
    csv_path.write_text(_CSV, encoding="utf-8")
    return lib_root, csv_path, payload


def _plan(lib_root: Path, csv_path: Path):
    records = nic.load_raw_library(lib_root / "ashare.jsonl")
    rows = nic.load_audit_rows(csv_path)
    return records, nic.build_plan(
        records, rows,
        status_value=nic.DEFAULT_STATUS,
        admission_start=AUDIT_START,
        admission_end=AUDIT_END,
    )


def _by_name(records: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in records}


# ── 1. dry-run 变更计划必须逐条精确 ────────────────────────────────────────────

def test_plan_demotes_exactly_the_csv_negatives(tmp_path):
    lib_root, csv_path, _ = _write_fixture(tmp_path)
    _records, plan = _plan(lib_root, csv_path)

    # 只降 CSV 判负且当前 active 的两条；done_d 已是目标态不重复计入
    assert [d.name for d in plan.demotions] == ["neg_a", "neg_b"]
    assert [d.from_status for d in plan.demotions] == ["active", "active"]
    assert [d.to_status for d in plan.demotions] == ["naked_negative"] * 2
    # 裸 IC / t 来自 CSV 字面量，不是从库里抄的 ic_train（-0.031 / -0.009）
    assert [d.rank_ic for d in plan.demotions] == [-0.03, -0.01]
    assert [d.t for d in plan.demotions] == [-5.0, -3.2]
    assert plan.already_demoted == ["done_d"]


def test_plan_backfills_admission_fields_for_all_matched_nulls(tmp_path):
    lib_root, csv_path, _ = _write_fixture(tmp_path)
    _records, plan = _plan(lib_root, csv_path)

    ic_fills = {b.name: b.new for b in plan.backfills if b.field == "admission_ic"}
    # 负 2 条 + 正 1 条都回填；done_d 已有值不重填
    assert ic_fills == {"neg_a": -0.03, "neg_b": -0.01, "pos_c": 0.02}
    starts = {b.name for b in plan.backfills if b.field == "admission_start"}
    ends = {b.name for b in plan.backfills if b.field == "admission_end"}
    assert starts == ends == {"neg_a", "neg_b", "pos_c"}
    assert all(b.new == AUDIT_START for b in plan.backfills if b.field == "admission_start")
    assert all(b.new == AUDIT_END for b in plan.backfills if b.field == "admission_end")
    # absent_e 不在 CSV → 一个字段都不许被写
    assert "absent_e" not in {b.name for b in plan.backfills}


def test_plan_reports_both_directions_of_inconsistency(tmp_path):
    lib_root, csv_path, _ = _write_fixture(tmp_path)
    _records, plan = _plan(lib_root, csv_path)

    joined = "\n".join(plan.warnings)
    assert "ghost_f" in joined          # CSV 有、库里没有
    assert "absent_e" in joined         # 库里 active、CSV 缺
    assert plan.missing_in_library == ["ghost_f"]
    assert plan.missing_in_csv == ["absent_e"]


def test_plan_skips_record_when_naked_ic_sign_contradicts_ic_train(tmp_path):
    """R2 的前提是裸 IC 与 ic_train 120/120 同号；反号 = 前提失效 → 告警且一个字段都不动。"""
    lib_root, csv_path, _ = _write_fixture(tmp_path)
    lib_path = lib_root / "ashare.jsonl"
    rows = [dict(r) for r in _LIB_ROWS]
    rows[0]["ic_train"] = 0.031  # neg_a 的 ic_train 翻正，与 CSV 的 -0.03 反号
    lib_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )

    _records, plan = _plan(lib_root, csv_path)
    assert [d.name for d in plan.demotions] == ["neg_b"]
    assert "neg_a" not in {b.name for b in plan.backfills}
    assert any("neg_a" in w and "反号" in w for w in plan.warnings)


def test_plan_does_not_clobber_existing_admission_ic(tmp_path):
    """已有 admission_ic（lift 轨真值）与审计值不同 → 告警，绝不覆盖。"""
    lib_root, csv_path, _ = _write_fixture(tmp_path)
    lib_path = lib_root / "ashare.jsonl"
    rows = [dict(r) for r in _LIB_ROWS]
    rows[2]["admission_ic"] = 0.0777  # pos_c 已有值
    lib_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )

    _records, plan = _plan(lib_root, csv_path)
    assert "pos_c" not in {b.name for b in plan.backfills if b.field == "admission_ic"}
    assert any("pos_c" in w and "admission_ic" in w for w in plan.warnings)


# ── 2. apply：新 jsonl 逐字段正确 + bak + 幂等 ────────────────────────────────

def test_apply_writes_exact_fields_backup_and_is_idempotent(tmp_path):
    lib_root, csv_path, original = _write_fixture(tmp_path)
    lib_path = lib_root / "ashare.jsonl"

    rc = nic.main([
        "--library-root", str(lib_root), "--csv", str(csv_path),
        "--out-dir", str(tmp_path / "out"),
        "--admission-start", "20200605", "--admission-end", "20260605", "--apply", "--now", "2026-07-28",
    ])
    # 退出码 2 = 有一致性告警（本夹具刻意含 ghost_f / absent_e）；不是失败
    assert rc == 2

    after = _by_name(nic.load_raw_library(lib_path))
    assert len(after) == 5  # 一条都不许删

    assert after["neg_a"]["status"] == "naked_negative"
    assert after["neg_a"]["admission_ic"] == pytest.approx(-0.03)
    assert after["neg_a"]["admission_start"] == AUDIT_START
    assert after["neg_a"]["admission_end"] == AUDIT_END
    assert after["neg_a"]["updated_at"] == "2026-07-28"
    # 未涉及字段原样
    assert after["neg_a"]["ic_train"] == pytest.approx(-0.031)
    assert after["neg_a"]["expression"] == "rank(close)"
    assert after["neg_a"]["admission_track"] == "single"

    assert after["neg_b"]["status"] == "naked_negative"
    assert after["neg_b"]["admission_ic"] == pytest.approx(-0.01)

    # 正因子：只回填，status 保持 active
    assert after["pos_c"]["status"] == "active"
    assert after["pos_c"]["admission_ic"] == pytest.approx(0.02)
    assert after["pos_c"]["updated_at"] == "2026-07-28"

    # 已处理 / CSV 缺席：逐字节不变（最小 diff）
    orig_lines = {json.loads(ln)["name"]: ln for ln in original.splitlines()}
    new_lines = {
        json.loads(ln)["name"]: ln
        for ln in lib_path.read_text(encoding="utf-8").splitlines()
    }
    assert new_lines["done_d"] == orig_lines["done_d"]
    assert new_lines["absent_e"] == orig_lines["absent_e"]

    # 备份 = 原文件逐字节
    baks = sorted(lib_root.glob("ashare.jsonl.bak.*_naked_cleanup"))
    assert len(baks) == 1
    assert baks[0].read_text(encoding="utf-8") == original

    # 旁车 manifest
    man = json.loads((lib_root / "naked_ic_cleanup_manifest.json").read_text(encoding="utf-8"))
    assert man["market"] == "ashare"
    assert man["status_value"] == "naked_negative"
    assert [d["name"] for d in man["demoted"]] == ["neg_a", "neg_b"]
    assert man["n_backfilled_admission_ic"] == 3
    assert man["library_md5_before"] == hashlib.md5(original.encode("utf-8")).hexdigest()
    assert man["library_md5_after"] == hashlib.md5(
        lib_path.read_bytes()
    ).hexdigest()
    assert man["backup_path"] == str(baks[0])

    # ── 幂等：同一份库重跑 dry-run → 0 变更 ──
    _records2, plan2 = _plan(lib_root, csv_path)
    assert plan2.demotions == []
    assert plan2.backfills == []
    assert plan2.n_changed_records == 0


def test_dry_run_never_touches_the_library(tmp_path):
    lib_root, csv_path, original = _write_fixture(tmp_path)
    lib_path = lib_root / "ashare.jsonl"
    before = hashlib.md5(lib_path.read_bytes()).hexdigest()

    rc = nic.main([
        "--library-root", str(lib_root), "--csv", str(csv_path),
        "--out-dir", str(tmp_path / "out"),
        "--admission-start", "20200605", "--admission-end", "20260605", "--now", "2026-07-28",
    ])
    assert rc == 2  # 夹具含告警；dry-run 仍出计划
    assert hashlib.md5(lib_path.read_bytes()).hexdigest() == before
    assert lib_path.read_text(encoding="utf-8") == original
    assert not list(lib_root.glob("*.bak.*"))
    assert not (lib_root / "naked_ic_cleanup_manifest.json").exists()

    # 旁路产物写到 out-dir，且内容 = apply 后会写进库的那份
    proposed = tmp_path / "out" / "ashare.jsonl.proposed"
    assert proposed.exists()
    got = _by_name(nic.load_raw_library(proposed))
    assert got["neg_a"]["status"] == "naked_negative"
    assert got["pos_c"]["status"] == "active"
    assert got["pos_c"]["admission_ic"] == pytest.approx(0.02)


def test_serialize_roundtrips_bytes_for_untouched_library(tmp_path):
    """序列化口径必须与 ``_save_library`` 一致，否则「未变更即不变」的承诺不成立。"""
    _lib_root, _csv_path, original = _write_fixture(tmp_path)
    records = [json.loads(ln) for ln in original.splitlines()]
    assert nic.serialize(records) == original


# ── 3. 消费方容忍性：naked_negative 不进 active 集，且改回 status 即恢复 ──────

def test_naked_negative_is_excluded_from_active_consumers_and_reversible(tmp_path):
    from factorzen.discovery.factor_library import (
        _save_library,
        library_covered_by_family,
        library_covered_expressions,
        load_library,
    )

    lib_root, csv_path, _ = _write_fixture(tmp_path)
    nic.main([
        "--library-root", str(lib_root), "--csv", str(csv_path),
        "--out-dir", str(tmp_path / "out"),
        "--admission-start", "20200605", "--admission-end", "20260605", "--apply", "--now", "2026-07-28",
    ])
    root = str(lib_root)

    covered = library_covered_expressions("ashare", k=50, root=root)
    assert "rank(close)" not in covered          # neg_a 被剔
    assert "ts_mean(vol, 20)" not in covered     # neg_b 被剔
    assert "neg(turnover_rate)" in covered       # pos_c 仍在
    assert "abs(ret_1d)" in covered              # absent_e 未被动过

    fam, _crowded = library_covered_by_family("ashare", max_total=50, root=root)
    assert "rank(close)" not in fam
    assert "neg(turnover_rate)" in fam

    # 记录仍在库里可审计（不是被删）
    recs = load_library("ashare", root=root)
    assert len(recs) == 5
    assert {r.name for r in recs if r.status == "naked_negative"} == {
        "neg_a", "neg_b", "done_d"
    }
    # 未知 status 不丢字段：admission_ic 落盘后能读回
    assert next(r for r in recs if r.name == "neg_a").admission_ic == pytest.approx(-0.03)

    # 可逆：把 status 改回 active 即恢复消费
    for r in recs:
        if r.name == "neg_a":
            r.status = "active"
    _save_library("ashare", recs, root=root)
    assert "rank(close)" in library_covered_expressions("ashare", k=50, root=root)
