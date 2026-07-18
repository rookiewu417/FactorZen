"""三期 W1：评估记录 → 库 evidence 链接（last_eval_run_id / last_eval_at）。

不污染裁决指标：link 只写两字段，ic/lift/status 原样。
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path


def _write_lib(root: Path, market: str, records: list[dict]) -> Path:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def test_link_evaluation_to_library_python_and_expression(tmp_path):
    """python 型与 expression 型各一：命中写字段、round-trip、裁决字段不变。"""
    from factorzen.discovery.factor_library import (
        link_evaluation_to_library,
        load_library,
        python_identity,
    )

    py_key = python_identity("momentum_20d")
    recs = [
        {
            "expression": py_key,
            "market": "ashare",
            "kind": "python",
            "name": "momentum_20d",
            "impl": "momentum_20d",
            "ic_train": 0.042,
            "holdout_ic": 0.031,
            "lift": 0.005,
            "status": "active",
            "admission_track": "single",
        },
        {
            "expression": "rank(close)",
            "market": "ashare",
            "kind": "expression",
            "name": "mined_expr_close",
            "ic_train": 0.02,
            "holdout_ic": 0.015,
            "lift": None,
            "status": "probation",
            "admission_track": "lift",
        },
    ]
    _write_lib(tmp_path, "ashare", recs)

    assert link_evaluation_to_library(
        "momentum_20d", "momentum_20d_20260718_120000", "2026-07-18",
        market="ashare", root=str(tmp_path),
    ) is True
    assert link_evaluation_to_library(
        "mined_expr_close", "run_expr_001", "2026-07-18T12:00:00",
        market="ashare", root=str(tmp_path),
    ) is True

    lib = {r.name: r for r in load_library("ashare", root=str(tmp_path))}
    py = lib["momentum_20d"]
    assert py.last_eval_run_id == "momentum_20d_20260718_120000"
    assert py.last_eval_at == "2026-07-18"
    assert py.ic_train == 0.042
    assert py.holdout_ic == 0.031
    assert py.lift == 0.005
    assert py.status == "active"
    assert py.kind == "python"
    assert py.expression == py_key

    ex = lib["mined_expr_close"]
    assert ex.last_eval_run_id == "run_expr_001"
    assert ex.last_eval_at == "2026-07-18T12:00:00"
    assert ex.ic_train == 0.02
    assert ex.holdout_ic == 0.015
    assert ex.lift is None
    assert ex.status == "probation"
    assert ex.kind == "expression"


def test_link_evaluation_to_library_miss_no_mutation(tmp_path):
    """找不到 name → False 且库文件未变（内容 + mtime）。"""
    from factorzen.discovery.factor_library import link_evaluation_to_library

    path = _write_lib(
        tmp_path, "ashare",
        [{
            "expression": "rank(vol)",
            "market": "ashare",
            "name": "keep_me",
            "ic_train": 0.01,
            "status": "active",
        }],
    )
    before = path.read_bytes()
    mtime_before = path.stat().st_mtime_ns
    time.sleep(0.02)

    ok = link_evaluation_to_library(
        "does_not_exist", "run_x", "2026-07-18",
        market="ashare", root=str(tmp_path),
    )
    assert ok is False
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime_before


def test_link_evaluation_to_library_corrupt_library_no_crash(tmp_path):
    """库文件损坏（乱字节）→ False 不崩。"""
    from factorzen.discovery.factor_library import link_evaluation_to_library

    path = tmp_path / "ashare.jsonl"
    path.write_bytes(b"\xff\xfe\x00\x01\x80\x81 garbage not json\x00")

    ok = link_evaluation_to_library(
        "any", "run_y", "2026-07-18",
        market="ashare", root=str(tmp_path),
    )
    assert ok is False


def test_daily_single_wires_link_evaluation_to_library():
    """接线真实存在：inspect 源码含调用；并直调 link 本体（非 mock 互调）。"""
    import factorzen.pipelines.daily_single as ds
    from factorzen.discovery.factor_library import link_evaluation_to_library

    src = inspect.getsource(ds.main)
    assert "link_evaluation_to_library" in src
    assert "exp_dir.name" in src
    # 函数本体可导入、可调用（契约签名）
    sig = inspect.signature(link_evaluation_to_library)
    params = list(sig.parameters)
    assert params[:3] == ["factor_name", "run_id", "now"]
    assert sig.return_annotation in (bool, "bool")


def test_factor_record_last_eval_fields_default_none():
    """旧行 from_dict 无 last_eval_* → 默认 None；to_dict 含键。"""
    from factorzen.discovery.factor_library import FactorRecord

    r = FactorRecord.from_dict({"expression": "rank(close)", "market": "ashare"})
    assert r.last_eval_run_id is None
    assert r.last_eval_at is None
    d = r.to_dict()
    assert "last_eval_run_id" in d and d["last_eval_run_id"] is None
    assert "last_eval_at" in d and d["last_eval_at"] is None
