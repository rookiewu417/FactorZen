"""factor_evaluations 域：嵌套 factors 布局扫描与寻址。"""

from __future__ import annotations

import json

from factorzen.server.artifacts import ArtifactIndex


def _write_manifest(d, run_id: str, status: str = "success"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "git_sha": "abc",
                "config": {"factor": "x"},
            }
        ),
        encoding="utf-8",
    )


def test_list_runs_nested_and_skips_reports(tmp_path):
    factors = tmp_path / "factors"
    r1 = factors / "ashare" / "mom" / "evaluations" / "mom_20260101_000001"
    r2 = factors / "_runs" / "sweep_20260101_000002"
    _write_manifest(r1, "mom_20260101_000001")
    _write_manifest(r2, "sweep_20260101_000002")
    # reports 下伪造 manifest，不得当 run
    fake = factors / "reports" / "daily"
    fake.mkdir(parents=True)
    (fake / "manifest.json").write_text("{}", encoding="utf-8")

    runs = ArtifactIndex(tmp_path).list_runs("factor_evaluations")
    ids = {r["run_id"] for r in runs}
    assert ids == {"mom_20260101_000001", "sweep_20260101_000002"}


def test_run_detail_nested(tmp_path):
    factors = tmp_path / "factors"
    rid = "alpha_20260102_120000"
    d = factors / "ashare" / "alpha" / "evaluations" / rid
    _write_manifest(d, rid)
    (d / "metrics.json").write_text(json.dumps({"ic": 0.01}), encoding="utf-8")

    detail = ArtifactIndex(tmp_path).run_detail("factor_evaluations", rid)
    assert detail["run_id"] == rid
    assert detail["metrics"]["ic"] == 0.01
    # 前端产物链接以 path 为准：嵌套域返回真实 workspace 相对路径
    assert detail["path"] == f"factors/ashare/alpha/evaluations/{rid}"


def test_run_detail_flat_domain_path(tmp_path):
    rid = "sim_001"
    d = tmp_path / "sim" / rid
    _write_manifest(d, rid)
    detail = ArtifactIndex(tmp_path).run_detail("sim", rid)
    assert detail["path"] == f"sim/{rid}"


def test_safe_run_dir_rejects_traversal(tmp_path):
    import pytest

    idx = ArtifactIndex(tmp_path)
    with pytest.raises(FileNotFoundError):
        idx.run_detail("factor_evaluations", "../../evil")


def test_store_root_for_library_maps_to_factors(tmp_path):
    from factorzen.discovery.factor_store import store_root_for_library

    lib = tmp_path / "custom_lib"
    lib.mkdir()
    assert store_root_for_library(str(lib)).endswith("/factors")
    assert store_root_for_library(str(lib)) == str(lib / "factors")
