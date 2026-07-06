"""只读产物索引 ArtifactIndex 的测试(零侵入,不触发计算,损坏 manifest 跳过)。"""
from __future__ import annotations

import json

import polars as pl

from factorzen.server.artifacts import DOMAINS, ArtifactIndex


def _write_run(root, domain, run_id, manifest, metrics=None):
    d = root / domain / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return d


def test_domains_cover_core():
    assert {"portfolios", "sim", "execution", "combinations"} <= set(DOMAINS)


def test_list_runs(tmp_path):
    _write_run(tmp_path, "portfolios", "run1", {"git_sha": "abc", "status": "optimal"})
    runs = ArtifactIndex(tmp_path).list_runs("portfolios")
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run1"
    assert runs[0]["git_sha"] == "abc"


def test_list_runs_empty_domain(tmp_path):
    assert ArtifactIndex(tmp_path).list_runs("portfolios") == []


def test_list_runs_skips_corrupt_manifest(tmp_path):
    d = tmp_path / "sim" / "bad"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{ not valid json", encoding="utf-8")
    # 损坏 manifest 跳过,不抛
    assert ArtifactIndex(tmp_path).list_runs("sim") == []


def test_run_detail_with_metrics(tmp_path):
    _write_run(tmp_path, "sim", "s1", {"run_id": "s1"}, metrics={"sharpe": 1.5})
    detail = ArtifactIndex(tmp_path).run_detail("sim", "s1")
    assert detail["manifest"]["run_id"] == "s1"
    assert detail["metrics"]["sharpe"] == 1.5


def test_run_detail_missing_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        ArtifactIndex(tmp_path).run_detail("sim", "nope")


def test_nav_series_execution(tmp_path):
    _write_run(tmp_path, "execution", "e1", {})
    pl.DataFrame(
        {"as_of_date": ["2026-01-05", "2026-01-06"], "nav_after": [1_000_000.0, 1_010_000.0]}
    ).write_parquet(tmp_path / "execution" / "e1" / "nav.parquet")
    nav = ArtifactIndex(tmp_path).nav_series("execution", "e1")
    assert nav == [("2026-01-05", 1_000_000.0), ("2026-01-06", 1_010_000.0)]


def test_nav_series_absent_returns_empty(tmp_path):
    _write_run(tmp_path, "sim", "s1", {})
    assert ArtifactIndex(tmp_path).nav_series("sim", "s1") == []


def test_nav_series_rejects_path_traversal_run_id(tmp_path):
    """run_id 含 ../ 不应逃出 workspace/<domain> 去读外部文件。"""
    ws = tmp_path / "ws"
    (ws / "sim").mkdir(parents=True)
    evil = tmp_path / "evil"
    evil.mkdir()
    pl.DataFrame({"as_of_date": ["2026-01-01"], "nav_after": [42.0]}).write_parquet(
        evil / "nav.parquet"
    )
    # run_id="../../evil" → ws/sim/../../evil = tmp_path/evil（逃出 workspace）
    result = ArtifactIndex(ws).nav_series("sim", "../../evil")
    assert result == [], f"路径遍历应被拒绝，却读到 {result}"


def test_run_detail_rejects_path_traversal_run_id(tmp_path):
    import pytest

    ws = tmp_path / "ws"
    (ws / "sim").mkdir(parents=True)
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "manifest.json").write_text('{"secret": 1}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ArtifactIndex(ws).run_detail("sim", "../../evil")


def test_run_detail_rejects_non_whitelisted_domain(tmp_path):
    """非 DOMAINS 白名单的 domain 目录即使存在也不应被读取。"""
    import pytest

    d = tmp_path / "badcorp" / "x"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text('{"secret": 1}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ArtifactIndex(tmp_path).run_detail("badcorp", "x")


def test_nav_series_survives_corrupt_parquet(tmp_path):
    """nav.parquet 损坏时应记 warning 返回 []，而非抛异常拖垮 /api/nav 与 Dashboard。"""
    d = tmp_path / "sim" / "s1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    (d / "nav.parquet").write_text("not a valid parquet file", encoding="utf-8")
    assert ArtifactIndex(tmp_path).nav_series("sim", "s1") == []
