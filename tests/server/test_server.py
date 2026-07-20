"""
test_server_api.py：只读 REST API 的测试(TestClient 全离线)
test_server_artifacts.py：只读产物索引 ArtifactIndex 的测试(零侵入,不触发计算,损坏 manifest 跳过)
"""

from __future__ import annotations

import json

import polars as pl
from fastapi.testclient import TestClient

from factorzen.server.api import create_app
from factorzen.server.artifacts import DOMAINS, ArtifactIndex


# ==== 来自 test_server_api.py ====
def _client(tmp_path):
    return TestClient(create_app(tmp_path))

def _write_run__server_api(root, domain, run_id, manifest):
    d = root / domain / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d

def test_http_api_suite(tmp_path):
    """test_health；test_runs_lists；test_runs_unknown_domain_404；test_run_detail；test_run_detail_missing_404；test_nav；test_openapi_docs_available；test_dashboard_page；test_dashboard_page_empty_workspace；test_nav_unknown_domain_404"""
    # -- 原 test_health --
    def _section_0_test_health(tmp_path):
        r = _client(tmp_path).get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "portfolios" in body["domains"]

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_health(_tp0)

    # -- 原 test_runs_lists --
    def _section_1_test_runs_lists(tmp_path):
        _write_run__server_api(tmp_path, "portfolios", "run1", {"git_sha": "abc", "status": "optimal"})
        r = _client(tmp_path).get("/api/runs", params={"domain": "portfolios"})
        assert r.status_code == 200
        runs = r.json()["runs"]
        assert runs[0]["run_id"] == "run1" and runs[0]["git_sha"] == "abc"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_runs_lists(_tp1)

    # -- 原 test_runs_unknown_domain_404 --
    def _section_2_test_runs_unknown_domain_404(tmp_path):
        assert _client(tmp_path).get("/api/runs", params={"domain": "xxx"}).status_code == 404

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_runs_unknown_domain_404(_tp2)

    # -- 原 test_run_detail --
    def _section_3_test_run_detail(tmp_path):
        _write_run__server_api(tmp_path, "sim", "s1", {"run_id": "s1", "status": "ok"})
        r = _client(tmp_path).get("/api/runs/sim/s1")
        assert r.status_code == 200
        assert r.json()["manifest"]["run_id"] == "s1"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_run_detail(_tp3)

    # -- 原 test_run_detail_missing_404 --
    def _section_4_test_run_detail_missing_404(tmp_path):
        assert _client(tmp_path).get("/api/runs/sim/nope").status_code == 404

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_run_detail_missing_404(_tp4)

    # -- 原 test_nav --
    def _section_5_test_nav(tmp_path):
        _write_run__server_api(tmp_path, "execution", "e1", {})
        pl.DataFrame(
            {"as_of_date": ["2026-01-05", "2026-01-06"], "nav_after": [1_000_000.0, 1_010_000.0]}
        ).write_parquet(tmp_path / "execution" / "e1" / "nav.parquet")
        r = _client(tmp_path).get("/api/nav/execution/e1")
        assert r.status_code == 200
        nav = r.json()["nav"]
        assert nav[0] == ["2026-01-05", 1_000_000.0]

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_nav(_tp5)

    # -- 原 test_openapi_docs_available --
    def _section_6_test_openapi_docs_available(tmp_path):
        assert _client(tmp_path).get("/openapi.json").status_code == 200

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_openapi_docs_available(_tp6)

    # -- 原 test_dashboard_page --
    def _section_7_test_dashboard_page(tmp_path):
        _write_run__server_api(tmp_path, "portfolios", "run1", {"git_sha": "abc123def", "status": "optimal"})
        r = _client(tmp_path).get("/")
        assert r.status_code == 200
        assert "FactorZen" in r.text
        assert "run1" in r.text

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_dashboard_page(_tp7)

    # -- 原 test_dashboard_page_empty_workspace --
    def _section_8_test_dashboard_page_empty_workspace(tmp_path):
        r = _client(tmp_path).get("/")
        assert r.status_code == 200

    _tp8 = tmp_path / "_s8"
    _tp8.mkdir(exist_ok=True)
    _section_8_test_dashboard_page_empty_workspace(_tp8)

    # -- 原 test_nav_unknown_domain_404 --
    def _section_9_test_nav_unknown_domain_404(tmp_path):
        assert _client(tmp_path).get("/api/nav/badcorp/x").status_code == 404

    _tp9 = tmp_path / "_s9"
    _tp9.mkdir(exist_ok=True)
    _section_9_test_nav_unknown_domain_404(_tp9)


# ==== 来自 test_server_artifacts.py ====
def _write_run__server_artifacts(root, domain, run_id, manifest, metrics=None):
    d = root / domain / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return d

def test_server_domain_services_suite(tmp_path):
    """test_domains_cover_core；test_list_runs；test_list_runs_empty_domain；test_list_runs_skips_corrupt_manifest；test_run_detail_with_metrics；test_run_detail_missing_raises；test_nav_series_execution；test_nav_series_absent_returns_empty；run_id 含 ../ 不应逃出 workspace/<domain> 去读外部文件。；test_run_detail_rejects_path_traversal_run_id；非 DOMAINS 白名单的 domain 目录即使存在也不应被读取。；nav.parquet 损坏时应记 warning 返回 []，而非抛异常拖垮 /api/nav 与 Dashboard。"""
    # -- 原 test_domains_cover_core --
    def _section_0_test_domains_cover_core():
        assert {"portfolios", "sim", "execution", "combinations"} <= set(DOMAINS)

    _section_0_test_domains_cover_core()

    # -- 原 test_list_runs --
    def _section_1_test_list_runs(tmp_path):
        _write_run__server_artifacts(tmp_path, "portfolios", "run1", {"git_sha": "abc", "status": "optimal"})
        runs = ArtifactIndex(tmp_path).list_runs("portfolios")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run1"
        assert runs[0]["git_sha"] == "abc"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_list_runs(_tp1)

    # -- 原 test_list_runs_empty_domain --
    def _section_2_test_list_runs_empty_domain(tmp_path):
        assert ArtifactIndex(tmp_path).list_runs("portfolios") == []

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_list_runs_empty_domain(_tp2)

    # -- 原 test_list_runs_skips_corrupt_manifest --
    def _section_3_test_list_runs_skips_corrupt_manifest(tmp_path):
        d = tmp_path / "sim" / "bad"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("{ not valid json", encoding="utf-8")
        # 损坏 manifest 跳过,不抛
        assert ArtifactIndex(tmp_path).list_runs("sim") == []

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_list_runs_skips_corrupt_manifest(_tp3)

    # -- 原 test_run_detail_with_metrics --
    def _section_4_test_run_detail_with_metrics(tmp_path):
        _write_run__server_artifacts(tmp_path, "sim", "s1", {"run_id": "s1"}, metrics={"sharpe": 1.5})
        detail = ArtifactIndex(tmp_path).run_detail("sim", "s1")
        assert detail["manifest"]["run_id"] == "s1"
        assert detail["metrics"]["sharpe"] == 1.5

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_run_detail_with_metrics(_tp4)

    # -- 原 test_run_detail_missing_raises --
    def _section_5_test_run_detail_missing_raises(tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            ArtifactIndex(tmp_path).run_detail("sim", "nope")

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_run_detail_missing_raises(_tp5)

    # -- 原 test_nav_series_execution --
    def _section_6_test_nav_series_execution(tmp_path):
        _write_run__server_artifacts(tmp_path, "execution", "e1", {})
        pl.DataFrame(
            {"as_of_date": ["2026-01-05", "2026-01-06"], "nav_after": [1_000_000.0, 1_010_000.0]}
        ).write_parquet(tmp_path / "execution" / "e1" / "nav.parquet")
        nav = ArtifactIndex(tmp_path).nav_series("execution", "e1")
        assert nav == [("2026-01-05", 1_000_000.0), ("2026-01-06", 1_010_000.0)]

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_nav_series_execution(_tp6)

    # -- 原 test_nav_series_absent_returns_empty --
    def _section_7_test_nav_series_absent_returns_empty(tmp_path):
        _write_run__server_artifacts(tmp_path, "sim", "s1", {})
        assert ArtifactIndex(tmp_path).nav_series("sim", "s1") == []

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_nav_series_absent_returns_empty(_tp7)

    # -- 原 test_nav_series_rejects_path_traversal_run_id --
    def _section_8_test_nav_series_rejects_path_traversal_run_id(tmp_path):
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

    _tp8 = tmp_path / "_s8"
    _tp8.mkdir(exist_ok=True)
    _section_8_test_nav_series_rejects_path_traversal_run_id(_tp8)

    # -- 原 test_run_detail_rejects_path_traversal_run_id --
    def _section_9_test_run_detail_rejects_path_traversal_run_id(tmp_path):
        import pytest

        ws = tmp_path / "ws"
        (ws / "sim").mkdir(parents=True)
        evil = tmp_path / "evil"
        evil.mkdir()
        (evil / "manifest.json").write_text('{"secret": 1}', encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            ArtifactIndex(ws).run_detail("sim", "../../evil")

    _tp9 = tmp_path / "_s9"
    _tp9.mkdir(exist_ok=True)
    _section_9_test_run_detail_rejects_path_traversal_run_id(_tp9)

    # -- 原 test_run_detail_rejects_non_whitelisted_domain --
    def _section_10_test_run_detail_rejects_non_whitelisted_domain(tmp_path):
        import pytest

        d = tmp_path / "badcorp" / "x"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text('{"secret": 1}', encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            ArtifactIndex(tmp_path).run_detail("badcorp", "x")

    _tp10 = tmp_path / "_s10"
    _tp10.mkdir(exist_ok=True)
    _section_10_test_run_detail_rejects_non_whitelisted_domain(_tp10)

    # -- 原 test_nav_series_survives_corrupt_parquet --
    def _section_11_test_nav_series_survives_corrupt_parquet(tmp_path):
        d = tmp_path / "sim" / "s1"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("{}", encoding="utf-8")
        (d / "nav.parquet").write_text("not a valid parquet file", encoding="utf-8")
        assert ArtifactIndex(tmp_path).nav_series("sim", "s1") == []

    _tp11 = tmp_path / "_s11"
    _tp11.mkdir(exist_ok=True)
    _section_11_test_nav_series_survives_corrupt_parquet(_tp11)


