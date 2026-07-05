"""只读 REST API 的测试(TestClient 全离线)。"""
from __future__ import annotations

import json

import polars as pl
from fastapi.testclient import TestClient

from factorzen.server.api import create_app


def _client(tmp_path):
    return TestClient(create_app(tmp_path))


def _write_run(root, domain, run_id, manifest):
    d = root / domain / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def test_health(tmp_path):
    r = _client(tmp_path).get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "portfolios" in body["domains"]


def test_runs_lists(tmp_path):
    _write_run(tmp_path, "portfolios", "run1", {"git_sha": "abc", "status": "optimal"})
    r = _client(tmp_path).get("/api/runs", params={"domain": "portfolios"})
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert runs[0]["run_id"] == "run1" and runs[0]["git_sha"] == "abc"


def test_runs_unknown_domain_404(tmp_path):
    assert _client(tmp_path).get("/api/runs", params={"domain": "xxx"}).status_code == 404


def test_run_detail(tmp_path):
    _write_run(tmp_path, "sim", "s1", {"run_id": "s1", "status": "ok"})
    r = _client(tmp_path).get("/api/runs/sim/s1")
    assert r.status_code == 200
    assert r.json()["manifest"]["run_id"] == "s1"


def test_run_detail_missing_404(tmp_path):
    assert _client(tmp_path).get("/api/runs/sim/nope").status_code == 404


def test_nav(tmp_path):
    _write_run(tmp_path, "execution", "e1", {})
    pl.DataFrame(
        {"as_of_date": ["2026-01-05", "2026-01-06"], "nav_after": [1_000_000.0, 1_010_000.0]}
    ).write_parquet(tmp_path / "execution" / "e1" / "nav.parquet")
    r = _client(tmp_path).get("/api/nav/execution/e1")
    assert r.status_code == 200
    nav = r.json()["nav"]
    assert nav[0] == ["2026-01-05", 1_000_000.0]


def test_openapi_docs_available(tmp_path):
    assert _client(tmp_path).get("/openapi.json").status_code == 200


def test_dashboard_page(tmp_path):
    _write_run(tmp_path, "portfolios", "run1", {"git_sha": "abc123def", "status": "optimal"})
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert "FactorZen" in r.text
    assert "run1" in r.text


def test_dashboard_page_empty_workspace(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
