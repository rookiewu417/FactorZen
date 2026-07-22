"""运营 / 报告 API 与 OpsViewIndex 测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from factorzen.server import opsview as opsview_mod
from factorzen.server.api import create_app
from factorzen.server.opsview import OpsViewIndex


def _client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_ops_campaigns_list(tmp_path):
    """campaign 列表 done/exitcode/command 解析。"""
    camp = tmp_path / "_ops" / "campaigns" / "mine_a"
    camp.mkdir(parents=True)
    (camp / "command.txt").write_text(
        "fz mine team --market ashare --iterations 60", encoding="utf-8"
    )
    (camp / "done").write_text("", encoding="utf-8")
    (camp / "exitcode").write_text("0\n", encoding="utf-8")
    (camp / "mine.log").write_text("line1\nline2\n", encoding="utf-8")

    # 未完成的 campaign
    camp2 = tmp_path / "_ops" / "campaigns" / "mine_b"
    camp2.mkdir(parents=True)
    (camp2 / "command.txt").write_text("x" * 300, encoding="utf-8")

    r = _client(tmp_path).get("/api/ops/campaigns")
    assert r.status_code == 200
    body = r.json()
    by_name = {c["name"]: c for c in body["campaigns"]}
    assert set(by_name) == {"mine_a", "mine_b"}

    a = by_name["mine_a"]
    assert a["done"] is True
    assert a["exitcode"] == "0"
    assert a["command"] is not None
    assert a["command"].startswith("fz mine")
    assert a["mtime"]  # 非空 iso 字符串

    b = by_name["mine_b"]
    assert b["done"] is False
    assert b["exitcode"] is None
    assert len(b["command"]) == 200  # 截断到 200


def test_ops_campaigns_empty(tmp_path):
    r = _client(tmp_path).get("/api/ops/campaigns")
    assert r.status_code == 200
    assert r.json()["campaigns"] == []


def test_ops_campaign_log_tail(tmp_path):
    camp = tmp_path / "_ops" / "campaigns" / "c1"
    camp.mkdir(parents=True)
    lines = [f"L{i}" for i in range(50)]
    (camp / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 旧 log
    (camp / "old.log").write_text("old\n", encoding="utf-8")
    import time

    time.sleep(0.05)
    (camp / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    r = _client(tmp_path).get("/api/ops/campaigns/c1/log", params={"tail": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "c1"
    assert body["log_file"] == "run.log"
    assert len(body["lines"]) == 10
    assert body["lines"][-1] == "L49"
    assert body["lines"][0] == "L40"


def test_ops_campaign_log_no_log(tmp_path):
    camp = tmp_path / "_ops" / "campaigns" / "empty"
    camp.mkdir(parents=True)
    r = _client(tmp_path).get("/api/ops/campaigns/empty/log")
    assert r.status_code == 200
    assert r.json()["lines"] == []
    assert r.json()["log_file"] is None


def test_ops_campaign_path_traversal_404(tmp_path):
    (tmp_path / "_ops" / "campaigns").mkdir(parents=True)
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "x.log").write_text("secret\n", encoding="utf-8")

    client = _client(tmp_path)
    r = client.get("/api/ops/campaigns/../evil/log")
    # FastAPI 可能把路径规范化；无论怎样不能读到 secret
    assert r.status_code in (404, 405)

    import pytest

    with pytest.raises(FileNotFoundError):
        OpsViewIndex(tmp_path).campaign_log("../evil")


def test_reports_list_filters_extensions(tmp_path):
    """只列 .json/.md/.html/.txt；按 mtime 降序。"""
    base = tmp_path / "reports"
    (base / "sub").mkdir(parents=True)
    (base / "a.json").write_text("{}", encoding="utf-8")
    (base / "b.md").write_text("# hi", encoding="utf-8")
    (base / "c.bin").write_text("\x00", encoding="utf-8")
    (base / "sub" / "d.html").write_text("<p>x</p>", encoding="utf-8")
    (base / "sub" / "e.txt").write_text("txt", encoding="utf-8")

    r = _client(tmp_path).get("/api/reports")
    assert r.status_code == 200
    files = r.json()["files"]
    paths = {f["path"] for f in files}
    assert "a.json" in paths
    assert "b.md" in paths
    assert "sub/d.html" in paths
    assert "sub/e.txt" in paths
    assert "c.bin" not in paths
    for f in files:
        assert "size" in f and "mtime" in f


def test_reports_file_read(tmp_path):
    base = tmp_path / "reports"
    base.mkdir()
    (base / "note.md").write_text("# title\nbody\n", encoding="utf-8")

    r = _client(tmp_path).get("/api/reports/file", params={"path": "note.md"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "note.md"
    assert body["content"] == "# title\nbody\n"
    assert body["size"] > 0


def test_reports_path_traversal_404(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "secret.txt").write_text("leak", encoding="utf-8")

    client = _client(tmp_path)
    for path in ("../secret.txt", "../../etc/passwd", "foo/../../../secret.txt"):
        r = client.get("/api/reports/file", params={"path": path})
        assert r.status_code == 404, f"path={path!r} got {r.status_code}"


def test_reports_disallowed_extension_404(tmp_path):
    base = tmp_path / "reports"
    base.mkdir()
    (base / "x.bin").write_text("data", encoding="utf-8")
    r = _client(tmp_path).get("/api/reports/file", params={"path": "x.bin"})
    assert r.status_code == 404


def test_reports_file_too_large_413(tmp_path, monkeypatch):
    """超限返回 413；上限 monkeypatch 为小值。"""
    monkeypatch.setattr(opsview_mod, "REPORT_MAX_BYTES", 10)
    base = tmp_path / "reports"
    base.mkdir()
    (base / "big.txt").write_text("x" * 50, encoding="utf-8")

    r = _client(tmp_path).get("/api/reports/file", params={"path": "big.txt"})
    assert r.status_code == 413
