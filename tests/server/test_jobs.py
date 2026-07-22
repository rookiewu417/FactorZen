"""JobManager 与 /api/jobs* 端点测试（仅 python -c 级轻命令）。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factorzen.server.api import create_app
from factorzen.server.jobs import JobManager


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _wait_finished(jm: JobManager, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = jm.job_detail(job_id)
        if d.get("status") == "finished":
            return d
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} 未在 {timeout}s 内 finished: {jm.job_detail(job_id)}")


def test_submit_lifecycle_exit0_and_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    jm = JobManager(tmp_path / "jobs", workspace_dir=tmp_path, project_root=tmp_path)

    def _light(_kind: str, _argv: list[str]) -> list[str]:
        return [sys.executable, "-c", "print('hi')"]

    monkeypatch.setattr(jm, "_build_command", _light)
    meta = jm.submit([], kind="cli", title="hello")
    job_id = meta["job_id"]
    assert "pid" in meta
    assert meta["title"] == "hello"

    detail = _wait_finished(jm, job_id)
    assert detail["status"] == "finished"
    assert detail["exit_code"] == 0

    log = jm.job_log(job_id, tail=50)
    assert any("hi" in line for line in log["lines"])

    listed = jm.list_jobs()
    assert any(j["job_id"] == job_id for j in listed)


def test_kill_running_sleep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    jm = JobManager(tmp_path / "jobs", workspace_dir=tmp_path, project_root=tmp_path)

    def _sleep(_kind: str, _argv: list[str]) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(60)"]

    monkeypatch.setattr(jm, "_build_command", _sleep)
    meta = jm.submit([], kind="cli", title="sleeper")
    job_id = meta["job_id"]

    # 等到 running
    deadline = time.time() + 5
    while time.time() < deadline:
        d = jm.job_detail(job_id)
        if d.get("status") == "running":
            break
        time.sleep(0.05)
    else:
        pytest.fail("sleep 任务未进入 running")

    result = jm.kill(job_id)
    assert result["killed"] is True

    # 终止后应变为 finished 或至少不再 running
    deadline = time.time() + 5
    while time.time() < deadline:
        d = jm.job_detail(job_id)
        if d.get("status") != "running":
            break
        time.sleep(0.05)
    assert jm.job_detail(job_id)["status"] != "running"


def test_script_path_escape_and_non_py(tmp_path: Path):
    jm = JobManager(tmp_path / "jobs", workspace_dir=tmp_path, project_root=tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "ok.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "configs" / "note.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print(1)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configs"):
        jm._build_command("script", ["other.py"])

    with pytest.raises(ValueError, match=r"\.py"):
        jm._build_command("script", ["configs/note.txt"])

    with pytest.raises(ValueError, match=r"非法|逃逸|configs"):
        jm._build_command("script", ["configs/../../etc/passwd.py"])

    with pytest.raises(ValueError, match=r"非法"):
        jm._build_command("script", ["../secrets.py"])

    # 合法路径可构造
    cmd = jm._build_command("script", ["configs/ok.py", "--x"])
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("ok.py")
    assert cmd[2] == "--x"


def test_api_script_bad_paths_400(tmp_path: Path):
    client = _client(tmp_path)
    (tmp_path / "configs").mkdir()

    r = client.post(
        "/api/jobs",
        json={"kind": "script", "argv": ["evil.py"], "title": "x"},
    )
    assert r.status_code == 400

    r2 = client.post(
        "/api/jobs",
        json={"kind": "script", "argv": ["configs/../x.py"], "title": "x"},
    )
    assert r2.status_code == 400


def test_job_id_traversal_404(tmp_path: Path):
    client = _client(tmp_path)
    for evil in ["../x", "a/b", "..", "foo/../../etc"]:
        assert client.get(f"/api/jobs/{evil}").status_code == 404
        assert client.get(f"/api/jobs/{evil}/log").status_code == 404
        assert client.post(f"/api/jobs/{evil}/kill").status_code == 404


def test_kill_non_running_409(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    jm = JobManager(tmp_path / "jobs", workspace_dir=tmp_path, project_root=tmp_path)

    def _hi(_kind: str, _argv: list[str]) -> list[str]:
        return [sys.executable, "-c", "print(1)"]

    monkeypatch.setattr(jm, "_build_command", _hi)
    meta = jm.submit([], kind="cli", title="done")
    _wait_finished(jm, meta["job_id"])

    with pytest.raises(RuntimeError, match="非 running"):
        jm.kill(meta["job_id"])

    # API 层
    client = _client(tmp_path)
    # 把已完成 job 的目录拷到 client 的 jobs 下较麻烦；直接用同一 jm 的 jobs_dir 建 app 不方便
    # 改用伪造 finished meta
    jobs_dir = tmp_path / "_ops" / "webui_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jid = "20200101_000000_dead01"
    jd = jobs_dir / jid
    jd.mkdir()
    (jd / "meta.json").write_text(
        json.dumps(
            {
                "job_id": jid,
                "kind": "cli",
                "title": "old",
                "argv": [],
                "pid": 1,
                "started_at": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (jd / "status.json").write_text(
        json.dumps({"exit_code": 0, "ended_at": "2020-01-01T00:00:01+00:00"}),
        encoding="utf-8",
    )
    r = client.post(f"/api/jobs/{jid}/kill")
    assert r.status_code == 409


def test_orphaned_dead_pid(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    jid = "20200101_000000_orphan"
    jd = jobs_dir / jid
    jd.mkdir()
    # 用不存在的 pid
    dead_pid = 2**22  # 通常不存在
    (jd / "meta.json").write_text(
        json.dumps(
            {
                "job_id": jid,
                "kind": "cli",
                "title": "ghost",
                "argv": [],
                "pid": dead_pid,
                "started_at": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    jm = JobManager(jobs_dir, workspace_dir=tmp_path, project_root=tmp_path)
    detail = jm.job_detail(jid)
    assert detail["status"] == "orphaned"

    listed = jm.list_jobs()
    assert listed[0]["status"] == "orphaned"


def test_unknown_kind_400(tmp_path: Path):
    client = _client(tmp_path)
    r = client.post(
        "/api/jobs",
        json={"kind": "shell", "argv": ["ls"], "title": "nope"},
    )
    assert r.status_code == 400


def test_cli_argv_null_byte_rejected(tmp_path: Path):
    jm = JobManager(tmp_path / "jobs", workspace_dir=tmp_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="空字节"):
        jm._build_command("cli", ["--x\x00y"])
