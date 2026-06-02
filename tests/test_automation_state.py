"""automation/state.py 单元测试。

测试 run_record 上下文管理器和 load_runs 函数。
"""

import json

import pytest

from factorzen.automation.state import _write_record, load_runs, run_record


@pytest.fixture(autouse=True)
def tmp_state_file(tmp_path, monkeypatch):
    """将 STATE_FILE 重定向到临时目录，保证测试隔离。"""
    tmp_file = tmp_path / "automation" / "runs.jsonl"
    monkeypatch.setattr("factorzen.automation.state.STATE_FILE", tmp_file)
    yield tmp_file


# ────────────────────────────────────────────────────────────────────────────────
# _write_record
# ────────────────────────────────────────────────────────────────────────────────


def test_write_record_creates_file(tmp_state_file):
    """_write_record 应自动创建父目录和文件。"""
    _write_record("test_job", "2025-01-01T00:00:00", "2025-01-01T00:00:01", "success", None)
    assert tmp_state_file.exists()


def test_write_record_valid_json(tmp_state_file):
    """_write_record 写入的每行应是合法 JSON。"""
    _write_record("test_job", "2025-01-01T00:00:00", "2025-01-01T00:00:01", "success", None)
    lines = tmp_state_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["job_name"] == "test_job"
    assert record["status"] == "success"
    assert record["error"] is None


def test_write_record_appends(tmp_state_file):
    """多次调用 _write_record 应追加，不覆盖。"""
    _write_record("job1", "2025-01-01T00:00:00", "2025-01-01T00:00:01", "success", None)
    _write_record("job2", "2025-01-01T00:00:02", "2025-01-01T00:00:03", "failure", "err")
    lines = tmp_state_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["job_name"] == "job1"
    assert json.loads(lines[1])["job_name"] == "job2"


# ────────────────────────────────────────────────────────────────────────────────
# run_record context manager
# ────────────────────────────────────────────────────────────────────────────────


def test_run_record_success_writes_success(tmp_state_file):
    """成功完成的代码块应写入 status='success'。"""
    with run_record("my_job"):
        pass  # no exception

    records = load_runs()
    assert len(records) == 1
    assert records[0]["job_name"] == "my_job"
    assert records[0]["status"] == "success"
    assert records[0]["error"] is None


def test_run_record_failure_writes_failure(tmp_state_file):
    """抛出异常的代码块应写入 status='failure' 并包含 error 字符串。"""
    with pytest.raises(ValueError, match="something went wrong"), run_record("failing_job"):
        raise ValueError("something went wrong")

    records = load_runs()
    assert len(records) == 1
    assert records[0]["job_name"] == "failing_job"
    assert records[0]["status"] == "failure"
    assert "something went wrong" in records[0]["error"]


def test_run_record_reraises_exception(tmp_state_file):
    """run_record 必须在记录后重新抛出原始异常。"""
    with pytest.raises(RuntimeError), run_record("job"):
        raise RuntimeError("boom")


def test_run_record_records_timestamps(tmp_state_file):
    """start_ts 和 end_ts 字段应存在且都是非空字符串。"""
    with run_record("timed_job"):
        pass

    records = load_runs()
    assert records[0]["start_ts"]
    assert records[0]["end_ts"]


def test_run_record_multiple(tmp_state_file):
    """多次使用 run_record 应产生多条独立记录。"""
    with run_record("job_a"):
        pass
    with pytest.raises(KeyError), run_record("job_b"):
        raise KeyError("k")
    with run_record("job_c"):
        pass

    records = load_runs()
    assert len(records) == 3
    statuses = [r["status"] for r in records]
    assert statuses == ["success", "failure", "success"]


# ────────────────────────────────────────────────────────────────────────────────
# load_runs
# ────────────────────────────────────────────────────────────────────────────────


def test_load_runs_empty_when_no_file(tmp_state_file):
    """STATE_FILE 不存在时，load_runs 应返回空列表。"""
    assert not tmp_state_file.exists()
    result = load_runs()
    assert result == []


def test_load_runs_returns_records_in_order(tmp_state_file):
    """load_runs 应按写入顺序（最旧→最新）返回记录。"""
    for i in range(5):
        _write_record(f"job_{i}", f"2025-01-0{i+1}T00:00:00", f"2025-01-0{i+1}T00:00:01",
                      "success", None)
    records = load_runs()
    assert [r["job_name"] for r in records] == [f"job_{i}" for i in range(5)]


def test_load_runs_limits_to_n(tmp_state_file):
    """load_runs(n) 应只返回最后 n 条记录。"""
    for i in range(10):
        _write_record(f"job_{i}", "ts", "ts", "success", None)
    records = load_runs(n=3)
    assert len(records) == 3
    assert records[0]["job_name"] == "job_7"
    assert records[2]["job_name"] == "job_9"


def test_load_runs_returns_all_when_fewer_than_n(tmp_state_file):
    """记录数少于 n 时，load_runs 应返回所有记录。"""
    _write_record("only_job", "ts", "ts", "success", None)
    records = load_runs(n=100)
    assert len(records) == 1
