"""任务运行状态记录器。

将每次 job 运行的 start/end/status/error 追加写入 JSONL 文件，
便于审计与监控。
"""

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

STATE_FILE = Path("output/automation/runs.jsonl")
_write_lock = threading.Lock()


def _write_record(
    job_name: str,
    start_ts: str,
    end_ts: str,
    status: str,
    error: str | None,
) -> None:
    """将单条运行记录追加到 STATE_FILE。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "job_name": job_name,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "status": status,
        "error": error,
    }
    with _write_lock, STATE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def run_record(job_name: str) -> Generator[None, None, None]:
    """记录任务运行开始/结束/状态的上下文管理器。

    成功时写入 status="success"，异常时写入 status="failure" 并包含错误信息。
    异常会在记录后重新抛出。

    Parameters
    ----------
    job_name : str
        任务名称，作为记录的唯一标识。

    Yields
    ------
    None
    """
    start = datetime.now().isoformat()
    try:
        yield
        _write_record(job_name, start, datetime.now().isoformat(), "success", None)
    except Exception as exc:
        _write_record(job_name, start, datetime.now().isoformat(), "failure", str(exc))
        raise


def load_runs(n: int = 100) -> list[dict]:
    """读取最后 n 条运行记录。

    Parameters
    ----------
    n : int
        最多返回的记录条数（从最新开始取）。

    Returns
    -------
    list[dict]
        按写入顺序（最旧到最新）返回的记录列表。
    """
    if not STATE_FILE.exists():
        return []
    lines = STATE_FILE.read_text(encoding="utf-8").splitlines()
    recent = lines[-n:] if len(lines) > n else lines
    records = []
    for line in recent:
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
