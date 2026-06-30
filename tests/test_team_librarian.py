# tests/test_team_librarian.py
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AttemptRecord


def test_record_then_recall_roundtrip(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
    ]
    record(idx, attempts, run_id="r1")
    r = recall(idx, k=5)
    assert "ts_mean(close, 5)" in r.seen and "rank(vol)" in r.seen   # 归一化查重集
    assert "rank(vol)" in r.known_invalid                            # 未过护栏
    assert "ts_mean(close, 5)" in r.known_valid                      # 过护栏


def test_recall_empty_index(tmp_path: Path):
    r = recall(ExperimentIndex(str(tmp_path / "none.jsonl")), k=5)
    assert r.seen == set() and r.known_invalid == [] and r.known_valid == []
