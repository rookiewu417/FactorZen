# tests/test_team_experiment_index.py
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex


def _recs():
    return [
        {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
         "holdout_ic": 0.03, "dsr": 0.7, "passed": True, "verdict": "keep"},
        {"expression": "rank(vol)", "hypothesis": "换手", "ic_train": 0.001,
         "holdout_ic": 0.0, "dsr": 0.1, "passed": False, "verdict": "drop"},
    ]


def test_append_then_load_roundtrip(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
    idx.append(_recs())
    idx2 = ExperimentIndex(str(tmp_path / "exp.jsonl"))   # 新实例，跨 "session"
    loaded = idx2.load()
    assert len(loaded) == 2
    assert loaded[0]["expression"] == "ts_mean(close,5)"


def test_seen_expressions_normalized(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
    idx.append(_recs())
    seen = idx.seen_expressions()
    # 归一化形式（带空格）应能匹配无空格原始查询
    assert "ts_mean(close, 5)" in seen           # 归一化后带空格
    assert "rank(vol)" in seen


def test_known_invalid_and_valid(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "exp.jsonl"))
    idx.append(_recs())
    assert "rank(vol)" in idx.known_invalid(k=5)      # passed=False / 低 IC
    assert "ts_mean(close, 5)" in idx.known_valid(k=5) # passed=True（归一化）
    assert "ts_mean(close, 5)" not in idx.known_invalid(k=5)


def test_load_missing_file_empty(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "nope.jsonl"))
    assert idx.load() == [] and idx.seen_expressions() == set()
