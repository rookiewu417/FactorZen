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


def test_record_backfills_holdout_ic(tmp_path: Path):
    """record(candidates=...) → holdout_ic 写入 index → known_valid 按 holdout_ic 降序。

    注意：idx.load() 返回原始（非归一化）expression 字符串，
    candidates 可用归一化或非归一化形式，_normalize 匹配均可；
    known_valid() 返回归一化形式。
    """
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "反转", "rank(vol)", True, 0.03, True, "keep", None, ir_train=0.2),
    ]
    # rank(vol) 的 holdout_ic 更高——若排序正确，known_valid[0] 应为 rank(vol)
    # candidates 用归一化形式（空格）验证 _normalize 匹配路径
    candidates = [
        {"expression": "ts_mean(close, 5)", "holdout_ic": 0.02, "ic_train": 0.05},
        {"expression": "rank(vol)", "holdout_ic": 0.06, "ic_train": 0.03},
    ]
    record(idx, attempts, run_id="r1", candidates=candidates)
    recs = idx.load()
    # idx.load() 返回原始 expression（AttemptRecord.expression，无空格）
    hic_map = {r["expression"]: r.get("holdout_ic") for r in recs}
    assert hic_map.get("ts_mean(close,5)") == 0.02, f"holdout_ic 未写入: {hic_map}"
    assert hic_map.get("rank(vol)") == 0.06
    # known_valid 按 holdout_ic 降序，返回归一化形式：rank(vol) 排第一
    r = recall(idx, k=5)
    assert r.known_valid[0] == "rank(vol)", f"期望 rank(vol) 排第一，实际 {r.known_valid}"


def test_record_backfills_critic_verdict(tmp_path: Path):
    """AttemptRecord.critic_verdict 非 None 时，record 正确写入 verdict 字段。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
    ]
    record(idx, attempts, run_id="r1")
    recs = idx.load()
    # idx.load() 返回原始 expression（AttemptRecord.expression，无空格）
    verdict_map = {r["expression"]: r.get("verdict") for r in recs}
    assert verdict_map.get("ts_mean(close,5)") == "keep"
    assert verdict_map.get("rank(vol)") == "drop"
