"""W3 A2/A3: lift_rejected 写回 experiment_index 与召回通道。"""
from __future__ import annotations

from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIBRARY_CORRELATED,
    REJECT_CATEGORY_LIFT_REJECTED,
)

_DW = {
    "start": "20200101",
    "end": "20201231",
    "universe": "csi300",
    "market": "ashare",
}
_DW_OTHER = {
    "start": "20210101",
    "end": "20211231",
    "universe": "csi300",
    "market": "ashare",
}


def _lift_reject(
    expr: str,
    *,
    lift: float | None = 0.0005,
    reason: str = "below_bar",
    ts: str | None = "2026-01-02T00:00:00",
    data_window: dict | None = None,
    ic_train: float | None = 0.02,
) -> dict:
    rec: dict = {
        "expression": expr,
        "data_window": data_window if data_window is not None else dict(_DW),
        "reject_category": REJECT_CATEGORY_LIFT_REJECTED,
        "passed": False,
        "compile_ok": True,
        "ic_train": ic_train,
        "residual_ic_train": 0.008,
        "lift": lift,
        "lift_se": 0.001,
        "lift_reason": reason,
        "baseline_rank_ic": 0.03,
        "admission_start": "2020-10-01",
        "admission_end": "2020-12-31",
        "source": "session_auto_lift",
    }
    if ts is not None:
        rec["ts"] = ts
    return rec


def test_reject_category_constant():
    assert REJECT_CATEGORY_LIFT_REJECTED == "lift_rejected"


def test_known_lift_rejects_recalls_and_scopes(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        _lift_reject("rank(vol)", ts="2026-01-01T00:00:00", lift=0.0001),
        _lift_reject("ts_mean(close, 5)", ts="2026-01-03T00:00:00", lift=0.0002),
        _lift_reject(
            "rank(amount)",
            ts="2026-01-04T00:00:00",
            lift=0.0003,
            data_window=_DW_OTHER,
        ),
        # 非 lift_rejected 不应召回
        {
            "expression": "rank(open)",
            "data_window": dict(_DW),
            "reject_category": REJECT_CATEGORY_LIBRARY_CORRELATED,
            "passed": False,
            "compile_ok": True,
            "ic_train": 0.01,
            "ts": "2026-01-05T00:00:00",
        },
    ])
    out = idx.known_lift_rejects(k=5, data_window=_DW)
    exprs = [r["expression"] for r in out]
    assert "ts_mean(close, 5)" in exprs or any("ts_mean" in e for e in exprs)
    assert "rank(vol)" in exprs
    # 跨窗口不召回
    assert not any("amount" in e for e in exprs)
    # 形状
    for r in out:
        assert set(r.keys()) == {"expression", "lift", "lift_reason"}
    # ts 降序：ts_mean 更新
    assert out[0]["expression"].startswith("ts_mean") or "close" in out[0]["expression"]
    assert out[0]["lift"] == 0.0002


def test_known_lift_rejects_last_wins(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        _lift_reject("rank(vol)", reason="group_gate_fail", lift=None, ts="2026-01-01T00:00:00"),
        _lift_reject("rank(vol)", reason="below_bar", lift=0.0004, ts="2026-01-02T00:00:00"),
    ])
    out = idx.known_lift_rejects(k=5, data_window=_DW)
    assert len(out) == 1
    assert out[0]["lift_reason"] == "below_bar"
    assert out[0]["lift"] == 0.0004


def test_known_invalid_excludes_lift_rejected(tmp_path: Path):
    """lift_rejected 不进 known_invalid，进 known_lift_rejects。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        _lift_reject("rank(vol)", lift=0.0001),
        {
            "expression": "rank(open)",
            "data_window": dict(_DW),
            "passed": False,
            "compile_ok": True,
            "ic_train": 0.001,
        },
    ])
    invalid = idx.known_invalid(k=10, data_window=_DW)
    lift_r = idx.known_lift_rejects(k=10, data_window=_DW)
    assert "rank(vol)" not in invalid
    assert any(r["expression"] == "rank(vol)" for r in lift_r)
    assert "rank(open)" in invalid


def test_leaf_stats_counts_lift_rejected(tmp_path: Path):
    """lift_rejected 仍计 n_exprs（compile_ok=True）。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([_lift_reject("ts_mean(holder_num_chg, 5)", ic_train=0.01)])
    stats = idx.leaf_stats(["holder_num_chg"], data_window=_DW)
    assert stats["holder_num_chg"]["n_exprs"] == 1
    assert stats["holder_num_chg"]["n_passed"] == 0
