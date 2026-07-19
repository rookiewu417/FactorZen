"""W3 A4/A5: lift 拒绝写回 experiment_index（session 钩子 + CLI --apply）。"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.agents.team_orchestrator import _session_end_auto_lift
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIFT_QUEUE,
    REJECT_CATEGORY_LIFT_REJECTED,
)


class _FakeCtx:
    leaf_map = None


def _panel(n_dates: int, n_stocks: int = 10, start=dt.date(2022, 1, 3)):
    days, d = [], start
    while len(days) < n_dates:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for dd in days:
            rows.append({
                "trade_date": dd, "ts_code": c,
                "factor_value": float(hash((c, dd)) % 100) / 100.0,
            })
    return pl.DataFrame(rows)


def _holdout_and_mat():
    daily = _panel(200).rename({"factor_value": "close"}).with_columns(
        pl.col("close").alias("open"),
        pl.col("close").alias("high"),
        pl.col("close").alias("low"),
        (pl.col("close") * 1e6).alias("vol"),
        (pl.col("close") * 1e7).alias("amount"),
    )
    holdout = daily.filter(pl.col("trade_date") >= dt.date(2022, 8, 1))
    mat_panel = _panel(200)

    def mat(expr: str):
        return mat_panel

    return daily, holdout, mat


def _state_with_lift_queue(exprs: list[str]) -> AgentState:
    state = AgentState(seed=1)
    for e in exprs:
        state.attempts.append(AttemptRecord(
            0, "H", e, True, 0.02, False, "keep", None,
            ir_train=0.1, reject_category=REJECT_CATEGORY_LIFT_QUEUE,
            residual_ic_train=0.008,
        ))
    return state


def test_session_lift_group_gate_fail_writes_index(monkeypatch, tmp_path: Path):
    """组门不过 → kept 全体写 lift_rejected / group_gate_fail。"""
    idx_path = tmp_path / "experiment_index.jsonl"
    index = ExperimentIndex(str(idx_path))
    state = _state_with_lift_queue(["ts_mean(close, 5)", "rank(vol)"])
    daily, holdout, mat = _holdout_and_mat()
    dw = {"start": "20220101", "end": "20221231", "universe": "csi300", "market": "ashare"}

    def fake_group(*a, **k):
        return {
            "lift": 0.0001, "lift_se": 0.01, "error": None,
            "n_candidates": 2, "expressions": ["ts_mean(close, 5)", "rank(vol)"],
        }

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.run_lift_tests",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应跑逐候选")),
    )
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        lambda *a, **k: {"added_active": 0, "added_probation": 0, "rejected": 0},
        raising=False,
    )
    # coverage 全放行
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.make_lift_context",
        lambda *a, **k: SimpleNamespace(
            horizon=1, prepped=daily, admission_start="2022-08-01",
            admission_end="2022-12-31",
        ),
    )

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root=str(tmp_path / "lib"), seed=1,
        auto_lift=True, lift_se_mult=1.0, data_window=dw,
        materialize_candidate=mat, index=index, horizon=1,
    )
    del meta
    recs = index.load()
    asserts = [r for r in recs if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED]
    assert len(asserts) == 2
    for r in asserts:
        assert r["lift_reason"] == "group_gate_fail"
        assert r["passed"] is False
        assert r["compile_ok"] is True
        assert r["source"] == "session_auto_lift"
        assert r["data_window"] == dw
        assert r.get("ts")


def test_session_lift_below_bar_writes_index(monkeypatch, tmp_path: Path):
    """组门过、单候选 admission reject → below_bar 写回。"""
    idx_path = tmp_path / "experiment_index.jsonl"
    index = ExperimentIndex(str(idx_path))
    state = _state_with_lift_queue(["ts_mean(close, 5)", "rank(vol)"])
    daily, holdout, mat = _holdout_and_mat()
    dw = {"start": "20220101", "end": "20221231", "universe": "csi300", "market": "ashare"}

    def fake_group(*a, **k):
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "n_candidates": 2, "expressions": ["ts_mean(close, 5)", "rank(vol)"],
        }

    def fake_per(*a, **k):
        return [
            {
                "expression": "ts_mean(close, 5)",
                "lift": 0.0001, "lift_se": 0.01,
                "lift_second_half": -0.001, "baseline": 0.02, "passed": False,
            },
            {
                "expression": "rank(vol)",
                "lift": 0.008, "lift_se": 0.001,
                "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
            },
        ]

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        lambda *a, **k: {"added_active": 0, "added_probation": 1, "rejected": 1},
        raising=False,
    )
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.make_lift_context",
        lambda *a, **k: SimpleNamespace(
            horizon=1, prepped=daily, admission_start="2022-08-01",
            admission_end="2022-12-31",
        ),
    )

    _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root=str(tmp_path / "lib"), seed=1,
        data_window=dw, materialize_candidate=mat, index=index, horizon=1,
    )
    recs = [
        r for r in index.load()
        if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED
    ]
    assert len(recs) == 1
    assert recs[0]["expression"] == "ts_mean(close, 5)"
    assert recs[0]["lift_reason"] == "below_bar"
    assert recs[0]["lift"] == 0.0001
    assert recs[0]["source"] == "session_auto_lift"


def test_session_lift_index_none_zero_write(monkeypatch, tmp_path: Path):
    """index=None → 零写入（零回归）。"""
    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily, holdout, mat = _holdout_and_mat()

    def fake_group(*a, **k):
        return {
            "lift": 0.0001, "lift_se": 0.01, "error": None,
            "n_candidates": 1, "expressions": ["ts_mean(close, 5)"],
        }

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.make_lift_context",
        lambda *a, **k: SimpleNamespace(
            horizon=1, prepped=daily, admission_start="2022-08-01",
            admission_end="2022-12-31",
        ),
    )
    # 不应创建任何 index 文件
    idx_path = tmp_path / "should_not_exist.jsonl"
    _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root=str(tmp_path / "lib"), seed=1,
        materialize_candidate=mat, index=None, horizon=1,
    )
    assert not idx_path.exists()


# ── A5 CLI ──────────────────────────────────────────────────────────────────


def _write_session(tmp_path: Path, *, index_path: str | None, expr: str = "rank(close)") -> Path:
    run_dir = tmp_path / "sess1"
    run_dir.mkdir()
    params = {
        "start": "20200101",
        "end": "20201231",
        "universe": "csi300",
        "market": "ashare",
    }
    if index_path is not None:
        params["index_path"] = index_path
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [{
                "expression": expr,
                "reject_category": "lift_queue",
                "residual_ic_train": 0.02,
                "ic_train": 0.02,
                "n_residual_holdout_days": 100,
            }],
            "candidates": [],
            "params": params,
        }),
        encoding="utf-8",
    )
    return run_dir


def _fake_daily() -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.date(2020, 1, 2)],
        "ts_code": ["000001.SZ"],
        "close": [10.0],
        "close_adj": [10.0],
    })


def test_cli_apply_writes_lift_rejects_to_index(tmp_path, monkeypatch):
    """--apply → 写回来源 session 对应 index（含 group_gate_fail + below_bar）。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    # index 放在 session 父目录（回退路径）
    parent = tmp_path / "workspace"
    parent.mkdir()
    run_dir = parent / "sess1"
    run_dir.mkdir()
    # 故意写一个不存在的 worktree 绝对路径，触发回退
    bad_index = "/tmp/nonexistent_worktree_xyz/experiment_index.jsonl"
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [
                {
                    "expression": "rank(close)",
                    "reject_category": "lift_queue",
                    "residual_ic_train": 0.02,
                    "ic_train": 0.02,
                },
                {
                    "expression": "rank(vol)",
                    "reject_category": "lift_queue",
                    "residual_ic_train": 0.015,
                    "ic_train": 0.015,
                },
            ],
            "candidates": [],
            "params": {
                "start": "20200101",
                "end": "20201231",
                "universe": "csi300",
                "market": "ashare",
                "index_path": bad_index,
            },
        }),
        encoding="utf-8",
    )
    fallback_index = parent / "experiment_index.jsonl"

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)

    def fake_group(*a, **k):
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "n_candidates": 2,
        }

    def fake_lift(cands, **k):
        return [
            {
                "expression": "rank(close)",
                "lift": 0.0001, "lift_se": 0.01,
                "lift_second_half": -0.001, "baseline": 0.02, "passed": False,
            },
            {
                "expression": "rank(vol)",
                "lift": 0.008, "lift_se": 0.001,
                "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
            },
        ]

    monkeypatch.setattr(lt_mod, "run_group_lift", fake_group)
    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    monkeypatch.setattr(
        fl, "upsert_lift_admissions",
        lambda *a, **k: {"added_active": 0, "added_probation": 1, "rejected": 1},
    )

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101", "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--apply",
        "--top-m", "0",
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert fallback_index.exists()
    idx = ExperimentIndex(str(fallback_index))
    rejects = [
        r for r in idx.load()
        if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED
    ]
    # only rank(close) rejected by admission
    assert len(rejects) == 1
    assert rejects[0]["expression"] == "rank(close)"
    assert rejects[0]["lift_reason"] == "below_bar"
    assert rejects[0]["source"] == "cli_lift_test"
    assert rejects[0]["data_window"]["start"] == "20200101"
    assert rejects[0]["data_window"]["market"] == "ashare"


def test_cli_apply_group_gate_fail_writes(tmp_path, monkeypatch):
    """组门不过行（error 以 group_gate_fail 开头）也写回。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    parent = tmp_path / "ws"
    parent.mkdir()
    index_path = parent / "experiment_index.jsonl"
    run_dir = _write_session(parent, index_path=str(index_path), expr="rank(close)")
    # rewrite session under parent with good index_path
    run_dir = parent / "sess_gg"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [{
                "expression": "rank(close)",
                "reject_category": "lift_queue",
                "residual_ic_train": 0.02,
                "ic_train": 0.02,
            }],
            "candidates": [],
            "params": {
                "start": "20200101", "end": "20201231",
                "universe": "csi300", "market": "ashare",
                "index_path": str(index_path),
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)

    def fake_group(*a, **k):
        return {"lift": 0.0001, "lift_se": 0.01, "error": None, "n_candidates": 1}

    monkeypatch.setattr(lt_mod, "run_group_lift", fake_group)
    monkeypatch.setattr(
        lt_mod, "run_lift_tests",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应逐候选")),
    )
    monkeypatch.setattr(
        fl, "upsert_lift_admissions",
        lambda *a, **k: {"added_active": 0, "added_probation": 0, "rejected": 0},
    )

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101", "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--apply", "--top-m", "0",
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    rejects = [
        r for r in ExperimentIndex(str(index_path)).load()
        if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED
    ]
    assert len(rejects) == 1
    assert rejects[0]["lift_reason"] == "group_gate_fail"


def test_cli_dry_run_zero_index_write(tmp_path, monkeypatch):
    """dry-run 不写 experiment_index。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates

    parent = tmp_path / "ws"
    parent.mkdir()
    index_path = parent / "experiment_index.jsonl"
    run_dir = parent / "sess_dry"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [{
                "expression": "rank(close)",
                "reject_category": "lift_queue",
                "residual_ic_train": 0.02,
            }],
            "candidates": [],
            "params": {
                "start": "20200101", "end": "20201231",
                "market": "ashare", "index_path": str(index_path),
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_fake_daily(), None, {}),
    )
    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(
        lt_mod, "run_group_lift",
        lambda *a, **k: {"lift": 0.0001, "lift_se": 0.01, "error": None},
    )
    monkeypatch.setattr(
        fl, "upsert_lift_admissions",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry-run 不入库")),
    )

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101", "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--top-m", "0",
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert not index_path.exists()
