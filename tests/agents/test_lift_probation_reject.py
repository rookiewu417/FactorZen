"""
test_lift_probation_cap.py：auto-lift 默认 probation cap + evidence_tier / tag-legacy。
test_lift_reject_writeback.py：W3 A4/A5: lift 拒绝写回 experiment_index（session 钩子 + CLI --apply）。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.agents.team_orchestrator import _session_end_auto_lift
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIFT_QUEUE,
    REJECT_CATEGORY_LIFT_REJECTED,
)

# ==== 来自 test_lift_probation_cap.py ====
# ── helpers ──────────────────────────────────────────────────────────────────


def _lift_row(expr, *, lift, lift_se=0.0, lift_second_half=0.01,
              lift_first_half=0.01, baseline=0.04, **extra):
    d = {
        "expression": expr,
        "lift": lift,
        "lift_se": lift_se,
        "lift_first_half": lift_first_half,
        "lift_second_half": lift_second_half,
        "baseline": baseline,
    }
    d.update(extra)
    return d


def _meta(**kw):
    base = {
        "session_dir": "sess/abc",
        "run_id": "run42",
        "universe": "csi300",
        "eval_start": "20200101",
        "eval_end": "20260101",
        "horizon": 5,
        "git_sha": "deadbeef",
        "now": "2026-07-14",
    }
    base.update(kw)
    return base


def _active_lift_row(expr="rank(close)"):
    """决策为 active 的 lift 行（lift≥门槛、SE 过门、second_half>0）。"""
    return _lift_row(
        expr, lift=0.005, lift_se=0.001, lift_second_half=0.004,
        ic_train=0.02, holdout_ic=0.01,
    )


def _probation_lift_row(expr="rank(open)"):
    """决策为 probation 的 lift 行（过总量门槛但 second_half≤0）。"""
    return _lift_row(
        expr, lift=0.004, lift_se=0.001, lift_second_half=-0.001,
        ic_train=0.01,
    )


# ── 1–3：upsert_lift_admissions status cap ───────────────────────────────────


def test_active_decision_capped_to_probation_by_default(tmp_path):
    """decision=active + 默认 allow_active=False → status=probation，裁决保留。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    out = upsert_lift_admissions(
        [_active_lift_row()],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["added_probation"] == 1
    assert out.get("added_active", 0) == 0
    assert out.get("capped_active", 0) == 1

    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1
    r = lib[0]
    assert r.status == "probation"
    assert r.admission_decision == "active"
    assert r.admission_track == "lift"


def test_allow_active_true_writes_active(tmp_path):
    """decision=active + allow_active=True → status=active（现行为）。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    out = upsert_lift_admissions(
        [_active_lift_row()],
        market="ashare", root=str(tmp_path), meta=_meta(),
        allow_active=True,
    )
    assert out["added_active"] == 1
    assert out.get("capped_active", 0) == 0

    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.status == "active"
    assert r.admission_decision == "active"


def test_probation_decision_unaffected_by_cap(tmp_path):
    """decision=probation → 不受 cap 影响，admission_decision 仍为 probation。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    out = upsert_lift_admissions(
        [_probation_lift_row()],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["added_probation"] == 1
    assert out.get("capped_active", 0) == 0

    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.status == "probation"
    assert r.admission_decision == "probation"


# ── P4：已 forward-confirmed 的 lift active 不被 cap 撤销 ────────────────────


def test_confirmed_lift_active_kept_on_active_retest(tmp_path):
    """已 forward-confirmed 的 lift active + 复测 pass → 保持 active，确认字段不洗。

    默认 allow_active=False 的 cap 只限制「首次自动 active」，不得把已确认
    active 打回 probation，也不得清空 forward_confirmed_at / forward_n_days。
    """
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    confirmed_at = "2026-05-01"
    prev = FactorRecord(
        expression="rank(close)",
        market="ashare",
        status="active",
        admission_track="lift",
        admission_decision="active",
        forward_confirmed_at=confirmed_at,
        forward_n_days=60,
        lift=0.005,
        lift_se=0.001,
        lift_second_half=0.004,
        added_at="2026-01-01",
        updated_at="2026-05-01",
    )
    _save_library("ashare", [prev], root=str(tmp_path))

    out = upsert_lift_admissions(
        [_active_lift_row("rank(close)")],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(now="2026-07-14", run_id="retest_p4"),
        # 默认 allow_active=False：旧实现会 cap 到 probation 并洗确认字段
    )
    assert out.get("added_active", 0) == 1
    assert out.get("capped_active", 0) == 0
    assert out.get("added_probation", 0) == 0

    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.status == "active"
    assert r.admission_track == "lift"
    assert r.admission_decision == "active"
    assert r.forward_confirmed_at == confirmed_at
    assert r.forward_n_days == 60
    assert r.added_at == "2026-01-01"
    assert r.updated_at == "2026-07-14"


def test_confirmed_lift_active_demotes_on_probation_decision(tmp_path):
    """已 forward-confirmed 的 lift active + 复测统计降级 → 允许降到 probation。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    prev = FactorRecord(
        expression="rank(close)",
        market="ashare",
        status="active",
        admission_track="lift",
        admission_decision="active",
        forward_confirmed_at="2026-05-01",
        forward_n_days=60,
        lift=0.005,
        lift_se=0.001,
        lift_second_half=0.004,
        added_at="2026-01-01",
        updated_at="2026-05-01",
    )
    _save_library("ashare", [prev], root=str(tmp_path))

    out = upsert_lift_admissions(
        [_probation_lift_row("rank(close)")],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(now="2026-07-14"),
    )
    assert out["added_probation"] == 1
    assert out.get("added_active", 0) == 0
    assert out.get("capped_active", 0) == 0

    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.status == "probation"
    assert r.admission_decision == "probation"
    # 降级仍应保留确认 provenance（证据历史，非 status 本身）
    assert r.forward_confirmed_at == "2026-05-01"
    assert r.forward_n_days == 60



# ── 4：team hook 默认 cap ────────────────────────────────────────────────────


def test_team_hook_default_cap_writes_probation(monkeypatch, tmp_path):
    """team hook 不传 allow_active：组门过 + active 裁决 → 库内 status=probation。

    参照 test_team_lift_hook 的 group pass 全链 mock，但 upsert 走真实默认 cap。
    """
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.agents.team_orchestrator import _session_end_auto_lift
    from factorzen.discovery.factor_library import load_library
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    expr = "ts_mean(close, 5)"
    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression=expr,
        compile_ok=True, ic_train=0.02, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=1.0, n_train=100,
        residual_ic_train=0.01,
        reject_category=REJECT_CATEGORY_LIFT_QUEUE,
        reject_reason="x(lift队列,待组合裁决)",
    ))
    state.n_gray_zone = 1

    # mock daily + 覆盖充足的 materialize（对齐 test_team_lift_hook）
    rng = np.random.default_rng(1)
    n_days, n_stocks = 120, 40
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    daily = pl.DataFrame(rows)
    cut = days[int(len(days) * 0.8)]
    holdout = daily.filter(pl.col("trade_date") >= cut)

    def _panel(n_dates: int, start=cut):
        ds, dd = [], start
        while len(ds) < n_dates:
            if dd.weekday() < 5:
                ds.append(dd)
            dd += dt.timedelta(days=1)
        panel_rows = []
        for c in codes[:10]:
            for day in ds:
                panel_rows.append({
                    "trade_date": day, "ts_code": c,
                    "factor_value": float(hash((c, day)) % 100) / 100.0,
                })
        return pl.DataFrame(panel_rows)

    def mat(e):
        return _panel(80)

    def fake_group(*a, **k):
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "n_candidates": 1, "expressions": [expr],
        }

    def fake_per(queue, **k):
        return [{
            "expression": expr,
            "lift": 0.008, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
            "ic_train": 0.02, "holdout_ic": 0.01,
        }]

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    # 不 mock upsert：走真实默认 cap 路径

    class _FakeCtx:
        leaf_map = None
        horizon = 5

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root=str(tmp_path), seed=1,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        horizon=1,
    )
    # 默认 cap：统计决策 active 但计数进 probation
    assert meta.get("lift_error") is None, meta.get("lift_error")
    assert meta["lift_admissions"]["added_probation"] == 1
    assert meta["lift_admissions"].get("added_active", 0) == 0

    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1
    assert lib[0].expression == expr
    assert lib[0].status == "probation"
    assert lib[0].admission_decision == "active"


# ── 5：CLI --allow-active 透传 ───────────────────────────────────────────────


def test_cli_allow_active_forwarded(tmp_path, monkeypatch):
    """--apply --allow-active 时 upsert 收到 allow_active=True。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser

    run_dir = tmp_path / "session1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [{
                "expression": "rank(close)",
                "reject_category": "gray_zone",
                "residual_ic_train": 0.02,  # ≥ DEFAULT_GRAY_IC_FLOOR（避开 sub-floor 防呆）
                "n_residual_holdout_days": 100,
            }],
            "candidates": [],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli_main,
        "_prepare_agent_mining_data",
        lambda args: (
            pl.DataFrame({
                "trade_date": [dt.date(2020, 1, 2)],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
                "close_adj": [10.0],
            }),
            None,
            {},
        ),
    )
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates
    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(
        lt_mod,
        "run_lift_tests",
        lambda gray, **kw: [{
            "expression": "rank(close)",
            "lift": 0.005, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
        }],
    )
    upsert_calls: list = []

    def fake_upsert(results, **kw):
        upsert_calls.append({"results": results, **kw})
        return {"added_active": 1, "added_probation": 0, "rejected": 0}

    monkeypatch.setattr(fl, "upsert_lift_admissions", fake_upsert)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--apply",
        "--allow-active",
    ])
    assert args.allow_active is True
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(upsert_calls) == 1
    assert upsert_calls[0]["allow_active"] is True


def test_cli_apply_default_allow_active_false(tmp_path, monkeypatch):
    """--apply 默认不传 active 权限 → allow_active=False。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser

    run_dir = tmp_path / "session1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [{
                "expression": "rank(close)",
                "reject_category": "gray_zone",
                "residual_ic_train": 0.02,  # ≥ DEFAULT_GRAY_IC_FLOOR（避开 sub-floor 防呆）
            }],
            "candidates": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_main,
        "_prepare_agent_mining_data",
        lambda args: (
            pl.DataFrame({
                "trade_date": [dt.date(2020, 1, 2)],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
                "close_adj": [10.0],
            }),
            None,
            {},
        ),
    )
    from tests._cli_lift_mocks import patch_cli_lift_pre_gates
    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(
        lt_mod,
        "run_lift_tests",
        lambda gray, **kw: [{
            "expression": "rank(close)",
            "lift": 0.005, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
        }],
    )
    upsert_calls: list = []

    def fake_upsert(results, **kw):
        upsert_calls.append(kw)
        return {"added_active": 0, "added_probation": 1, "rejected": 0}

    monkeypatch.setattr(fl, "upsert_lift_admissions", fake_upsert)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--apply",
    ])
    assert getattr(args, "allow_active", False) is False
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert len(upsert_calls) == 1
    assert upsert_calls[0].get("allow_active") is False


# ── 6：tag-legacy ────────────────────────────────────────────────────────────


def test_tag_legacy_marks_none_only_and_idempotent(tmp_path):
    """None×2 + v2×1 → 标 2 条 legacy；v2 不动；重跑 0 条；不改 status。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        tag_legacy_records,
    )

    recs = [
        FactorRecord(
            expression="rank(close)", market="ashare", status="active",
            evidence_tier=None, ic_train=0.05,
        ),
        FactorRecord(
            expression="rank(open)", market="ashare", status="active",
            evidence_tier=None, ic_train=0.04,
        ),
        FactorRecord(
            expression="rank(vol)", market="ashare", status="active",
            evidence_tier="v2", ic_train=0.03,
        ),
    ]
    _save_library("ashare", recs, root=str(tmp_path))

    out = tag_legacy_records("ashare", root=str(tmp_path))
    assert out["tagged"] == 2

    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert lib["rank(close)"].evidence_tier == "legacy"
    assert lib["rank(open)"].evidence_tier == "legacy"
    assert lib["rank(vol)"].evidence_tier == "v2"
    # 不改 status
    assert all(r.status == "active" for r in lib.values())

    out2 = tag_legacy_records("ashare", root=str(tmp_path))
    assert out2["tagged"] == 0


def test_cli_tag_legacy(tmp_path, capsys):
    """CLI tag-legacy 子命令可解析并打印计数。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    lib_root = tmp_path / "lib"
    _save_library("ashare", [
        FactorRecord(expression="rank(close)", market="ashare", status="active"),
        FactorRecord(
            expression="rank(open)", market="ashare", status="active",
            evidence_tier="v2",
        ),
    ], root=str(lib_root))

    args = build_parser().parse_args([
        "factor-library", "tag-legacy",
        "--market", "ashare",
        "--root", str(lib_root),
    ])
    rc = cli_main._cmd_factor_library_tag_legacy(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "1" in out  # tagged 1
    assert "legacy" in out.lower() or "tag" in out.lower() or "标" in out


# ── 7：新写入 evidence_tier=v2 ───────────────────────────────────────────────


def test_upsert_sets_evidence_tier_v2(tmp_path):
    """single 轨 upsert 新记录 evidence_tier=='v2'。"""
    from factorzen.discovery.factor_library import load_library, upsert

    upsert(
        "ashare",
        [{"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
          "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100}],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        run_id="r", session_dir="s", git_sha="a", now="2026-07-14",
        root=str(tmp_path),
    )
    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.evidence_tier == "v2"


def test_upsert_lift_sets_evidence_tier_v2(tmp_path):
    """lift 轨 upsert_lift_admissions 新记录 evidence_tier=='v2'。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    upsert_lift_admissions(
        [_active_lift_row()],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.evidence_tier == "v2"


def test_old_jsonl_missing_tier_and_decision_loads(tmp_path):
    """旧 jsonl 缺 admission_decision / evidence_tier → None。"""
    from factorzen.discovery.factor_library import load_library

    path = Path(tmp_path) / "ashare.jsonl"
    old = {
        "expression": "rank(close)",
        "market": "ashare",
        "status": "active",
        "ic_train": 0.05,
    }
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.admission_decision is None
    assert r.evidence_tier is None

# ==== 来自 test_lift_reject_writeback.py ====
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
