"""auto-lift 默认 probation cap + evidence_tier / tag-legacy。

审查结论：lift 统计门未校准前，auto 路径最多写 probation；
admission_decision 保留原始裁决；legacy 库只打标不降级。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

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
        reject_reason="x(lift队列,覆盖待lift验)",
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
                "residual_ic_train": 0.006,
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
                "residual_ic_train": 0.006,
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
