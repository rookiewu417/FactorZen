"""F3：LiftEvalContext 三消费方接通（team hook / CLI lift-test / rebuild 复审）。

全部 mock 离线；不碰真实数据湖。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

# ── helpers ──────────────────────────────────────────────────────────────────


def _as_ymd(v) -> str | None:
    """统一成 YYYYMMDD 字符串，便于与 holdout 边界比较。"""
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y%m%d")
    s = str(v).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:4] + s[5:7] + s[8:10]
    return s.replace("-", "")[:8]


def _mock_daily(n_days: int = 120, n_stocks: int = 8, seed: int = 1) -> pl.DataFrame:
    import datetime as dt

    import numpy as np

    rng = np.random.default_rng(seed)
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
    return pl.DataFrame(rows)


def _panel(n_dates: int, n_stocks: int = 10, start=date(2022, 1, 3)):
    days, d = [], start
    while len(days) < n_dates:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for dd in days:
            rows.append({
                "trade_date": dd, "ts_code": c,
                "factor_value": float(hash((c, dd)) % 100) / 100.0,
            })
    return pl.DataFrame(rows)


def _holdout_and_mat(n_days: int = 120):
    daily = _mock_daily(n_days=n_days)
    dates = daily["trade_date"].unique().sort()
    cut = dates[int(len(dates) * 0.8)]
    holdout = daily.filter(pl.col("trade_date") >= cut)

    def mat(expr):
        return _panel(80, start=cut)

    return daily, holdout, mat


def _state_with_lift_queue(exprs: list[str]):
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    st = AgentState(seed=1)
    for e in exprs:
        st.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=e,
            compile_ok=True, ic_train=0.02, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=1.0, n_train=100,
            residual_ic_train=0.01,
            reject_category=REJECT_CATEGORY_LIFT_QUEUE,
            reject_reason="x(lift队列)",
        ))
    st.n_gray_zone = len(exprs)
    return st


class _FakeCtx:
    leaf_map = None


def _write_gray_session(tmp_path: Path, *, holdout_start: str | None = None) -> Path:
    run_dir = tmp_path / "session1"
    run_dir.mkdir(parents=True, exist_ok=True)
    man: dict = {
        "attempts": [
            {
                "expression": "rank(close)",
                "reject_category": "gray_zone",
                "residual_ic_train": 0.006,
                "n_residual_holdout_days": 100,
            },
        ],
        "candidates": [],
        "params": {"start": "20200101", "end": "20201231", "holdout_ratio": 0.2},
    }
    if holdout_start is not None:
        man["holdout_start"] = holdout_start
    (run_dir / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False), encoding="utf-8",
    )
    return run_dir


def _fake_daily_full() -> pl.DataFrame:
    """可过 _preprocess_daily 的最小帧（含 vol）。"""
    return pl.DataFrame({
        "trade_date": [date(2020, 1, 2), date(2020, 1, 3)],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "close": [10.0, 10.5],
        "open": [9.9, 10.4],
        "high": [10.2, 10.6],
        "low": [9.8, 10.3],
        "vol": [1e6, 1.1e6],
        "amount": [1e7, 1.1e7],
        "close_adj": [10.0, 10.5],
    })


# ── 1. team hook 窗口落盘 ────────────────────────────────────────────────────


def test_team_hook_admission_window_in_ctx_and_meta(monkeypatch):
    """组测收到的 ctx.admission_start == holdout 首日；meta 含 admission_start/end/horizon。"""
    from factorzen.agents.team_orchestrator import _session_end_auto_lift
    from factorzen.discovery.lift_test import DEFAULT_HORIZON

    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily, holdout, mat = _holdout_and_mat()
    captured: dict = {}

    def fake_group(*a, **k):
        captured["group_kw"] = k
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "n_candidates": 1, "expressions": ["ts_mean(close, 5)"],
        }

    def fake_per(*a, **k):
        captured["per_kw"] = k
        return [{
            "expression": "ts_mean(close, 5)",
            "lift": 0.008, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
        }]

    def fake_upsert(rows, *, market, **kw):
        captured["upsert_meta"] = kw.get("meta") or {}
        return {"added_active": 1, "added_probation": 0, "rejected": 0, "errors": []}

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        fake_upsert, raising=False,
    )

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp/lib", seed=1,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        ret_df=_panel(100).rename({"factor_value": "ret"}),
        horizon=DEFAULT_HORIZON,
    )

    hs = holdout["trade_date"].min()
    he = holdout["trade_date"].max()
    gctx = captured["group_kw"]["ctx"]
    assert gctx is not None
    assert _as_ymd(gctx.admission_start) == _as_ymd(hs)
    assert _as_ymd(gctx.admission_end) == _as_ymd(he)
    assert gctx.horizon == DEFAULT_HORIZON

    pctx = captured["per_kw"]["ctx"]
    assert _as_ymd(pctx.admission_start) == _as_ymd(hs)
    assert pctx.horizon == DEFAULT_HORIZON

    assert _as_ymd(meta.get("admission_start")) == _as_ymd(hs)
    assert _as_ymd(meta.get("admission_end")) == _as_ymd(he)
    assert meta.get("horizon") == DEFAULT_HORIZON

    um = captured["upsert_meta"]
    assert um.get("horizon") == DEFAULT_HORIZON


# ── 2. hook 注入优先 ─────────────────────────────────────────────────────────


def test_team_hook_injected_materializer_skips_default(monkeypatch):
    """显式 materialize_candidate 注入时不构造默认 / prepped materializer。"""
    from factorzen.agents.team_orchestrator import _session_end_auto_lift

    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily, holdout, mat = _holdout_and_mat()
    calls: list[str] = []

    def boom_default(*a, **k):
        calls.append("default")
        raise AssertionError("不应构造 _default_materializer")

    def boom_prepped(*a, **k):
        calls.append("prepped")
        raise AssertionError("注入路径不应构造 _materializer_from_prepped")

    monkeypatch.setattr(
        "factorzen.discovery.lift_test._default_materializer", boom_default,
    )
    monkeypatch.setattr(
        "factorzen.discovery.lift_test._materializer_from_prepped", boom_prepped,
    )

    def fake_group(*a, **k):
        return {
            "lift": 0.0, "lift_se": 0.1, "error": None,
            "expressions": ["ts_mean(close, 5)"],
        }

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp/lib", seed=1,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        horizon=1,
    )
    assert calls == [], f"注入 materialize 时不应构造默认 mat，got {calls}"
    assert meta.get("lift_error") is None


# ── 3. CLI rebuild 接通（rec.horizon 进 ctx）──────────────────────────────────


def test_cli_rebuild_wires_daily_and_record_horizon(tmp_path, monkeypatch):
    """库内 lift 记录 horizon=10 → 默认 runner 调 run_lift_tests 时 ctx.horizon==10。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    import factorzen.discovery.lift_test as lt_mod
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    _save_library(
        "ashare",
        [
            FactorRecord(
                expression="rank(vol)", market="ashare", status="active",
                admission_track="lift", ic_train=0.01, holdout_ic=0.0,
                lift=0.01, lift_se=0.001, lift_second_half=0.005,
                horizon=10,  # 与全局默认 5 不同
                added_at="2026-07-02", updated_at="2026-07-02",
            ),
        ],
        root=str(lib_root),
    )

    captured: list[dict] = []

    def fake_lift(cands, **kw):
        captured.append(kw)
        expr = cands[0]["expression"] if cands else "x"
        return [{
            "expression": expr,
            "lift": 0.006, "lift_se": 0.001,
            "lift_second_half": 0.003, "baseline": 0.05, "passed": True,
        }]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)
    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_fake_daily_full(), None, {}),
    )
    monkeypatch.setattr(fl, "collect_source_expressions", lambda market: [])
    monkeypatch.setattr(
        fl, "build_library_evaluator",
        lambda *a, **k: (lambda exprs: [], None),
    )
    # rebuild 落库根指向 tmp
    orig_rebuild = fl.rebuild

    def rebuild_to_tmp(*a, **kw):
        kw.setdefault("root", str(lib_root))
        return orig_rebuild(*a, **kw)

    monkeypatch.setattr(fl, "rebuild", rebuild_to_tmp)

    args = build_parser().parse_args([
        "factor-library", "rebuild",
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
    ])
    # 强制 root：rebuild 默认 DEFAULT_ROOT；上面 monkeypatch 已 setdefault root
    # 但 CLI 未传 root——依赖 monkeypatch 包装
    rc = cli_main._cmd_factor_library_rebuild(args)
    assert rc == 0, "lift_review_error 应为 None 且 exit 0"
    assert captured, "默认 runner 应调用 run_lift_tests"
    ctx = captured[0].get("ctx")
    assert ctx is not None, "应传 ctx"
    assert ctx.horizon == 10, f"复审 horizon 应取 rec.horizon=10，got {ctx.horizon}"


# ── 4. CLI rebuild 缺 daily 回归 ─────────────────────────────────────────────


def test_cli_rebuild_missing_daily_still_errors(monkeypatch, capsys):
    """装配返回 None → 现有空帧报错路径不变（exit 1）。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (None, None, {}),
    )
    args = build_parser().parse_args([
        "factor-library", "rebuild",
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
    ])
    rc = cli_main._cmd_factor_library_rebuild(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "挖掘帧为空" in err


# ── 5. lift-test 窗口推导 ────────────────────────────────────────────────────


def _patch_lift_test_capture(monkeypatch, captured: list):
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_fake_daily_full(), None, {}),
    )

    def fake_lift(gray, **kw):
        captured.append(kw)
        return [{
            "expression": "rank(close)",
            "lift": 0.005, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
            "candidate_rank_ic": 0.025,
        }]

    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)


def test_lift_test_admission_from_manifest(tmp_path, monkeypatch):
    """manifest 含 holdout_start → run_lift_tests 收到推导出的 admission_start。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path, holdout_start="20200901")
    captured: list = []
    _patch_lift_test_capture(monkeypatch, captured)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    assert captured
    ctx = captured[0].get("ctx")
    assert ctx is not None
    assert _as_ymd(ctx.admission_start) == "20200901"


def test_lift_test_admission_flag_overrides_manifest(tmp_path, monkeypatch):
    """--admission-start 旗标覆盖 manifest 推导。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path, holdout_start="20200901")
    captured: list = []
    _patch_lift_test_capture(monkeypatch, captured)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--admission-start", "20201015",
        "--admission-end", "20201130",
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    ctx = captured[0]["ctx"]
    assert _as_ymd(ctx.admission_start) == "20201015"
    assert _as_ymd(ctx.admission_end) == "20201130"


def test_lift_test_no_window_warns(tmp_path, monkeypatch, capsys):
    """旗标与 manifest 皆无 holdout → admission_start=None + stderr 警告。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path, holdout_start=None)
    captured: list = []
    _patch_lift_test_capture(monkeypatch, captured)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    ctx = captured[0]["ctx"]
    assert ctx.admission_start is None
    err = capsys.readouterr().err
    assert "未裁剪到 holdout" in err or "无独立性保证" in err
