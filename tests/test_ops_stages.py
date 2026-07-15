"""ops 六个内置阶段的测试。

外部依赖(数据补齐/质量门/信号命令/执行驱动)全部 monkeypatch,
live_step/report 用 tmp 目录真实 SessionStore 验证 init 与账本消费。
"""
from __future__ import annotations

import subprocess as _sp
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

from factorzen.execution.store import SessionStore
from factorzen.ops.config import OpsConfig
from factorzen.ops.runner import STAGES
from factorzen.ops.stages import (
    OpsStageError,
    stage_audit,
    stage_data,
    stage_guard,
    stage_intraday_features,
    stage_live_step,
    stage_report,
    stage_signal,
)


def _cfg(**kw):
    base = {"session_dir": "s", "portfolio_run_dirs_glob": "g"}
    base.update(kw)
    return OpsConfig(**base)


# ── guard ─────────────────────────────────────────────
def test_stage_guard_trading_day(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.fetch_trade_cal",
        lambda s, e: pl.DataFrame({"cal_date": [s], "is_open": [1]}),
    )
    assert stage_guard(_cfg(), date(2026, 1, 20), {}) == {"trading_day": True}


def test_stage_guard_non_trading_day(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.fetch_trade_cal",
        lambda s, e: pl.DataFrame({"cal_date": [s], "is_open": [0]}),
    )
    assert stage_guard(_cfg(), date(2026, 1, 24), {}) == {"trading_day": False}


# ── data ──────────────────────────────────────────────
def test_stage_data_window_and_ok(monkeypatch):
    calls: dict = {}

    def rec(name):
        def f(*a, **k):
            calls[name] = a
            return SimpleNamespace(ok=True)

        return f

    monkeypatch.setattr("factorzen.ops.stages.ensure_daily", rec("daily"))
    monkeypatch.setattr("factorzen.ops.stages.ensure_adj_factor", rec("adj"))
    monkeypatch.setattr("factorzen.ops.stages.ensure_daily_basic", rec("basic"))
    monkeypatch.setattr("factorzen.ops.stages.ensure_index_daily", rec("index"))

    out = stage_data(_cfg(lookback_days=10, benchmark="000300.SH"), date(2026, 1, 20), {})
    assert out["daily"] is True and out["index_daily"] is True
    # 窗口 = [as_of - 10 天, as_of]
    assert calls["daily"] == ("20260110", "20260120")
    # index 首参为 benchmark
    assert calls["index"] == ("000300.SH", "20260110", "20260120")


# ── audit ─────────────────────────────────────────────
def test_stage_audit_pass(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.build_raw_data_audit",
        lambda **k: {"status": "ok", "errors": [], "warnings": []},
    )
    out = stage_audit(_cfg(audit_types=["daily", "daily_basic"]), date(2026, 1, 20), {})
    assert out == {"daily": "ok", "daily_basic": "ok"}


def test_stage_audit_error_raises(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.build_raw_data_audit",
        lambda **k: {"status": "error", "errors": ["缺口"], "warnings": []},
    )
    with pytest.raises(OpsStageError):
        stage_audit(_cfg(audit_types=["daily"]), date(2026, 1, 20), {})


def test_stage_audit_warning_respects_fail_on(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.build_raw_data_audit",
        lambda **k: {"status": "warning", "errors": [], "warnings": ["空值率高"]},
    )
    # fail_on=warning → 抛
    with pytest.raises(OpsStageError):
        stage_audit(_cfg(audit_types=["daily"], audit_fail_on="warning"), date(2026, 1, 20), {})
    # fail_on=error(默认)→ warning 放行
    assert stage_audit(_cfg(audit_types=["daily"]), date(2026, 1, 20), {}) == {"daily": "warning"}


# ── intraday_features ─────────────────────────────────
def test_stage_intraday_features_skipped_when_disabled(monkeypatch):
    """默认 intraday_leaves=False：no-op skip，不调 build。"""
    called: list[object] = []

    def fake_build(*a, **k):
        called.append((a, k))
        raise AssertionError("build_intraday_features 不应被调用")

    monkeypatch.setattr(
        "factorzen.intraday.features.engine.build_intraday_features",
        fake_build,
    )
    assert stage_intraday_features(_cfg(), date(2026, 1, 20), {}) == {"skipped": True}
    assert called == []


def test_stage_intraday_features_calls_build_with_window_and_freq(monkeypatch):
    """intraday_leaves=True：按 lookback 窗口与配置 freq 调用 build。"""
    seen: dict = {}

    def fake_build(start, end, *, freq="5min", overwrite=False, **k):
        seen.update(start=start, end=end, freq=freq, overwrite=overwrite)
        return SimpleNamespace(months=["202601"], rows=42)

    monkeypatch.setattr(
        "factorzen.intraday.features.engine.build_intraday_features",
        fake_build,
    )
    out = stage_intraday_features(
        _cfg(intraday_leaves=True, intraday_freq="5min", lookback_days=10),
        date(2026, 1, 20),
        {},
    )
    assert out == {
        "skipped": False,
        "months": ["202601"],
        "rows": 42,
        "freq": "5min",
    }
    assert seen == {
        "start": "20260110",
        "end": "20260120",
        "freq": "5min",
        "overwrite": False,
    }


def test_stages_order_intraday_features_before_signal():
    """STAGES 含 intraday_features，且位于 data/audit 之后、signal 之前。"""
    names = [n for n, _ in STAGES]
    assert "intraday_features" in names
    assert names.index("data") < names.index("intraday_features")
    assert names.index("audit") < names.index("intraday_features")
    assert names.index("intraday_features") < names.index("signal")


# ── signal ────────────────────────────────────────────
def test_stage_signal_skip_when_none():
    assert stage_signal(_cfg(), date(2026, 1, 20), {}) == {"skipped": True}


def test_stage_signal_runs(monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        seen["kw"] = k
        return SimpleNamespace(stdout="done", stderr="", returncode=0)

    monkeypatch.setattr("factorzen.ops.stages.subprocess.run", fake_run)
    out = stage_signal(_cfg(signal_command=["echo", "hi"]), date(2026, 1, 20), {})
    assert out["skipped"] is False
    assert seen["cmd"] == ["echo", "hi"]
    assert seen["kw"]["check"] is True and seen["kw"]["timeout"] == 3600


def test_stage_signal_failure_raises(monkeypatch):
    def boom(cmd, **k):
        raise _sp.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr("factorzen.ops.stages.subprocess.run", boom)
    with pytest.raises(OpsStageError):
        stage_signal(_cfg(signal_command=["x"]), date(2026, 1, 20), {})


# ── live_step ─────────────────────────────────────────
def test_stage_live_step_inits_and_passes_config(monkeypatch, tmp_path):
    session = tmp_path / "sess"
    pdir = tmp_path / "port-1"
    pdir.mkdir()
    daily_stub = pl.DataFrame({"ts_code": ["A.SZ"], "trade_date": [date(2026, 1, 20)]})
    monkeypatch.setattr("factorzen.ops.stages.fetch_daily", lambda s, e: daily_stub)

    seen: dict = {}

    def fake_step(sdir, d, run_dirs, daily, *, config):
        seen.update(config=config, run_dirs=run_dirs, as_of=d)
        return {"as_of": d.isoformat(), "nav_after": 1_000_000.0, "n_fills": 0, "skipped": False}

    monkeypatch.setattr("factorzen.ops.stages.run_daily_step", fake_step)

    cfg = _cfg(
        session_dir=str(session),
        portfolio_run_dirs_glob=str(tmp_path / "port-*"),
        initial_cash=500_000.0,
        slippage_bps=5.0,
    )
    out = stage_live_step(cfg, date(2026, 1, 20), {})
    assert out["n_fills"] == 0
    assert seen["config"] == {"initial_cash": 500_000.0, "slippage_bps": 5.0}
    assert seen["run_dirs"] == [str(pdir)]
    assert (session / "manifest.json").exists()  # 首次自动 init


def test_stage_live_step_empty_glob_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "factorzen.ops.stages.fetch_daily", lambda s, e: pl.DataFrame({"ts_code": ["A.SZ"]})
    )
    cfg = _cfg(session_dir=str(tmp_path / "sess"), portfolio_run_dirs_glob=str(tmp_path / "none-*"))
    with pytest.raises(OpsStageError):
        stage_live_step(cfg, date(2026, 1, 20), {})


# ── report ────────────────────────────────────────────
def test_stage_report_summarizes_ledger(tmp_path):
    session = tmp_path / "sess"
    store = SessionStore(session)
    store.init({"initial_cash": 1_000_000.0})
    store.append(
        {
            "as_of_date": "2026-01-20",
            "nav_before": 1_000_000.0,
            "nav_after": 1_002_000.0,
            "orders": [{"x": 1}],
            "acks": [],
            "fills": [{"a": 1}, {"b": 2}],
            "broker_state": {},
        }
    )
    out = stage_report(_cfg(session_dir=str(session)), date(2026, 1, 20), {})
    txt = out["summary_text"]
    assert "2026-01-20" in txt
    assert "1,002,000" in txt
    assert "成交 2" in txt
    assert out["n_fills"] == 2


def test_stage_report_no_record(tmp_path):
    session = tmp_path / "sess"
    SessionStore(session).init({"initial_cash": 1_000_000.0})
    out = stage_report(_cfg(session_dir=str(session)), date(2026, 1, 20), {})
    assert out["n_fills"] == 0
    assert "无执行记录" in out["summary_text"]
