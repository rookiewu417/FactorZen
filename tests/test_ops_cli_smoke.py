"""fz ops daily/status CLI 冒烟(dispatch/日期解析/返回码/状态打印)。"""
from __future__ import annotations

from datetime import date

from factorzen.cli.main import main
from factorzen.ops.state import OpsState


def _write_cfg(tmp_path, state_dir=None):
    p = tmp_path / "ops.yaml"
    sd = state_dir or (tmp_path / "state")
    p.write_text(
        f"session_dir: s\nportfolio_run_dirs_glob: g\nstate_dir: {sd}\n",
        encoding="utf-8",
    )
    return p


def test_fz_ops_daily_dispatches_with_date(monkeypatch, tmp_path):
    p = _write_cfg(tmp_path)
    seen: dict = {}

    def fake_run(cfg, as_of, notifier=None):
        seen["as_of"] = as_of
        seen["session_dir"] = cfg.session_dir
        return 0

    monkeypatch.setattr("factorzen.ops.runner.run_ops_daily", fake_run)
    rc = main(["ops", "daily", "--config", str(p), "--date", "20260720"])
    assert rc == 0
    assert seen["as_of"] == date(2026, 7, 20)
    assert seen["session_dir"] == "s"


def test_fz_ops_daily_defaults_to_today(monkeypatch, tmp_path):
    p = _write_cfg(tmp_path)
    seen: dict = {}

    def fake_run(cfg, as_of, notifier=None):
        seen["as_of"] = as_of
        return 0

    monkeypatch.setattr("factorzen.ops.runner.run_ops_daily", fake_run)
    rc = main(["ops", "daily", "--config", str(p)])
    assert rc == 0
    assert seen["as_of"] == date.today()


def test_fz_ops_daily_propagates_return_code(monkeypatch, tmp_path):
    p = _write_cfg(tmp_path)
    monkeypatch.setattr(
        "factorzen.ops.runner.run_ops_daily", lambda cfg, as_of, notifier=None: 1
    )
    rc = main(["ops", "daily", "--config", str(p), "--date", "20260720"])
    assert rc == 1


def test_fz_ops_status_prints_summary(tmp_path, capsys):
    sd = tmp_path / "state"
    p = _write_cfg(tmp_path, state_dir=sd)
    OpsState(sd, date(2026, 7, 20)).mark_done("guard", detail="ok")
    rc = main(["ops", "status", "--config", str(p), "--date", "20260720"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "guard" in out and "done" in out
