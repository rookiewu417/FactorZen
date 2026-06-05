"""tools/smoke_data.py 离线单测：全 mock build_raw_data_audit 与连通性，
覆盖审计聚合、状态优先级、退出码与 argparse 行为。不调用真实 Tushare。
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

# tools/ 不是包，按文件路径加载模块
_SPEC = importlib.util.spec_from_file_location(
    "smoke_data", Path(__file__).resolve().parents[1] / "tools" / "smoke_data.py"
)
assert _SPEC and _SPEC.loader
smoke_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_data)


def _audit(status: str, rows: int = 100, warnings=None, errors=None) -> dict:
    return {
        "status": status,
        "checks": {"total_rows": rows},
        "warnings": warnings or [],
        "errors": errors or [],
    }


# ── _worst_status ───────────────────────────────────────────


def test_worst_status_error_dominates():
    assert smoke_data._worst_status(["ok", "warning", "error"]) == "error"


def test_worst_status_warning_over_ok():
    assert smoke_data._worst_status(["ok", "warning", "ok"]) == "warning"


def test_worst_status_all_ok():
    assert smoke_data._worst_status(["ok", "ok"]) == "ok"


def test_worst_status_empty_is_ok():
    assert smoke_data._worst_status([]) == "ok"


# ── run_audits ──────────────────────────────────────────────


def test_run_audits_calls_audit_per_type(monkeypatch):
    calls = []

    def fake(*, data_type, start, end, universe_codes=None):
        calls.append(data_type)
        return _audit("ok")

    monkeypatch.setattr(smoke_data, "build_raw_data_audit", fake)
    results = smoke_data.run_audits(["daily", "finance"], "20230101", "20231231")
    assert set(results) == {"daily", "finance"}
    assert calls == ["daily", "finance"]


# ── summarize → 退出码 ──────────────────────────────────────


def test_summarize_all_ok_returns_0(capsys):
    code = smoke_data.summarize((True, "ok"), {"daily": _audit("ok")})
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_summarize_warning_returns_2():
    code = smoke_data.summarize(
        (True, "ok"), {"daily": _audit("warning", warnings=["缺 3 天"])}
    )
    assert code == 2


def test_summarize_error_returns_1():
    code = smoke_data.summarize(
        (True, "ok"), {"daily": _audit("error", errors=["分区为空"])}
    )
    assert code == 1


def test_summarize_connectivity_fail_is_error():
    code = smoke_data.summarize((False, "token 缺失"), {"daily": _audit("ok")})
    assert code == 1


def test_summarize_skipped_connectivity(capsys):
    code = smoke_data.summarize(None, {"daily": _audit("ok")})
    assert code == 0
    assert "跳过" in capsys.readouterr().out


# ── check_tushare_connectivity ──────────────────────────────


def _fake_pro():
    """init_tushare 桩：需有被 _retry 引用的 trade_cal 属性。"""
    return SimpleNamespace(trade_cal=lambda **kw: None)


def test_connectivity_success(monkeypatch):
    import factorzen.core.loader as loader_mod

    monkeypatch.setattr(loader_mod, "init_tushare", _fake_pro)

    class _DF:
        empty = False

        def __len__(self):
            return 5

    monkeypatch.setattr(loader_mod, "_retry", lambda fn, **kw: _DF())
    ok, msg = smoke_data.check_tushare_connectivity()
    assert ok and "正常" in msg


def test_connectivity_empty_result(monkeypatch):
    import factorzen.core.loader as loader_mod

    monkeypatch.setattr(loader_mod, "init_tushare", _fake_pro)

    class _Empty:
        empty = True

    monkeypatch.setattr(loader_mod, "_retry", lambda fn, **kw: _Empty())
    ok, _ = smoke_data.check_tushare_connectivity()
    assert not ok


def test_connectivity_exception(monkeypatch):
    import factorzen.core.loader as loader_mod

    def _boom():
        raise RuntimeError("no token")

    monkeypatch.setattr(loader_mod, "init_tushare", _boom)
    ok, msg = smoke_data.check_tushare_connectivity()
    assert not ok and "失败" in msg


# ── main / argparse ─────────────────────────────────────────


def test_main_skip_tushare_offline(monkeypatch):
    """--skip-tushare 不触发连通性检查，退出码由审计决定。"""
    monkeypatch.setattr(
        smoke_data, "build_raw_data_audit", lambda **kw: _audit("ok")
    )

    def _should_not_call():
        raise AssertionError("--skip-tushare 时不应检查连通性")

    monkeypatch.setattr(smoke_data, "check_tushare_connectivity", _should_not_call)
    code = smoke_data.main(["--skip-tushare", "--data-type", "daily"])
    assert code == 0


def test_main_json_output(monkeypatch, capsys):
    monkeypatch.setattr(smoke_data, "build_raw_data_audit", lambda **kw: _audit("ok"))
    monkeypatch.setattr(
        smoke_data, "check_tushare_connectivity", lambda: (True, "ok")
    )
    code = smoke_data.main(["--data-type", "daily", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert '"exit_code": 0' in out


def test_main_error_audit_exit_1(monkeypatch):
    monkeypatch.setattr(
        smoke_data, "build_raw_data_audit", lambda **kw: _audit("error", errors=["空"])
    )
    code = smoke_data.main(["--skip-tushare", "--data-type", "finance"])
    assert code == 1


def test_main_default_audits_all_three(monkeypatch):
    seen = []
    monkeypatch.setattr(
        smoke_data,
        "build_raw_data_audit",
        lambda **kw: seen.append(kw["data_type"]) or _audit("ok"),
    )
    smoke_data.main(["--skip-tushare"])
    assert set(seen) == {"daily", "daily_basic", "finance"}
