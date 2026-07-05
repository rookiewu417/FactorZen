"""ops runner 编排的测试:幂等续跑 / 失败告警 / 非交易日短路。

用 monkeypatch 整表替换 STAGES 为 stub,只验证编排逻辑(不重复测各 stage)。
"""
from __future__ import annotations

from datetime import date

from factorzen.ops.config import OpsConfig
from factorzen.ops.runner import run_ops_daily
from factorzen.ops.state import OpsState


class RecordNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, title: str, content: str, *, level: str = "info") -> bool:
        self.sent.append((level, title, content))
        return True


def _stub(name, executed, *, result=None, fail=False):
    def fn(cfg, as_of, ctx):
        executed.append(name)
        if fail:
            raise RuntimeError("boom")
        return result or {}

    return (name, fn)


def _cfg(tmp_path, **kw):
    base = {"session_dir": "s", "portfolio_run_dirs_glob": "g", "state_dir": str(tmp_path)}
    base.update(kw)
    return OpsConfig(**base)


def test_run_all_success_sends_daily_report(monkeypatch, tmp_path):
    executed: list[str] = []
    stages = [
        _stub("guard", executed, result={"trading_day": True}),
        _stub("data", executed, result={"daily": True}),
        _stub("report", executed, result={"summary_text": "NAV 100万"}),
    ]
    monkeypatch.setattr("factorzen.ops.runner.STAGES", stages)
    n = RecordNotifier()
    rc = run_ops_daily(_cfg(tmp_path), date(2026, 1, 20), notifier=n)
    assert rc == 0
    assert executed == ["guard", "data", "report"]
    assert any(lvl == "info" and "NAV 100万" in content for lvl, _, content in n.sent)


def test_run_stage_failure_returns_1_and_alerts(monkeypatch, tmp_path):
    executed: list[str] = []
    stages = [
        _stub("guard", executed, result={"trading_day": True}),
        _stub("data", executed, fail=True),
        _stub("report", executed, result={"summary_text": "x"}),
    ]
    monkeypatch.setattr("factorzen.ops.runner.STAGES", stages)
    n = RecordNotifier()
    rc = run_ops_daily(_cfg(tmp_path), date(2026, 1, 20), notifier=n)
    assert rc == 1
    assert executed == ["guard", "data"]  # report 未执行
    assert any(lvl == "error" for lvl, _, _ in n.sent)
    # state:data 标 failed、guard 标 done
    st = OpsState(tmp_path, date(2026, 1, 20))
    assert st.is_done("guard") is True
    assert st.is_done("data") is False


def test_run_resumes_skipping_done(monkeypatch, tmp_path):
    # 预置:guard/data 已完成
    pre = OpsState(tmp_path, date(2026, 1, 20))
    pre.mark_done("guard")
    pre.mark_done("data")
    executed: list[str] = []
    stages = [
        _stub("guard", executed, result={"trading_day": True}),
        _stub("data", executed, result={}),
        _stub("report", executed, result={"summary_text": "x"}),
    ]
    monkeypatch.setattr("factorzen.ops.runner.STAGES", stages)
    rc = run_ops_daily(_cfg(tmp_path), date(2026, 1, 20), notifier=RecordNotifier())
    assert rc == 0
    assert executed == ["report"]  # 只跑未完成阶段


def test_run_non_trading_day_short_circuits(monkeypatch, tmp_path):
    executed: list[str] = []
    stages = [
        _stub("guard", executed, result={"trading_day": False}),
        _stub("data", executed, result={}),
        _stub("report", executed, result={}),
    ]
    monkeypatch.setattr("factorzen.ops.runner.STAGES", stages)
    rc = run_ops_daily(_cfg(tmp_path), date(2026, 1, 24), notifier=RecordNotifier())
    assert rc == 0
    assert executed == ["guard"]  # data/report 未执行


def test_run_non_trading_day_not_marked_done_reruns_guard(monkeypatch, tmp_path):
    """非交易日短路后 guard 不落 done——重跑仍重新判断(不会误入 data)。"""
    executed: list[str] = []
    stages = [
        _stub("guard", executed, result={"trading_day": False}),
        _stub("data", executed, result={}),
    ]
    monkeypatch.setattr("factorzen.ops.runner.STAGES", stages)
    cfg = _cfg(tmp_path)
    run_ops_daily(cfg, date(2026, 1, 24), notifier=RecordNotifier())
    run_ops_daily(cfg, date(2026, 1, 24), notifier=RecordNotifier())
    assert executed == ["guard", "guard"]  # 两次都重跑 guard,始终不进 data
