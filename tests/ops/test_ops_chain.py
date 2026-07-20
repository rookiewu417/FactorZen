"""
test_ops_config.py：无人值守运营配置模型 OpsConfig 的测试
test_ops_state.py：ops 阶段级幂等状态 OpsState 的测试(原子落盘,重入跳过已完成阶段)
test_ops_stages.py：ops 六个内置阶段的测试
test_ops_runner.py：ops runner 编排的测试:幂等续跑 / 失败告警 / 非交易日短路
test_ops_notify.py：ops 通知层 Notifier 的测试(零依赖 webhook,失败不炸主链路)
test_ops_publish.py：track record 发布阶段 stage_publish 的测试(渲染净值页 + 接入 runner)
"""

from __future__ import annotations

import io
import json
import subprocess as _sp
import urllib.error
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

from factorzen.execution.store import SessionStore
from factorzen.ops.config import OpsConfig, load_ops_config
from factorzen.ops.notify import (
    StdoutNotifier,
    WebhookNotifier,
    build_notifier,
)
from factorzen.ops.runner import STAGES, run_ops_daily
from factorzen.ops.stages import (
    OpsStageError,
    stage_audit,
    stage_data,
    stage_guard,
    stage_intraday_features,
    stage_live_step,
    stage_publish,
    stage_report,
    stage_signal,
)
from factorzen.ops.state import OpsState


# ==== 来自 test_ops_config.py ====
def test_load_ops_config_roundtrip(tmp_path):
    """从 YAML 读取显式字段 + 未写字段取默认值。"""
    p = tmp_path / "ops.yaml"
    p.write_text(
        "session_dir: workspace/execution/prod-001\n"
        "portfolio_run_dirs_glob: 'workspace/portfolios/prod-*'\n"
        "lookback_days: 60\n",
        encoding="utf-8",
    )
    cfg = load_ops_config(p)
    assert cfg.session_dir == "workspace/execution/prod-001"
    assert cfg.portfolio_run_dirs_glob == "workspace/portfolios/prod-*"
    assert cfg.lookback_days == 60
    # 未写字段取默认值
    assert cfg.audit_fail_on == "error"
    assert cfg.benchmark == "000300.SH"
    assert cfg.initial_cash == 1_000_000.0
    assert cfg.notify_kind == "stdout"
    assert cfg.signal_command is None
    assert cfg.audit_types == ["daily", "daily_basic"]

def test_load_ops_config_accepts_str_path(tmp_path):
    """load_ops_config 接受 str 路径(非仅 Path)。"""
    p = tmp_path / "ops.yaml"
    p.write_text("session_dir: s\nportfolio_run_dirs_glob: g\n", encoding="utf-8")
    cfg = load_ops_config(str(p))
    assert cfg.session_dir == "s"

def test_load_ops_config_rejects_bad_fail_on(tmp_path):
    """audit_fail_on 只接受 error/warning,非法值报错。"""
    p = tmp_path / "ops.yaml"
    p.write_text(
        "session_dir: s\nportfolio_run_dirs_glob: g\naudit_fail_on: nonsense\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_ops_config(p)

def test_load_ops_config_rejects_unknown_field(tmp_path):
    """extra='forbid':未知字段(拼写错误)必须报错,而非静默忽略。"""
    p = tmp_path / "ops.yaml"
    p.write_text(
        "session_dir: s\nportfolio_run_dirs_glob: g\nlookback_dayz: 30\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_ops_config(p)

def test_load_ops_config_missing_required(tmp_path):
    """缺必填字段 session_dir 报错。"""
    p = tmp_path / "ops.yaml"
    p.write_text("portfolio_run_dirs_glob: g\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ops_config(p)

def test_load_ops_config_missing_file_raises(tmp_path):
    """文件不存在时抛错并带路径信息。"""
    missing = tmp_path / "nope.yaml"
    with pytest.raises((FileNotFoundError, ValueError)):
        load_ops_config(missing)

@pytest.mark.parametrize("bad", [0, -1])
def test_ops_config_rejects_nonpositive_lookback_days(bad):
    """lookback_days 必须 > 0：零/负窗口无法取数，须在配置层拒绝而非跑到中途才崩。"""
    with pytest.raises(ValueError):
        OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", lookback_days=bad)

@pytest.mark.parametrize("bad", [0.0])
def test_ops_config_rejects_nonpositive_initial_cash(bad):
    """initial_cash 必须 > 0：零/负本金无法纸面执行。"""
    with pytest.raises(ValueError):
        OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", initial_cash=bad)

def test_ops_config_rejects_negative_slippage():
    """slippage_bps 必须 >= 0：负滑点无经济意义（0 允许，表示零滑点对照）。"""
    with pytest.raises(ValueError):
        OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", slippage_bps=-1.0)

def test_ops_config_accepts_zero_slippage():
    """slippage_bps=0.0 合法（零滑点对照），不应被 >=0 约束误伤。"""
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", slippage_bps=0.0)
    assert cfg.slippage_bps == 0.0

def test_ops_config_defaults_directly():
    """直接构造(仅两个必填)时全部默认值就位。"""
    from factorzen.config.settings import OPS_SITE_DIR, OPS_STATE_DIR

    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g")
    assert cfg.lookback_days == 90
    assert cfg.universe is None
    assert cfg.slippage_bps == 0.0
    assert cfg.notify_url_env == "FACTORZEN_NOTIFY_WEBHOOK"
    assert cfg.publish_enabled is False
    assert cfg.publish_site_dir == str(OPS_SITE_DIR)
    assert cfg.state_dir == str(OPS_STATE_DIR)

# ==== 来自 test_ops_state.py ====
def test_mark_done_persists(tmp_path):
    """标记 done 后 is_done 为真,且落盘——新实例重读仍为真(支持跨进程重入)。"""
    st = OpsState(tmp_path, date(2026, 1, 5))
    assert st.is_done("data") is False
    st.mark_done("data", detail="补齐 60 日")
    assert st.is_done("data") is True
    # 新实例(模拟重跑进程)从磁盘恢复
    st2 = OpsState(tmp_path, date(2026, 1, 5))
    assert st2.is_done("data") is True

def test_mark_failed_not_done(tmp_path):
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_failed("audit", detail="daily 有缺口")
    assert st.is_done("audit") is False
    st2 = OpsState(tmp_path, date(2026, 1, 5))
    assert st2.is_done("audit") is False

def test_failed_then_done_overrides(tmp_path):
    """先失败后重跑成功:done 覆盖 failed(重入修复语义)。"""
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_failed("data", detail="超时")
    st.mark_done("data", detail="重跑成功")
    assert st.is_done("data") is True

def test_summary_contains_stages(tmp_path):
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_done("data")
    st.mark_failed("audit", detail="x")
    s = st.summary()
    assert s["data"]["status"] == "done"
    assert s["audit"]["status"] == "failed"
    assert s["audit"]["detail"] == "x"

def test_different_dates_isolated(tmp_path):
    """不同交易日的状态互不干扰(各自一个 json 文件)。"""
    a = OpsState(tmp_path, date(2026, 1, 5))
    a.mark_done("data")
    b = OpsState(tmp_path, date(2026, 1, 6))
    assert b.is_done("data") is False

def test_no_tmp_residue(tmp_path):
    """原子写不留 .tmp 残留文件。"""
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_done("data")
    assert list(tmp_path.glob("*.tmp")) == []

# ==== 来自 test_ops_stages.py ====
def _cfg__ops_stages(**kw):
    base = {"session_dir": "s", "portfolio_run_dirs_glob": "g"}
    base.update(kw)
    return OpsConfig(**base)

# ── guard ─────────────────────────────────────────────
def test_stage_guard_trading_day(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.fetch_trade_cal",
        lambda s, e: pl.DataFrame({"cal_date": [s], "is_open": [1]}),
    )
    assert stage_guard(_cfg__ops_stages(), date(2026, 1, 20), {}) == {"trading_day": True}

def test_stage_guard_non_trading_day(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.fetch_trade_cal",
        lambda s, e: pl.DataFrame({"cal_date": [s], "is_open": [0]}),
    )
    assert stage_guard(_cfg__ops_stages(), date(2026, 1, 24), {}) == {"trading_day": False}

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

    out = stage_data(_cfg__ops_stages(lookback_days=10, benchmark="000300.SH"), date(2026, 1, 20), {})
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
    out = stage_audit(_cfg__ops_stages(audit_types=["daily", "daily_basic"]), date(2026, 1, 20), {})
    assert out == {"daily": "ok", "daily_basic": "ok"}

def test_stage_audit_error_raises(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.build_raw_data_audit",
        lambda **k: {"status": "error", "errors": ["缺口"], "warnings": []},
    )
    with pytest.raises(OpsStageError):
        stage_audit(_cfg__ops_stages(audit_types=["daily"]), date(2026, 1, 20), {})

def test_stage_audit_warning_respects_fail_on(monkeypatch):
    monkeypatch.setattr(
        "factorzen.ops.stages.build_raw_data_audit",
        lambda **k: {"status": "warning", "errors": [], "warnings": ["空值率高"]},
    )
    # fail_on=warning → 抛
    with pytest.raises(OpsStageError):
        stage_audit(_cfg__ops_stages(audit_types=["daily"], audit_fail_on="warning"), date(2026, 1, 20), {})
    # fail_on=error(默认)→ warning 放行
    assert stage_audit(_cfg__ops_stages(audit_types=["daily"]), date(2026, 1, 20), {}) == {"daily": "warning"}

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
    assert stage_intraday_features(_cfg__ops_stages(), date(2026, 1, 20), {}) == {"skipped": True}
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
        _cfg__ops_stages(intraday_leaves=True, intraday_freq="5min", lookback_days=10),
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
    assert stage_signal(_cfg__ops_stages(), date(2026, 1, 20), {}) == {"skipped": True}

def test_stage_signal_runs(monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        seen["kw"] = k
        return SimpleNamespace(stdout="done", stderr="", returncode=0)

    monkeypatch.setattr("factorzen.ops.stages.subprocess.run", fake_run)
    out = stage_signal(_cfg__ops_stages(signal_command=["echo", "hi"]), date(2026, 1, 20), {})
    assert out["skipped"] is False
    assert seen["cmd"] == ["echo", "hi"]
    assert seen["kw"]["check"] is True and seen["kw"]["timeout"] == 3600

def test_stage_signal_failure_raises(monkeypatch):
    def boom(cmd, **k):
        raise _sp.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr("factorzen.ops.stages.subprocess.run", boom)
    with pytest.raises(OpsStageError):
        stage_signal(_cfg__ops_stages(signal_command=["x"]), date(2026, 1, 20), {})

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

    cfg = _cfg__ops_stages(
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
    cfg = _cfg__ops_stages(session_dir=str(tmp_path / "sess"), portfolio_run_dirs_glob=str(tmp_path / "none-*"))
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
    out = stage_report(_cfg__ops_stages(session_dir=str(session)), date(2026, 1, 20), {})
    txt = out["summary_text"]
    assert "2026-01-20" in txt
    assert "1,002,000" in txt
    assert "成交 2" in txt
    assert out["n_fills"] == 2

def test_stage_report_no_record(tmp_path):
    session = tmp_path / "sess"
    SessionStore(session).init({"initial_cash": 1_000_000.0})
    out = stage_report(_cfg__ops_stages(session_dir=str(session)), date(2026, 1, 20), {})
    assert out["n_fills"] == 0
    assert "无执行记录" in out["summary_text"]

# ==== 来自 test_ops_runner.py ====
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

def _cfg__ops_runner(tmp_path, **kw):
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
    rc = run_ops_daily(_cfg__ops_runner(tmp_path), date(2026, 1, 20), notifier=n)
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
    rc = run_ops_daily(_cfg__ops_runner(tmp_path), date(2026, 1, 20), notifier=n)
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
    rc = run_ops_daily(_cfg__ops_runner(tmp_path), date(2026, 1, 20), notifier=RecordNotifier())
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
    rc = run_ops_daily(_cfg__ops_runner(tmp_path), date(2026, 1, 24), notifier=RecordNotifier())
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
    cfg = _cfg__ops_runner(tmp_path)
    run_ops_daily(cfg, date(2026, 1, 24), notifier=RecordNotifier())
    run_ops_daily(cfg, date(2026, 1, 24), notifier=RecordNotifier())
    assert executed == ["guard", "guard"]  # 两次都重跑 guard,始终不进 data

# ==== 来自 test_ops_notify.py ====
def test_stdout_notifier_returns_true(capsys):
    assert StdoutNotifier().send("hi", "body", level="warn") is True
    out = capsys.readouterr().out
    assert "hi" in out and "body" in out

def test_webhook_notifier_posts_json(monkeypatch):
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["method"] = req.get_method()
        sent["body"] = json.loads(req.data.decode())
        return io.BytesIO(b"{}")

    monkeypatch.setattr("factorzen.ops.notify.urllib.request.urlopen", fake_urlopen)
    ok = WebhookNotifier("http://x/hook", retry_delay=0.0).send("t", "c", level="error")
    assert ok is True
    assert sent["url"] == "http://x/hook"
    assert sent["method"] == "POST"
    assert sent["body"] == {"title": "t", "content": "c", "level": "error"}

def test_webhook_notifier_retries_then_succeeds(monkeypatch):
    """前两次失败、第三次成功:重试机制生效,共 3 次尝试。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("down")
        return io.BytesIO(b"{}")

    monkeypatch.setattr("factorzen.ops.notify.urllib.request.urlopen", fake_urlopen)
    ok = WebhookNotifier("http://x/hook", max_retries=2, retry_delay=0.0).send("t", "c")
    assert ok is True
    assert calls["n"] == 3

def test_webhook_notifier_swallow_failure(monkeypatch):
    """全部失败:返回 False 而不抛(通知失败绝不能炸主链路)。"""

    def boom(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("factorzen.ops.notify.urllib.request.urlopen", boom)
    assert WebhookNotifier("http://x/hook", max_retries=2, retry_delay=0.0).send("t", "c") is False

def test_build_notifier_stdout():
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", notify_kind="stdout")
    assert isinstance(build_notifier(cfg), StdoutNotifier)

def test_build_notifier_webhook_with_env(monkeypatch):
    monkeypatch.setenv("FACTORZEN_NOTIFY_WEBHOOK", "http://hook")
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", notify_kind="webhook")
    n = build_notifier(cfg)
    assert isinstance(n, WebhookNotifier)
    assert n.url == "http://hook"

def test_build_notifier_webhook_missing_env_raises(monkeypatch):
    """webhook 模式但 env 缺失:启动期尽早抛 RuntimeError,而非运行时静默。"""
    monkeypatch.delenv("FACTORZEN_NOTIFY_WEBHOOK", raising=False)
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", notify_kind="webhook")
    with pytest.raises(RuntimeError):
        build_notifier(cfg)

# ==== 来自 test_ops_publish.py ====
def _cfg__ops_publish(**kw):
    base = {"session_dir": "s", "portfolio_run_dirs_glob": "g"}
    base.update(kw)
    return OpsConfig(**base)

def test_stage_publish_disabled_skips(tmp_path):
    cfg = _cfg__ops_publish(session_dir=str(tmp_path / "s"), publish_enabled=False)
    assert stage_publish(cfg, date(2026, 1, 20), {}) == {"skipped": True}

def test_stage_publish_renders_index(tmp_path):
    session = tmp_path / "s"
    store = SessionStore(session)
    store.init({"initial_cash": 1_000_000.0})
    store.append(
        {
            "as_of_date": "2026-01-19",
            "nav_before": 1_000_000.0,
            "nav_after": 1_005_000.0,
            "orders": [],
            "acks": [],
            "fills": [],
            "broker_state": {},
        }
    )
    store.append(
        {
            "as_of_date": "2026-01-20",
            "nav_before": 1_005_000.0,
            "nav_after": 1_003_000.0,
            "orders": [],
            "acks": [],
            "fills": [],
            "broker_state": {},
        }
    )
    site = tmp_path / "site"
    cfg = _cfg__ops_publish(session_dir=str(session), publish_enabled=True, publish_site_dir=str(site))
    out = stage_publish(cfg, date(2026, 1, 20), {})
    assert out["skipped"] is False
    idx = site / "index.html"
    assert idx.exists()
    html = idx.read_text(encoding="utf-8")
    # 免责声明(诚实边界)
    assert "纸面模拟" in html
    # 数据点与最新 NAV 进入页面
    assert "2026-01-20" in html
    assert "1005000" in html and "1003000" in html
    # 最大回撤:1005000→1003000 约 -0.199%
    assert "回撤" in html

def test_publish_stage_wired_into_runner_after_report():
    names = [n for n, _ in STAGES]
    assert "publish" in names
    assert names.index("publish") > names.index("report")
    # report 阶段仍在(未被替换)
    assert stage_report is not None

