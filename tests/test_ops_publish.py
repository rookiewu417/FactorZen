"""track record 发布阶段 stage_publish 的测试(渲染净值页 + 接入 runner)。"""
from __future__ import annotations

from datetime import date

from factorzen.execution.store import SessionStore
from factorzen.ops.config import OpsConfig
from factorzen.ops.runner import STAGES
from factorzen.ops.stages import stage_publish, stage_report


def _cfg(**kw):
    base = {"session_dir": "s", "portfolio_run_dirs_glob": "g"}
    base.update(kw)
    return OpsConfig(**base)


def test_stage_publish_disabled_skips(tmp_path):
    cfg = _cfg(session_dir=str(tmp_path / "s"), publish_enabled=False)
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
    cfg = _cfg(session_dir=str(session), publish_enabled=True, publish_site_dir=str(site))
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
