from datetime import date
from pathlib import Path

from factorzen.execution.store import SessionStore


def _rec(d, nav, bstate):
    return {
        "as_of_date": d.isoformat(),
        "nav_before": nav,
        "nav_after": nav,
        "broker_state": bstate,
        "orders": [],
        "fills": [],
    }


def test_init_creates_manifest(tmp_path: Path):
    s = SessionStore(tmp_path / "sess1")
    s.init({"broker": "paper", "initial_cash": 1e6, "seed": 0})
    assert (tmp_path / "sess1" / "manifest.json").exists()


def test_append_and_idempotency(tmp_path: Path):
    s = SessionStore(tmp_path / "sess1")
    s.init({"broker": "paper", "initial_cash": 1e6})
    d = date(2026, 1, 5)
    assert not s.has_date(d)
    s.append(_rec(d, 1e6, {"cash": 1e6, "pos": {}, "order_seq": 0}))
    assert s.has_date(d)  # 幂等哨兵
    assert s.load_state()["cash"] == 1e6
    assert s.nav_frame().height == 1


def test_resume_reads_latest_state(tmp_path: Path):
    s = SessionStore(tmp_path / "sess1")
    s.init({"broker": "paper", "initial_cash": 1e6})
    s.append(_rec(date(2026, 1, 5), 1e6, {"cash": 9e5, "pos": {}, "order_seq": 2}))
    s2 = SessionStore(tmp_path / "sess1")  # 新实例重载
    assert s2.load_state()["cash"] == 9e5
    assert s2.has_date(date(2026, 1, 5))
