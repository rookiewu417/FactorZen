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

def test_init_preserves_existing_manifest_config(tmp_path: Path):
    """已有会话再 init（如 fz live replay 复用 session）不应覆盖原 config——
    否则 fz live init 设的 slippage_bps/initial_cash 被 replay 的默认值清掉。"""
    import json

    s = SessionStore(tmp_path / "sess")
    s.init({"broker": "paper", "initial_cash": 2_000_000.0, "slippage_bps": 5.0})
    # 模拟 replay 用默认 config 再 init 同一 session
    s.init({"broker": "paper", "initial_cash": 1_000_000.0})
    cfg = json.loads((tmp_path / "sess" / "manifest.json").read_text())["config"]
    assert cfg["slippage_bps"] == 5.0, "已有会话的 slippage_bps 不应被覆盖"
    assert cfg["initial_cash"] == 2_000_000.0
