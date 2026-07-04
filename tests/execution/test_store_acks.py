from datetime import date
from pathlib import Path

from factorzen.execution.store import SessionStore


def _rec(d, orders, acks, fills, bstate):
    return {
        "as_of_date": d.isoformat(),
        "nav_before": 1e6,
        "nav_after": 1e6,
        "broker_state": bstate,
        "orders": orders,
        "acks": acks,
        "fills": fills,
    }


def test_append_persists_acks_and_reads_back(tmp_path: Path):
    s = SessionStore(tmp_path / "sess")
    s.init({"broker": "paper", "initial_cash": 1e6})
    orders = [{"ts_code": "X.SZ", "side": "buy", "volume": 1000, "price_type": "market", "price": None}]
    acks = [{"order_id": "paper-1", "ts_code": "X.SZ", "accepted": False, "reason": "suspended"}]
    fills = []
    s.append(_rec(date(2026, 1, 5), orders, acks, fills, {"cash": 1e6, "pos": {}, "order_seq": 1}))
    recs = s.ledger_records()
    assert len(recs) == 1
    assert recs[0]["acks"][0]["reason"] == "suspended"
    assert recs[0]["orders"][0]["ts_code"] == "X.SZ"


def test_ledger_records_backward_compat_no_acks(tmp_path: Path):
    # 模拟旧 payload（无 acks）：直接构造 ledger.parquet
    import json

    import polars as pl

    d = tmp_path / "sess"
    d.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "as_of_date": "2026-01-05",
                "nav_before": 1e6,
                "nav_after": 1e6,
                "payload": json.dumps({"orders": [], "fills": []}),
            }
        ]
    ).write_parquet(d / "ledger.parquet")
    recs = SessionStore(d).ledger_records()
    assert recs[0]["acks"] == []  # 旧无 acks → 空
