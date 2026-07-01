import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.drivers import run_daily_step
from factorzen.execution.store import SessionStore


def _pf(dir_, sig, code, w):
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(
        json.dumps({"signal_date": sig.isoformat(), "status": "optimal"})
    )
    return str(dir_)


def _daily(dates, code):
    return pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": code,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            }
            for d in dates
        ]
    )


def test_daily_step_idempotent_and_resumes(tmp_path: Path):
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    daily = _daily([d1, d2], "A.SZ")
    rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    s = SessionStore(tmp_path / "sess")
    s.init({"broker": "paper", **cfg})
    r1 = run_daily_step(tmp_path / "sess", d1, [rd], daily, config=cfg)
    assert not r1["skipped"] and r1["n_fills"] >= 1
    # 幂等：同日再跑跳过
    r1b = run_daily_step(tmp_path / "sess", d1, [rd], daily, config=cfg)
    assert r1b["skipped"]
    # resume：新进程语义——第二天 step 从磁盘 load_state 续跑
    r2 = run_daily_step(tmp_path / "sess", d2, [rd], daily, config=cfg)
    assert not r2["skipped"]
    assert SessionStore(tmp_path / "sess").nav_frame().height == 2  # 只两行(d1,d2)


def test_state_json_is_resumable_shape(tmp_path: Path):
    d1 = date(2026, 1, 5)
    daily = _daily([d1], "A.SZ")
    rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    s = SessionStore(tmp_path / "sess")
    s.init({"broker": "paper", **cfg})
    run_daily_step(tmp_path / "sess", d1, [rd], daily, config=cfg)
    st = SessionStore(tmp_path / "sess").load_state()
    assert set(st) == {"cash", "pos", "order_seq"}  # 可续跑态(非显示视图)
