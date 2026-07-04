import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.drivers import run_daily_step, run_replay
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


def test_daily_step_resume_nav_matches_single_shot_replay(tmp_path: Path) -> None:
    # 回归 Fix5：spec 承诺"daily 驱动跨两日 step 的 NAV == 连续两日 run_replay"。
    # 逐日"每天起一个新进程 + load_state 续跑"的 run_daily_step 序列，与一次性
    # run_replay 跑同一段历史，nav 序列必须数值相等（同 weights/同 daily/同
    # broker 逻辑，只是驱动方式不同：分次落盘续跑 vs 单进程内存循环）。
    d1, d2, d3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    daily = _daily([d1, d2, d3], "A.SZ")
    rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}

    sess_replay = tmp_path / "sess_replay"
    run_replay(
        session_dir=sess_replay,
        portfolio_run_dirs=[rd],
        daily=daily,
        initial_cash=cfg["initial_cash"],
        from_date=d1,
        to_date=d3,
        seed=0,
    )
    nav_replay = (
        SessionStore(sess_replay).nav_frame().sort("as_of_date")["nav_after"].to_list()
    )

    sess_daily = tmp_path / "sess_daily"
    SessionStore(sess_daily).init({"broker": "paper", **cfg})
    for d in (d1, d2, d3):
        # 每次调用视为独立进程：不复用 broker 实例，全靠 load_state 续跑。
        run_daily_step(sess_daily, d, [rd], daily, config=cfg)
    nav_daily = (
        SessionStore(sess_daily).nav_frame().sort("as_of_date")["nav_after"].to_list()
    )

    assert len(nav_replay) == len(nav_daily) == 3
    for replay_v, daily_v in zip(nav_replay, nav_daily, strict=True):
        assert abs(replay_v - daily_v) < 1e-6
