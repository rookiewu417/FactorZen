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


def _daily_var(dates_prices, code):
    """带价格变化的 daily（区分「续跑重建持仓」vs「从空仓重来」的 nav）。"""
    return pl.DataFrame(
        [
            {"trade_date": d, "ts_code": code, "open": p, "pre_close": p,
             "close": p, "vol": 1e8, "amount": 1e9}
            for d, p in dates_prices
        ]
    )


def test_replay_resume_extends_window_matches_single_shot(tmp_path: Path) -> None:
    """扩窗/崩溃恢复：在已有部分 ledger 的 session 上再 run_replay（延长 --to），
    必须 load_state 重建 broker，否则续跑日从空仓 initial_cash 起、nav 错。
    扩窗 replay 的 nav 序列须与一次性 replay 相同。"""
    d1, d2, d3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    daily = _daily_var([(d1, 10.0), (d2, 11.0), (d3, 12.0)], "A.SZ")  # 价格逐日上涨
    rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)

    full = tmp_path / "full"
    run_replay(session_dir=full, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d3, seed=0)
    nav_full = SessionStore(full).nav_frame().sort("as_of_date")["nav_after"].to_list()

    resume = tmp_path / "resume"
    run_replay(session_dir=resume, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d2, seed=0)
    run_replay(session_dir=resume, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d3, seed=0)  # 扩窗续跑
    nav_resume = SessionStore(resume).nav_frame().sort("as_of_date")["nav_after"].to_list()

    assert len(nav_full) == len(nav_resume) == 3
    for a, b in zip(nav_full, nav_resume, strict=True):
        assert abs(a - b) < 1e-6, f"扩窗 replay nav 须等于一次性 replay：{nav_full} vs {nav_resume}"


def test_replay_state_resumable_by_daily_step(tmp_path: Path) -> None:
    """run_replay 落的 state.json 须是可续跑态 {cash,pos,order_seq}，使后续
    run_daily_step 能 load_state 续跑，而非因显示视图 float(dict) 抛 TypeError。"""
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    daily = _daily_var([(d1, 10.0), (d2, 11.0)], "A.SZ")
    rd = _pf(tmp_path / "pf", d1, "A.SZ", 0.5)
    sess = tmp_path / "sess"
    run_replay(session_dir=sess, portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d1, seed=0)

    st = SessionStore(sess).load_state()
    assert set(st) == {"cash", "pos", "order_seq"}, f"replay 应落可续跑态，实际 {set(st)}"

    # 后续 fz live step 续跑不应因显示视图 float(dict) 崩溃
    r = run_daily_step(sess, d2, [rd], daily, config={"initial_cash": 1_000_000.0})
    assert not r["skipped"] and r["nav_after"] is not None
