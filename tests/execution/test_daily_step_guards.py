"""run_daily_step 的交易日历守卫(E3) + 日期单调性守卫(E2)。

E3：as_of 为非交易日(daily 无该日行)时 market 为空，若照常 step 会落一条纯现金塌陷 nav
   行且被 has_date 永久锁死、无法修复。须直接跳过不落盘。
E2：resume 无日期单调性守卫时，补跑早于已推进日期的 as_of 会用「未来的」broker 状态步进
   过去，ledger 乱序、state 被污染。须拒绝。
"""
import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.drivers import run_daily_step
from factorzen.execution.store import SessionStore


def _pf(dir_, sig, code, w):
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps({"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)


def _daily(dates, code):
    return pl.DataFrame([{"trade_date": d, "ts_code": code, "open": 10.0, "pre_close": 10.0,
                          "close": 10.0, "vol": 1e8, "amount": 1e9} for d in dates])


def test_non_trading_day_is_skipped_not_recorded(tmp_path: Path):
    d1, holiday = date(2026, 1, 5), date(2026, 1, 10)  # 1/10 非交易日(daily 无行)
    daily = _daily([d1], "A.SZ")
    rd = _pf(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})
    run_daily_step(sess, d1, [rd], daily, config=cfg)  # 正常执行日

    r = run_daily_step(sess, holiday, [rd], daily, config=cfg)
    assert r["skipped"], "非交易日应跳过"
    # 不应落塌陷 nav 行；ledger 只含 d1
    got = SessionStore(sess).nav_frame()["as_of_date"].to_list()
    assert holiday.isoformat() not in got, "非交易日不应落 ledger 行（否则被 has_date 永久锁死）"


def test_backwards_as_of_rejected(tmp_path: Path):
    d5, d6, d7 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    daily = _daily([d5, d6, d7], "A.SZ")
    rd = _pf(tmp_path / "pf", date(2026, 1, 2), "A.SZ", 0.5)
    cfg = {"initial_cash": 1_000_000.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})

    run_daily_step(sess, d5, [rd], daily, config=cfg)
    run_daily_step(sess, d7, [rd], daily, config=cfg)  # 已推进到 d7
    nav_before = SessionStore(sess).nav_frame()["as_of_date"].to_list()

    # 补跑 d6（早于已推进的 d7）→ 应拒绝，不污染 ledger
    r = run_daily_step(sess, d6, [rd], daily, config=cfg)
    assert r["skipped"], "早于已推进日期的补跑应被拒绝"
    nav_after = SessionStore(sess).nav_frame()["as_of_date"].to_list()
    assert nav_after == nav_before, "乱序补跑不应改动 ledger"
