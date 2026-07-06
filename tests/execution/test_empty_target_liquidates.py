"""空目标信号 = 清仓语义：applicable 但为空 → 正常 step 清仓；真·无 applicable 才跳过。"""
import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.drivers import run_replay
from factorzen.execution.store import SessionStore


def _pf(dir_, sig, weights: dict):
    dir_.mkdir(parents=True, exist_ok=True)
    codes = list(weights)
    ws = [weights[c] for c in codes]
    pl.DataFrame({"ts_code": codes, "target_weight": ws},
                 schema={"ts_code": pl.Utf8, "target_weight": pl.Float64}).write_parquet(dir_/"weights.parquet")
    (dir_/"manifest.json").write_text(json.dumps({"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)

def _daily(dates, code):
    return pl.DataFrame([{"trade_date": d, "ts_code": code, "open": 10.0, "pre_close": 10.0,
                          "close": 10.0, "vol": 1e8, "amount": 1e9} for d in dates])

def test_empty_applicable_signal_liquidates(tmp_path: Path):
    d1, d2, d3 = date(2026,1,5), date(2026,1,6), date(2026,1,7)
    daily = _daily([d1,d2,d3], "A.SZ")
    buy = _pf(tmp_path/"buy", d1, {"A.SZ": 0.9})          # d1 建仓
    empty = _pf(tmp_path/"empty", d2, {})                 # d2 空目标 → 应清仓
    run_replay(session_dir=tmp_path/"s", portfolio_run_dirs=[buy, empty], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d3, seed=0)
    store = SessionStore(tmp_path/"s")
    ledger = store.ledger_records()
    # 次日执行(s<d)：d1 建仓信号于 d2 生效建仓，d2 空目标信号于 d3 生效清仓。
    d3rec = next(r for r in ledger if r["as_of_date"] == d3.isoformat())
    sells = [o for o in d3rec["orders"] if o.get("side") == "sell" and o.get("ts_code") == "A.SZ"]
    assert sells, f"d3 空目标应有清仓卖单, 实际 orders={d3rec['orders']}"
    # d3 起（持久化的 broker 续跑态）持仓为空
    bs = store.load_state()
    held = {c: p for c, p in bs.get("pos", bs.get("positions", {})).items()
            if (p.get("volume", 0) if isinstance(p, dict) else 0) > 0}
    assert held == {}, f"空目标后应清仓, 实际 {held}"

def test_no_applicable_signal_still_skips(tmp_path: Path):
    d1, d2 = date(2026,1,5), date(2026,1,6)
    daily = _daily([d1,d2], "A.SZ")
    late = _pf(tmp_path/"late", d2, {"A.SZ": 0.9})        # 信号日 d2
    run_replay(session_dir=tmp_path/"s", portfolio_run_dirs=[late], daily=daily,
               initial_cash=1_000_000.0, from_date=d1, to_date=d1, seed=0)  # 只跑 d1
    # d1 无适用信号(信号 d2>d1) → 跳过, ledger 空
    assert SessionStore(tmp_path/"s").nav_frame().height == 0
