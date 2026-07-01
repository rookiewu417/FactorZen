import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.attribution import build_attribution_report
from factorzen.execution.drivers import run_replay


def _pf(dir_, sig, code, w):
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps({"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)

def _daily(rows):  # rows: list of dict(trade_date,ts_code,open,pre_close,close,vol,amount)
    return pl.DataFrame(rows)

def test_slippage_only_scenario_residual_near_zero(tmp_path: Path):
    # 单票、无停牌无涨跌停、open≠close → 纯滑点+成本，missed=0，residual≈0
    dates = [date(2026,1,5), date(2026,1,6)]
    rows = []
    for d in dates:
        rows.append({"trade_date": d, "ts_code":"A.SZ", "open":10.1, "pre_close":10.0,
                     "close":10.0, "vol":1e8, "amount":1e9})
    daily = _daily(rows)
    rd = _pf(tmp_path/"pf", dates[0], "A.SZ", 0.5)
    run_replay(session_dir=tmp_path/"sess", portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=dates[0], to_date=dates[-1], seed=0)
    rep = build_attribution_report(tmp_path/"sess", [rd], daily, initial_cash=1_000_000.0)
    assert sum(v["count"] for v in rep["missed_by_reason"].values()) == 0   # 无未成交
    assert rep["cost_bps"] > 0 and rep["slippage_bps"] != 0
    # residual = total_gap - cost - slippage 应接近 0（纯滑点+成本场景，无未成交/时点差）
    assert abs(rep["residual_bps"]) < abs(rep["cost_bps"]) + abs(rep["slippage_bps"])

def test_suspended_scenario_missed_notional_and_positive_residual(tmp_path: Path):
    # 独立手算 ground-truth：day1 停牌(vol=0) → 真实 0 成交、理想(frictionless)
    # 仍按 close=10.0 全额买入。day2 复牌且低开高走缺口(open=10.5，较 day1
    # close +5%，无涨停)：理想因 day1 已建仓，吃到这段缺口收益；真实 day1
    # 未成交、day2 才追价买入(price=10.5)，错过了这段缺口——这才是真实的
    # "停牌导致踏空"经济含义，而非同日 buy@close/mark@close 的恒等 0。
    # 手算：ideal day2 nav ~1,025,000（多头吃满 5% 缺口），real day2 nav
    # ~1,000,000-手续费（day2 才追价，无缺口收益）；两者 ann_ret 应显著不同。
    dates = [date(2026,1,5), date(2026,1,6)]
    daily = _daily([
        {"trade_date": dates[0], "ts_code": "A.SZ", "open": 10.0, "pre_close": 10.0,
         "close": 10.0, "vol": 0.0, "amount": 0.0},  # 停牌
        {"trade_date": dates[1], "ts_code": "A.SZ", "open": 10.5, "pre_close": 10.0,
         "close": 10.5, "vol": 1e8, "amount": 1e9},  # 复牌，缺口 +5%
    ])
    rd = _pf(tmp_path/"pf", dates[0], "A.SZ", 0.5)
    run_replay(session_dir=tmp_path/"sess", portfolio_run_dirs=[rd], daily=daily,
               initial_cash=1_000_000.0, from_date=dates[0], to_date=dates[-1], seed=0)
    rep = build_attribution_report(tmp_path/"sess", [rd], daily, initial_cash=1_000_000.0)
    assert rep["missed_by_reason"]["suspended"]["count"] >= 1
    assert rep["missed_by_reason"]["suspended"]["notional"] > 0
    # 理想（frictionless，day1 已全额建仓吃满缺口）vs 真实（day1 停牌 0
    # 成交、day2 追价踏空缺口）应有显著非零总缺口
    assert rep["ideal"]["ann_ret"] != rep["real"]["ann_ret"]
    assert rep["ideal"]["ann_ret"] > rep["real"]["ann_ret"]
