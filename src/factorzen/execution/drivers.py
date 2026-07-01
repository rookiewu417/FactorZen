"""驱动：把行情窗口逐日喂给 engine.step。replay = 单进程循环历史交易日。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.daily.evaluation.backtest import _precompute_adv_20d_by_date
from factorzen.execution.brokers.paper import PaperBroker
from factorzen.execution.engine import step
from factorzen.execution.store import SessionStore
from factorzen.sim.engine import _load_weights_by_date


def _market_of_day(
    daily: pl.DataFrame, d: date, adv_by_date: dict[date, dict[str, float]] | None = None
) -> dict[str, dict[str, Any]]:
    day = daily.filter(pl.col("trade_date") == d)
    adv_today = (adv_by_date or {}).get(d, {})
    out: dict[str, dict[str, Any]] = {}
    for row in day.iter_rows(named=True):
        out[row["ts_code"]] = {
            "open": row.get("open"), "pre_close": row.get("pre_close"),
            "close": row.get("close"), "vol": row.get("vol"),
            "adv": adv_today.get(row["ts_code"]),
        }
    return out


def run_replay(
    session_dir: str | Path,
    portfolio_run_dirs: list[str],
    daily: pl.DataFrame,
    initial_cash: float,
    from_date: date | None = None,
    to_date: date | None = None,
    seed: int = 0,
) -> dict:
    weights_by_date = _load_weights_by_date(portfolio_run_dirs)  # {signal_date: DF[ts_code,target_weight]}
    store = SessionStore(session_dir)
    store.init({"broker": "paper", "initial_cash": initial_cash, "seed": seed,
                "command": ["fz", "live", "replay"]})
    broker = PaperBroker(initial_cash=initial_cash)

    all_dates = sorted(daily.select("trade_date").unique()["trade_date"].to_list())
    dates = [d for d in all_dates
             if (from_date is None or d >= from_date) and (to_date is None or d <= to_date)]
    # 容量约束需要真实 ADV：对 daily 全量 trade_dates 预计算一次 trailing 20 日
    # 成交额均值（_precompute_adv_20d_by_date 对当日 shift(1)，无未来函数）。
    # daily 缺 amount 列时该函数优雅降级返回 {}，adv 保持 None，不崩、不报错。
    adv_by_date = _precompute_adv_20d_by_date(daily, all_dates)

    current_weights: dict[str, float] = {}
    n_steps = 0
    for d in dates:
        if store.has_date(d):
            # 幂等哨兵：重跑同一 session_dir 时跳过已落盘的交易日，避免
            # ledger/nav 追加重复行。
            continue
        market = _market_of_day(daily, d, adv_by_date)
        broker.advance_to(d, market)
        # 采用「≤ 当日的最新一次信号」的目标权重（PIT）
        applicable = [s for s in weights_by_date if s <= d]
        if applicable:
            latest = max(applicable)
            wdf = weights_by_date[latest]
            current_weights = dict(
                zip(wdf["ts_code"].to_list(), wdf["target_weight"].to_list(), strict=True)
            )
        if not current_weights:
            continue
        ref_price = {c: m["close"] for c, m in market.items() if m.get("close")}
        rec = step(broker, current_weights, ref_price)
        rec["as_of_date"] = d.isoformat()
        store.append(rec)
        n_steps += 1

    final_nav = broker.get_cash().total_asset
    return {"session_dir": str(Path(session_dir)), "n_steps": n_steps, "final_nav": final_nav}


def run_daily_step(
    session_dir: str | Path,
    as_of: date,
    portfolio_run_dirs: list[str],
    daily: pl.DataFrame,
    *,
    config: dict,
) -> dict:
    """单日推进：供每日调度（如 cron/DAG）逐日调用的可续跑入口。

    与 ``run_replay``（单进程一次性跑完整段历史）不同，本函数每次只处理一个
    交易日，状态靠 ``SessionStore.load_state``/``append`` 落盘续跑，容忍
    「每天起一个新进程」的调度模式。幂等：``store.has_date`` 命中则跳过，不
    重复下单/追加 ledger 行。
    """
    store = SessionStore(session_dir)
    if store.has_date(as_of):
        return {"as_of": as_of.isoformat(), "nav_after": None, "n_fills": 0, "skipped": True}
    broker = PaperBroker(
        initial_cash=float(config["initial_cash"]),
        slippage_bps=float(config.get("slippage_bps", 0.0)),
    )
    st = store.load_state()
    if st is not None:
        broker.load_state(st)
    # 当日 market（daily 已含所需窗口；调用方保证 daily 覆盖 as_of 及其前
    # ~20 交易日以算 ADV）
    all_dates = sorted(daily.select("trade_date").unique()["trade_date"].to_list())
    adv_by_date = _precompute_adv_20d_by_date(daily, all_dates)
    market = _market_of_day(daily, as_of, adv_by_date)
    broker.advance_to(as_of, market)
    weights_by_date = _load_weights_by_date(portfolio_run_dirs)
    applicable = [s for s in weights_by_date if s <= as_of]
    if not applicable:
        return {
            "as_of": as_of.isoformat(),
            "nav_after": broker.get_cash().total_asset,
            "n_fills": 0,
            "skipped": False,
        }
    wdf = weights_by_date[max(applicable)]
    weights = dict(zip(wdf["ts_code"].to_list(), wdf["target_weight"].to_list(), strict=True))
    ref_price = {c: m["close"] for c, m in market.items() if m.get("close")}
    rec = step(broker, weights, ref_price)
    rec["as_of_date"] = as_of.isoformat()
    rec["broker_state"] = broker.state()  # 可续跑态（覆盖 step 的显示视图）
    store.append(rec)
    return {
        "as_of": as_of.isoformat(),
        "nav_after": rec["nav_after"],
        "n_fills": len(rec["fills"]),
        "skipped": False,
    }
