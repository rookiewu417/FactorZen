"""驱动：把行情窗口逐日喂给 engine.step。replay = 单进程循环历史交易日。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.execution.brokers.paper import PaperBroker
from factorzen.execution.engine import step
from factorzen.execution.store import SessionStore
from factorzen.sim.engine import _load_weights_by_date


def _market_of_day(daily: pl.DataFrame, d: date) -> dict[str, dict[str, Any]]:
    day = daily.filter(pl.col("trade_date") == d)
    out: dict[str, dict[str, Any]] = {}
    for row in day.iter_rows(named=True):
        out[row["ts_code"]] = {
            "open": row.get("open"), "pre_close": row.get("pre_close"),
            "close": row.get("close"), "vol": row.get("vol"),
            "adv": row.get("adv"),
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

    current_weights: dict[str, float] = {}
    n_steps = 0
    for d in dates:
        market = _market_of_day(daily, d)
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
