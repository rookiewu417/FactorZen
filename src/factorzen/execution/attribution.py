"""A 类分歧归因：理想(frictionless孪生) vs 可达(真实ledger)，桶+residual。

诚实：总缺口独立测(两条NAV)；成本/滑点逐笔精确;未成交给名义额;residual=余量不强制0。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.daily.evaluation.backtest import _precompute_adv_20d_by_date
from factorzen.execution.brokers.paper import PaperBroker
from factorzen.execution.drivers import _market_of_day
from factorzen.execution.engine import step
from factorzen.execution.store import SessionStore
from factorzen.sim.engine import _load_weights_by_date

TRADING_DAYS = 252


def _metrics(nav: list[float]) -> dict:
    if len(nav) < 2:
        return {"ann_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    arr = np.array(nav, dtype=float)
    rets = arr[1:] / arr[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return {"ann_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    ann = float(np.mean(rets) * TRADING_DAYS)
    vol = float(np.std(rets) * np.sqrt(TRADING_DAYS))
    sharpe = ann / vol if vol > 0 else 0.0
    cum = np.concatenate([[1.0], np.cumprod(1 + rets)])
    max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
    return {"ann_ret": ann, "sharpe": sharpe, "max_dd": max_dd}


def _ideal_nav(portfolio_run_dirs: list[str], daily: pl.DataFrame, initial_cash: float) -> list[float]:
    """frictionless 孪生：同 weights/dates，按 close 全额零成本成交，收集理想 nav 序列。"""
    weights_by_date = _load_weights_by_date(portfolio_run_dirs)
    all_dates = sorted(daily.select("trade_date").unique()["trade_date"].to_list())
    adv_by_date = _precompute_adv_20d_by_date(daily, all_dates)
    broker = PaperBroker(initial_cash=initial_cash, frictionless=True)
    nav = [initial_cash]
    cur: dict[str, float] = {}
    for d in all_dates:
        market = _market_of_day(daily, d, adv_by_date)
        broker.advance_to(d, market)
        applicable = [s for s in weights_by_date if s <= d]
        if applicable:
            wdf = weights_by_date[max(applicable)]
            cur = dict(zip(wdf["ts_code"].to_list(), wdf["target_weight"].to_list(), strict=True))
        if not cur:
            nav.append(broker.get_cash().total_asset)
            continue
        ref_price = {c: m["close"] for c, m in market.items() if m.get("close")}
        step(broker, cur, ref_price)
        nav.append(broker.get_cash().total_asset)
    return nav


def build_attribution_report(
    session_dir: str | Path,
    portfolio_run_dirs: list[str],
    daily: pl.DataFrame,
    *,
    initial_cash: float,
) -> dict:
    store = SessionStore(session_dir)
    recs = store.ledger_records()
    real_nav = [initial_cash] + [r["nav_after"] for r in recs]
    ideal_nav = _ideal_nav(portfolio_run_dirs, daily, initial_cash)

    # 逐笔精确桶：成本 + 滑点（滑点=filled×(open−close)，ref=close 定量、exec=open）
    px: dict[tuple[str, str], dict] = {}
    for row in daily.iter_rows(named=True):
        px[(row["trade_date"].isoformat(), row["ts_code"])] = {
            "open": row.get("open"),
            "close": row.get("close"),
        }
    base = max(initial_cash, 1.0)
    cost_sum = 0.0
    slip_sum = 0.0
    missed: dict[str, dict] = {}
    for r in recs:
        d = r["as_of_date"]
        for f in r["fills"]:
            cost_sum += float(f["cost"])
            p = px.get((d, f["ts_code"]), {})
            o, c = p.get("open"), p.get("close")
            if o is not None and c is not None:
                sign = 1.0 if f["side"] == "buy" else -1.0
                slip_sum += f["filled_volume"] * (float(o) - float(c)) * sign
        # 未成交/部分成交：按 ack reason 归名义额
        filled_by_code: dict[str, float] = {}
        for f in r["fills"]:
            filled_by_code[f["ts_code"]] = filled_by_code.get(f["ts_code"], 0) + f["filled_volume"]
        for od, ack in zip(r["orders"], r["acks"], strict=True):
            if ack.get("accepted"):
                continue
            reason = ack.get("reason") or "unknown"
            c = px.get((d, od["ts_code"]), {}).get("close")
            shortfall = od["volume"] - filled_by_code.get(od["ts_code"], 0)
            notional = shortfall * float(c) if c is not None else 0.0
            m = missed.setdefault(reason, {"count": 0, "notional": 0.0})
            m["count"] += 1
            m["notional"] += notional

    n_days = len(recs)
    # ideal/real 的 ann_ret 是「日均收益 * 252」的年化口径（_metrics），因此
    # total_gap_bps 也是年化 bps。cost_sum/slip_sum 是整段窗口的累计 $ 成本，
    # 必须同样折算成「年化 bps」（累计成本占比 / 窗口天数 * 252）才能与
    # total_gap_bps 同口径相减，否则 residual 会被窗口长度人为放大或缩小
    # （短窗口尤其明显：把几天的一次性成本外推成一整年会被放大 252/n_days
    # 倍，ann_ret 差值同理放大，两边不做同一折算就没有可比性）。
    ann_factor = TRADING_DAYS / n_days if n_days > 0 else 0.0
    cost_bps = cost_sum / base * 1e4 * ann_factor
    slip_bps = slip_sum / base * 1e4 * ann_factor
    ideal_m = _metrics(ideal_nav)
    real_m = _metrics(real_nav)
    total_gap = ideal_m["ann_ret"] - real_m["ann_ret"]
    total_gap_bps = total_gap * 1e4
    residual_bps = total_gap_bps - cost_bps - slip_bps

    report = {
        "ideal": ideal_m,
        "real": real_m,
        "total_gap_ann_ret": total_gap,
        "cost_bps": cost_bps,
        "slippage_bps": slip_bps,
        "residual_bps": residual_bps,
        "missed_by_reason": missed,
        "n_days": n_days,
    }
    (Path(session_dir) / "attribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report
