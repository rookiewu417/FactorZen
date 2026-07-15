"""crypto 专属净值回测（不碰 A 股 backtest.py 快路径）。

为什么独立：A 股快路径深度耦合涨跌停(前缀判板)/T+1/印花税/252 年化，且成本接口
签名不兼容 crypto，funding 概念不存在。crypto 用一套干净的日频 NAV 递推：
- 收益：daily close 收益，权重带符号(可做空)。
- 换手成本：|Δweight|·notional × (taker+slippage)，买卖对称、无印花税。
- funding 逐期计提：持仓 × 当日 funding（多头付正、空头收）。
- 年化 365。

MVP 简化：调仓日之间持有目标权重不做日内漂移（等价每日拉回目标），换手成本只在
调仓日计提；funding 逐日按持仓计提。撮合口径 T+0（信号日即生效）。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.config.settings import SIM_DIR
from factorzen.markets.crypto.costs import CryptoCostModel

_CRYPTO_PERIODS_PER_YEAR = 365


def _matrix(df: pl.DataFrame | None, value_col: str, dates: list, date_idx: dict, sym_idx: dict,
            fill: float) -> np.ndarray:
    m = np.full((len(dates), len(sym_idx)), fill, dtype=float)
    if df is None or df.is_empty():
        return m
    for row in df.iter_rows(named=True):
        di = date_idx.get(row["trade_date"])
        sj = sym_idx.get(row["ts_code"])
        if di is not None and sj is not None and row[value_col] is not None:
            m[di, sj] = float(row[value_col])
    return m


def _coerce_signal_keys(weights_by_date: dict, freq: str) -> dict:
    """intraday 时把 date 信号键升为当日零点 datetime(date/datetime 混排会 TypeError)。"""
    from datetime import date as _date
    from datetime import datetime as _datetime
    if freq == "daily":
        return weights_by_date
    return {
        (_datetime(k.year, k.month, k.day)
         if isinstance(k, _date) and not isinstance(k, _datetime) else k): v
        for k, v in weights_by_date.items()
    }


def simulate_crypto_nav(
    weights_by_date: dict,
    daily: pl.DataFrame,
    funding: pl.DataFrame | None = None,
    *,
    cost_model: CryptoCostModel | None = None,
    periods_per_year: int = _CRYPTO_PERIODS_PER_YEAR,
) -> dict:
    """给定各调仓日权重 + crypto 日 bar (+funding)，跑净值回测。

    返回 ``{"nav": pl.DataFrame, "metrics": dict}``。
    nav 列：``trade_date, gross_return, cost, borrow_cost(=funding), net_return, nav, cash_weight``。
    """
    if cost_model is None:
        cost_model = CryptoCostModel()
    symbols = sorted({c for w in weights_by_date.values() for c in w["ts_code"].to_list()})
    if not symbols:
        raise ValueError("weights_by_date 为空，无法回测")
    sig_dates = sorted(weights_by_date)
    first_signal = sig_dates[0]

    all_dates = sorted(daily["trade_date"].unique().to_list())
    dates = [d for d in all_dates if d >= first_signal]
    if len(dates) < 2:
        raise ValueError("回测区间内交易日不足（价格日 < 2 或均早于首个信号日）")

    date_idx = {d: i for i, d in enumerate(dates)}
    sym_idx = {s: j for j, s in enumerate(symbols)}
    close = _matrix(daily, "close", dates, date_idx, sym_idx, np.nan)
    fund = _matrix(funding, "funding_rate", dates, date_idx, sym_idx, 0.0)

    # 日收益 r[t] = close[t]/close[t-1]-1（缺失→0）
    r = np.zeros_like(close)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = close[1:] / close[:-1] - 1.0
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # 逐日目标权重（前向填充最近信号）
    T, N = len(dates), len(symbols)
    W = np.zeros((T, N))
    current = np.zeros(N)
    sig_ptr = 0
    for i, d in enumerate(dates):
        while sig_ptr < len(sig_dates) and sig_dates[sig_ptr] <= d:
            current = np.zeros(N)
            wdf = weights_by_date[sig_dates[sig_ptr]]
            for row in wdf.iter_rows(named=True):
                if row["ts_code"] in sym_idx:
                    current[sym_idx[row["ts_code"]]] = float(row["target_weight"])
            sig_ptr += 1
        W[i] = current

    taker_slip = cost_model.taker + cost_model.slippage
    nav = 1.0
    navs, nets, grosses, costs, fundings, turnovers, cash_w = [], [], [], [], [], [], []
    for i in range(T):
        w_active = W[i]                       # 从今日起持有
        held = W[i - 1] if i > 0 else np.zeros(N)  # 今日内持有的仓位（昨日设定）
        gross = float(np.sum(held * r[i])) if i > 0 else 0.0
        funding_cost = float(np.sum(held * fund[i])) if i > 0 else 0.0
        turnover = float(np.sum(np.abs(w_active - held)))
        cost = turnover * taker_slip          # = trade_cost(notional=turnover)
        net = gross - cost - funding_cost
        nav *= 1.0 + net
        navs.append(nav)
        nets.append(net)
        grosses.append(gross)
        costs.append(cost)
        fundings.append(funding_cost)
        turnovers.append(turnover)
        cash_w.append(1.0 - float(np.sum(np.abs(w_active))))

    nav_df = pl.DataFrame({
        "trade_date": dates, "gross_return": grosses, "cost": costs,
        "borrow_cost": fundings, "net_return": nets, "nav": navs, "cash_weight": cash_w,
    })
    metrics = _metrics(np.array(nets), np.array(navs), np.array(turnovers),
                       float(sum(costs)), periods_per_year)
    metrics["total_funding"] = float(sum(fundings))  # 累计资金费成本（多头付/空头收，净额）
    return {"nav": nav_df, "metrics": metrics}


def run_crypto_simulation(
    portfolio_run_dirs: list[str],
    profile,
    start: str,
    end: str,
    *,
    symbols: list[str] | None = None,
    freq: str | None = None,
    out_dir: str = str(SIM_DIR),
    run_id: str | None = None,
) -> dict:
    """crypto 模拟交易编排：读组合权重 → 拉 crypto bar+funding → NAV 回测 → 落盘。

    复用 sim.engine._load_weights_by_date（市场无关：读 manifest.signal_date + weights.parquet）。
    产出 ``nav.parquet``/``metrics.json``/``manifest.json``，schema 对齐 A 股 sim 供下游复用。
    """
    import json
    import subprocess
    from pathlib import Path

    from factorzen.sim.engine import _load_weights_by_date

    freq = freq or profile.base_freq
    weights_by_date = _load_weights_by_date(portfolio_run_dirs)
    if not weights_by_date:
        raise ValueError("未从组合目录读到任何权重（缺 manifest.signal_date 或 weights.parquet）")
    weights_by_date = _coerce_signal_keys(weights_by_date, freq)
    if symbols is None:
        symbols = sorted({c for w in weights_by_date.values() for c in w["ts_code"].to_list()})

    provider = profile.provider
    assert hasattr(provider, "fetch_funding"), "run_crypto_simulation 需 crypto profile(provider 缺 funding 扩展)"
    bars = provider.fetch_bars(symbols, start, end, freq)
    funding = provider.fetch_funding(symbols, start, end, freq)
    sim = simulate_crypto_nav(
        weights_by_date, bars, funding, cost_model=profile.costs,
        periods_per_year=int(profile.calendar.periods_per_year(freq)),
    )

    rid = run_id or "sim"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    sim["nav"].write_parquet(run_dir / "nav.parquet")
    (run_dir / "metrics.json").write_text(json.dumps(sim["metrics"], ensure_ascii=False, indent=2))
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"
    # 可复现铁律#3：与 A 股 sim.engine 对齐，补 inputs/窗口/成本/command
    from factorzen.sim.engine import _jsonable

    sig_dates = list(weights_by_date.keys())
    nav_df = sim["nav"]
    n_exec_dates = int(nav_df.height) if not nav_df.is_empty() else 0

    def _sig_iso(d: object) -> str:
        return d.isoformat() if hasattr(d, "isoformat") else str(d)

    manifest = {
        "run_id": rid,
        "market": profile.name,
        "freq": freq,
        "n_signals": len(weights_by_date),
        "n_symbols": len(symbols),
        "git_sha": git_sha,
        "inputs": list(portfolio_run_dirs),
        "start": start,  # 回测窗口（调用方入参）；信号窗口见 signal_start/end
        "end": end,
        "signal_start": _sig_iso(min(sig_dates)) if sig_dates else None,
        "signal_end": _sig_iso(max(sig_dates)) if sig_dates else None,
        "n_exec_dates": n_exec_dates,
        "cost_model": _jsonable(profile.costs),
        "command": "crypto sim run",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    m = sim["metrics"]
    return {"run_dir": str(run_dir), "sharpe": m["sharpe"], "max_dd": m["max_dd"],
            "ann_ret": m["ann_ret"]}


def _metrics(net: np.ndarray, nav: np.ndarray, turnover: np.ndarray,
             total_cost: float, ppy: int) -> dict:
    if net.size == 0:
        return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "avg_turnover": 0.0, "total_cost": 0.0}
    ann_ret = float(np.mean(net) * ppy)
    ann_vol = float(np.std(net) * np.sqrt(ppy))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
    peak = np.maximum.accumulate(nav)
    max_dd = float(np.min(nav / peak - 1.0)) if nav.size else 0.0
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd,
            "avg_turnover": float(np.mean(turnover)), "total_cost": total_cost}
