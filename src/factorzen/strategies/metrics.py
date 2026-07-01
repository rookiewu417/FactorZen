"""策略回测指标：从执行会话（nav.parquet + ledger.parquet）算完整指标 + 表格化打印。

比 ``execution.attribution._metrics``（仅 ann_ret/sharpe/max_dd）更全，且从 ledger
读成交明细算换手/成本，供策略回测「跑完 print 一张表」。
"""
from __future__ import annotations

import numpy as np

from factorzen.execution.store import SessionStore

TRADING_DAYS = 252

# (key, 中文标签, 格式)  —— 表格行顺序
_ROWS = [
    ("total_return", "总收益", "pct"),
    ("ann_ret", "年化收益", "pct"),
    ("ann_vol", "年化波动", "pct"),
    ("sharpe", "夏普", "num"),
    ("max_dd", "最大回撤", "pct"),
    ("calmar", "Calmar", "num"),
    ("win_rate", "日胜率", "pct"),
    ("ann_turnover", "年化换手(双边)", "num"),
    ("total_cost_bps", "累计成本(bps)", "bps"),
    ("n_days", "交易日数", "int"),
    ("n_fills", "成交笔数", "int"),
]


def _metrics_from_nav(navs: list[float]) -> dict:
    """由 NAV 序列（含起点）算净值类指标。"""
    zero = {k: 0.0 for k in
            ("total_return", "ann_ret", "ann_vol", "sharpe", "max_dd", "calmar", "win_rate")}
    zero["n_days"] = 0
    if len(navs) < 2:
        return zero
    arr = np.asarray(navs, dtype=float)
    rets = arr[1:] / arr[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return zero
    ann_ret = float(np.mean(rets) * TRADING_DAYS)
    ann_vol = float(np.std(rets) * np.sqrt(TRADING_DAYS))
    cum = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1.0))
    return {
        "total_return": float(arr[-1] / arr[0] - 1.0),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "max_dd": max_dd,
        "calmar": ann_ret / abs(max_dd) if max_dd < 0 else 0.0,
        "win_rate": float(np.mean(rets > 0)),
        "n_days": len(rets),
    }


def session_metrics(session_dir: str, initial_cash: float) -> dict:
    """读执行会话，算完整指标：净值类 + 换手/成本（从 ledger 成交明细）。"""
    store = SessionStore(session_dir)
    nav_df = store.nav_frame()
    nav_after = nav_df.sort("as_of_date")["nav_after"].to_list() if nav_df.height else []
    navs = [float(initial_cash), *[float(x) for x in nav_after]]
    m = _metrics_from_nav(navs)

    traded = 0.0
    cost = 0.0
    n_fills = 0
    for rec in store.ledger_records():
        for f in rec["fills"]:
            traded += float(f["filled_volume"]) * float(f["price"])
            cost += float(f["cost"])
            n_fills += 1

    mean_nav = sum(navs) / len(navs)
    years = max(m["n_days"], 1) / TRADING_DAYS
    m["ann_turnover"] = traded / mean_nav / years if mean_nav > 0 else 0.0
    m["total_cost"] = cost
    m["total_cost_bps"] = cost / initial_cash * 1e4 if initial_cash > 0 else 0.0
    m["n_fills"] = n_fills
    return m


def _fmt(val, kind: str) -> str:
    if kind == "pct":
        return f"{val * 100:+.2f}%"
    if kind == "bps":
        return f"{val:.1f}"
    if kind == "int":
        return f"{int(val)}"
    return f"{val:.3f}"  # num


def format_metrics_table(named: dict[str, dict], *, order: list[str] | None = None) -> str:
    """named = {列名: metrics dict} → 对齐的中文指标对比表（等宽）。"""
    cols = order or list(named)
    label_w = max(len(lbl) for _, lbl, _ in _ROWS) + 2
    col_w = max(10, *(len(c) for c in cols)) + 2
    head = "指标".ljust(label_w) + "".join(c.rjust(col_w) for c in cols)
    lines = [head, "-" * len(head)]
    for key, lbl, kind in _ROWS:
        row = lbl.ljust(label_w)
        for c in cols:
            v = named[c].get(key)
            row += ("—".rjust(col_w) if v is None else _fmt(v, kind).rjust(col_w))
        lines.append(row)
    return "\n".join(lines)
