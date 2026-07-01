"""择时 vs 基线实验驱动：生成两套 weights 产物 → 各走 run_replay → 汇总 NAV 指标对比。

离线端到端：产物生成（``trend_timing``）→ 逐日 replay（``execution.drivers``）→
NAV 指标（复用 ``execution.attribution._metrics``，年化 ×252）全走本地 DataFrame，
无网络依赖，供真实数据场景（HS300 缓存）与单测共用同一入口。
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.execution.attribution import _metrics
from factorzen.execution.drivers import run_replay
from factorzen.execution.store import SessionStore
from factorzen.strategies.trend_timing import generate_trend_timing_products


def _nav_metrics(session_dir: Path, initial_cash: float) -> dict:
    """还原 session 的完整 NAV 序列（补回起点 ``initial_cash``）并算年化指标。"""
    nav_df = SessionStore(session_dir).nav_frame()
    nav = [initial_cash] + (nav_df["nav_after"].to_list() if nav_df.height else [])
    return _metrics(nav)


def run_trend_timing_experiment(
    out_root: str,
    index_daily: pl.DataFrame,
    price_daily: pl.DataFrame,
    rebalance_dates: list[date],
    *,
    initial_cash: float,
    from_date: date,
    to_date: date,
    members_fn: Callable[[str, str], list[str]] | None = None,
    ma_window: int = 200,
    top_n: int = 50,
    seed: int = 0,
    **kw: Any,
) -> dict:
    """跑「策略（择时 overlay）vs 基线（始终满仓）」两套离线实验，返回指标对比。

    策略（``timing=True``）产物落 ``out_root/strategy/products/``，会话落
    ``out_root/strategy/session/``；基线（``timing=False``）同理落
    ``out_root/baseline/``。两套各自独立走 ``generate_trend_timing_products``
    →``run_replay``，互不共享落盘目录/状态。

    Parameters
    ----------
    out_root : str
        实验根目录。
    index_daily, price_daily, rebalance_dates, members_fn, ma_window, top_n
        透传给 ``generate_trend_timing_products``（``timing`` 由本函数按
        策略/基线分别固定，不接受调用方覆盖）。
    initial_cash, from_date, to_date, seed
        透传给 ``run_replay``。
    **kw
        额外透传给 ``generate_trend_timing_products``（如 ``index_code``）。

    Returns
    -------
    dict
        ``{"strategy": {"metrics": {...}, "session_dir": str},
           "baseline": {"metrics": {...}, "session_dir": str}}``，
        ``metrics`` 含 ``ann_ret``/``sharpe``/``max_dd``。
    """
    root = Path(out_root)
    out: dict[str, dict] = {}
    for label, timing in (("strategy", True), ("baseline", False)):
        products_dir = root / label / "products"
        run_dirs = generate_trend_timing_products(
            str(products_dir),
            index_daily,
            price_daily,
            rebalance_dates,
            members_fn=members_fn,
            ma_window=ma_window,
            top_n=top_n,
            timing=timing,
            **kw,
        )

        session_dir = root / label / "session"
        run_replay(
            session_dir=session_dir,
            portfolio_run_dirs=run_dirs,
            daily=price_daily,
            initial_cash=initial_cash,
            from_date=from_date,
            to_date=to_date,
            seed=seed,
        )

        out[label] = {
            "metrics": _nav_metrics(session_dir, initial_cash),
            "session_dir": str(session_dir),
        }
    return out
