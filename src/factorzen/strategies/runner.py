"""策略实验驱动：产物生成 → replay 与/或 sim，两套消费路径并存。

- **replay 轨**（``execution.drivers.run_replay``）：纸面逐日撮合会话，供
  ``run_trend_timing_experiment`` / ``run_momentum_rotation_experiment`` 使用。
- **sim 轨**（``sim.engine.run_portfolio_simulation``）：统一日环回测引擎，
  经 ``run_strategy_simulation`` 桥接；任意 ``generate_*_products`` 产出的
  run_dirs 均可喂入（与 sim 的 weights/manifest 契约一致）。

两轨互不替代：replay 偏向前执行/会话状态；sim 偏组合研究回测与净值落盘。
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.daily.evaluation.backtest import CostModel
from factorzen.daily.evaluation.cost_models import CostModelBase
from factorzen.execution.drivers import run_replay
from factorzen.sim.engine import run_portfolio_simulation
from factorzen.strategies.metrics import format_metrics_table, session_metrics
from factorzen.strategies.momentum_rotation import generate_momentum_rotation_products
from factorzen.strategies.trend_timing import generate_trend_timing_products


def run_strategy_simulation(
    portfolio_run_dirs: list[str],
    daily: pl.DataFrame,
    *,
    out_dir: str,
    run_id: str | None = None,
    cost_model: CostModel | CostModelBase | None = None,
) -> dict:
    """把策略 weights 产物 run_dirs + daily 面板喂给 ``run_portfolio_simulation``。

    透传 ``cost_model`` / ``run_id`` / ``out_dir``，返回 sim 结果 dict
    （含 ``run_dir`` / ``sharpe`` / ``max_dd`` / ``ann_ret``）。任意
    ``generate_*_products`` 产出的目录列表均可作为 ``portfolio_run_dirs``。
    """
    return run_portfolio_simulation(
        portfolio_run_dirs,
        daily,
        out_dir=out_dir,
        run_id=run_id,
        cost_model=cost_model,
    )


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
    print_table: bool = True,
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
            "metrics": session_metrics(str(session_dir), initial_cash),
            "session_dir": str(session_dir),
        }

    if print_table:
        print(format_metrics_table({lbl: out[lbl]["metrics"] for lbl in out}))
    return out


def run_momentum_rotation_experiment(
    out_root: str,
    index_dailies: dict[str, pl.DataFrame],
    price_daily: pl.DataFrame,
    rebalance_dates: list[date],
    *,
    initial_cash: float,
    from_date: date,
    to_date: date,
    members_fn: Callable[[str, str], list[str]] | None = None,
    lookback: int = 126,
    top_n: int = 50,
    seed: int = 0,
    print_table: bool = True,
    **kw: Any,
) -> dict:
    """宽基动量轮动：生成产物 → replay，返回会话指标（与 trend_timing replay 轨对称）。

    产物落 ``out_root/products/``，会话落 ``out_root/session/``。
    若还需 sim 轨，请对返回的 ``run_dirs`` 再调 ``run_strategy_simulation``。

    Returns
    -------
    dict
        ``{"metrics": {...}, "session_dir": str, "run_dirs": list[str]}``。
    """
    root = Path(out_root)
    products_dir = root / "products"
    run_dirs = generate_momentum_rotation_products(
        str(products_dir),
        index_dailies,
        price_daily,
        rebalance_dates,
        members_fn=members_fn,
        lookback=lookback,
        top_n=top_n,
        **kw,
    )

    session_dir = root / "session"
    run_replay(
        session_dir=session_dir,
        portfolio_run_dirs=run_dirs,
        daily=price_daily,
        initial_cash=initial_cash,
        from_date=from_date,
        to_date=to_date,
        seed=seed,
    )

    metrics = session_metrics(str(session_dir), initial_cash)
    if print_table:
        print(format_metrics_table({"momentum_rotation": metrics}))
    return {
        "metrics": metrics,
        "session_dir": str(session_dir),
        "run_dirs": run_dirs,
    }
