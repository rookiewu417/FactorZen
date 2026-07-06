# src/factorzen/pipelines/factor_mine.py
"""fz mine 的 pipeline 入口：拉数据 → run_session。"""
from __future__ import annotations

from factorzen.discovery.mining_session import run_session


def run_mine(*, start: str, end: str, universe: str | None = None,
             n_trials: int = 200, top_k: int = 10, seed: int = 42,
             method: str = "random", holdout_ratio: float = 0.2,
             train_ratio: float = 0.7, decorr_threshold: float = 0.7,
             min_n_train: int = 5, dsr_alpha: float = 0.05,
             workers: int = 1) -> dict:
    from factorzen.core.universe import get_universe
    from factorzen.daily.data.context import FactorDataContext

    uni = None
    if universe:
        uni = get_universe(end, universe)["ts_code"].to_list()
    ctx = FactorDataContext(start=start, end=end, required_data=["daily", "daily_basic"],
                            lookback_days=60, universe=uni)
    daily = ctx.daily.collect()
    return run_session(daily, n_trials=n_trials, top_k=top_k, seed=seed, method=method,
                       holdout_ratio=holdout_ratio, train_ratio=train_ratio,
                       decorr_threshold=decorr_threshold, min_n_train=min_n_train,
                       dsr_alpha=dsr_alpha, eval_start=start, workers=workers)
