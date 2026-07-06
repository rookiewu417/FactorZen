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
    # 把 daily_basic join 进来（与 ExpressionFactor.compute 一致），否则搜索空间里
    # total_mv/pb/pe_ttm 等 BASIC_FEATURES 叶子在缓存帧里不存在→候选 compile
    # ColumnNotFound 被静默跳过，估值/换手类因子永远挖不出。
    basic = ctx.daily_basic.collect()
    if not basic.is_empty():
        daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
    return run_session(daily, n_trials=n_trials, top_k=top_k, seed=seed, method=method,
                       holdout_ratio=holdout_ratio, train_ratio=train_ratio,
                       decorr_threshold=decorr_threshold, min_n_train=min_n_train,
                       dsr_alpha=dsr_alpha, eval_start=start, workers=workers)
