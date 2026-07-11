# src/factorzen/pipelines/factor_mine.py
"""fz mine 的 pipeline 入口：拉数据 → run_session。"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.guardrails import DEFAULT_DSR_ALPHA
from factorzen.discovery.mining_session import run_session

_TRADING_YEAR = 252
# agent/team（LLM）路的预热前缀交易日数。LLM 窗口无搜索空间上界，实测 structured 爱提
# 250/252 日长窗因子（nested → required_lookback 可达 ~500）。取两个嵌套交易年（2×252=504）
# 作前缀，覆盖『年度统计量的 z-score/变化率』这类合理长窗因子，免得被预热门（正确地）判欠
# 预热、永远评估不到。比 search_space_max_lookback（180，只覆盖随机搜索 windows≤60）大，
# 是 agent 路专用；M1 `run_mine` 仍用默认 180（其搜索空间上界）。更深的嵌套长窗仍会被
# （正确地）判欠预热——这是数据供给的诚实上界，不是 bug。
AGENT_WARMUP_LOOKBACK = 2 * _TRADING_YEAR  # 504 交易日 ≈ 两年


def prepare_mining_daily(start: str, end: str, universe: str | None = None,
                         lookback_days: int | None = None) -> pl.DataFrame:
    """构建 A 股挖掘/评估用日线帧：**复权价**(FactorDataContext 的 close_adj) + join
    daily_basic(激活 total_mv/pb/pe_ttm 等 BASIC_FEATURES 叶子)。

    搜索路径(run_mine)与 Agent 挖掘路径(fz mine agent/team)共用本函数，消除双路径漂移——
    否则 agent 路径用 loader.fetch_daily 的未复权 close 冒充复权价(除权日假收益)、且缺
    daily_basic/派生叶子，LLM 被广告的叶子过半在评估帧不存在。

    ``lookback_days``：预热前缀交易日数。``None``（默认）取 `search_space_max_lookback()`
    —— 覆盖搜索空间最大回看，否则长窗口/深嵌套因子在 `eval_start` 处欠预热被预热门误拒。
    """
    from factorzen.core.universe import get_universe
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.search.random_search import search_space_max_lookback

    if lookback_days is None:
        lookback_days = search_space_max_lookback()
    uni = None
    if universe:
        uni = get_universe(end, universe)["ts_code"].to_list()
    ctx = FactorDataContext(start=start, end=end, required_data=["daily", "daily_basic"],
                            lookback_days=lookback_days, universe=uni)
    daily = ctx.daily.collect()
    basic = ctx.daily_basic.collect()
    if not basic.is_empty():
        daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
    return daily


def run_mine(*, start: str, end: str, universe: str | None = None,
             n_trials: int = 200, top_k: int = 10, seed: int = 42,
             method: str = "random", holdout_ratio: float = 0.2,
             train_ratio: float = 0.7, decorr_threshold: float = 0.7,
             min_n_train: int = 5, dsr_alpha: float = DEFAULT_DSR_ALPHA,
             workers: int = 1) -> dict:
    daily = prepare_mining_daily(start, end, universe)
    return run_session(daily, n_trials=n_trials, top_k=top_k, seed=seed, method=method,
                       holdout_ratio=holdout_ratio, train_ratio=train_ratio,
                       decorr_threshold=decorr_threshold, min_n_train=min_n_train,
                       dsr_alpha=dsr_alpha, eval_start=start, workers=workers)
