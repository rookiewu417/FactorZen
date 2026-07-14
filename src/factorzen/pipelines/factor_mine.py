# src/factorzen/pipelines/factor_mine.py
"""fz mine 的 pipeline 入口：拉数据 → run_session。"""
from __future__ import annotations

import logging

import polars as pl

from factorzen.discovery.guardrails import DEFAULT_DSR_ALPHA
from factorzen.discovery.mining_session import run_session

_LOG = logging.getLogger(__name__)

_TRADING_YEAR = 252
# agent/team（LLM）路的预热前缀交易日数。LLM 窗口无搜索空间上界，实测 structured 爱提
# 250/252 日长窗因子（nested → required_lookback 可达 ~500）。取两个嵌套交易年（2×252=504）
# 作前缀，覆盖『年度统计量的 z-score/变化率』这类合理长窗因子，免得被预热门（正确地）判欠
# 预热、永远评估不到。比 search_space_max_lookback（180，只覆盖随机搜索 windows≤60）大，
# 是 agent 路专用；M1 `run_mine` 仍用默认 180（其搜索空间上界）。更深的嵌套长窗仍会被
# （正确地）判欠预热——这是数据供给的诚实上界，不是 bug。
AGENT_WARMUP_LOOKBACK = 2 * _TRADING_YEAR  # 504 交易日 ≈ 两年


def _universe_asof_fallback(universe: str, end: str, *, max_months: int = 36) -> list[str]:
    """命名 universe 在 ``end`` 无成分快照时，按月回退找最近有成分的日期（as-of）。

    修 OOM 主因：指数成分数据未回补到 ``end`` 时 `get_universe(end, name)` 返回空 →
    空池被 `FactorDataContext` 当「不过滤」→ 装配全市场（数千股，15x 数据膨胀 → OOM）。
    回退到最近有成分的快照（成分随时间缓慢漂移，用近端快照评估历史窗口足够）；回退 ``max_months``
    月仍空则**报错**（绝不静默退化成全市场，那会 OOM 且改变评估口径）。
    """
    import datetime as _dt

    from factorzen.core.universe import get_universe as _gu

    probe = _dt.datetime.strptime(end, "%Y%m%d").date()
    for _ in range(max_months):
        probe = probe - _dt.timedelta(days=30)
        cand = _gu(probe.strftime("%Y%m%d"), universe)["ts_code"].to_list()
        if cand:
            _LOG.warning(
                "universe %s 在 %s 无成分快照，as-of 回退到 %s（%d 只）——"
                "指数成分数据可能未回补到该日期。", universe, end, probe.strftime("%Y%m%d"), len(cand))
            return cand
    raise ValueError(
        f"universe={universe!r} 在 {end} 及此前 {max_months} 个月均无成分快照；"
        f"请回补指数成分数据，或改用 --universe all_a / 显式 --start --end。")


def _attach_in_universe(daily: pl.DataFrame, membership: pl.DataFrame,
                        start: str) -> pl.DataFrame:
    """将逐日 membership join 到 daily，得到 ``in_universe`` 布尔列。

    - membership 覆盖 [评估窗 start, end]；预热段（trade_date < start）自然 left-join 为
      null → False（行保留，供滚动算子连续时序；预热行不参与评估截面）。
    - 评估窗内非成分日 = False；成分日 = True。
    - **绝不按日删行**——并集股票的全部连续行保留。
    """
    if membership.is_empty():
        return daily.with_columns(pl.lit(False).alias("in_universe"))

    # membership.trade_date 为 Utf8；对齐 daily.trade_date 的 dtype 再 join
    td_dtype = daily.schema.get("trade_date")
    mem = membership.select(["trade_date", "ts_code"]).unique()
    if td_dtype == pl.Date:
        mem = mem.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d")
        )
    elif td_dtype is not None and td_dtype != pl.Utf8:
        # Datetime 等：先解析为 Date 再 cast
        mem = mem.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").cast(td_dtype)
        )

    mem = mem.with_columns(pl.lit(True).alias("in_universe"))
    out = daily.join(mem, on=["trade_date", "ts_code"], how="left")
    return out.with_columns(pl.col("in_universe").fill_null(False))


def prepare_mining_daily(start: str, end: str, universe: str | None = None,
                         lookback_days: int | None = None) -> pl.DataFrame:
    """构建 A 股挖掘/评估用日线帧：**复权价**(FactorDataContext 的 close_adj) + join
    daily_basic(激活 total_mv/pb/pe_ttm 等 BASIC_FEATURES 叶子)。

    搜索路径(run_mine)与 Agent 挖掘路径(fz mine agent/team)共用本函数，消除双路径漂移——
    否则 agent 路径用 loader.fetch_daily 的未复权 close 冒充复权价(除权日假收益)、且缺
    daily_basic/派生叶子，LLM 被广告的叶子过半在评估帧不存在。

    ``lookback_days``：预热前缀交易日数。``None``（默认）取 `search_space_max_lookback()`
    —— 覆盖搜索空间最大回看，否则长窗口/深嵌套因子在 `eval_start` 处欠预热被预热门误拒。

    命名 universe（非 None）时：
    1. 用 ``get_universe_membership(start, end, universe)`` 构造评估窗逐日 PIT 成分
       （预热段不进 membership；预热行 ``in_universe=False`` 但保留）；
    2. 拉取集合 = 窗口内曾在成分内的 ``ts_code`` 并集（替代期末快照，消除幸存偏差）；
    3. 原始 daily **不按日删行**——滚动算子需要连续时序；评估截面由
       ``in_universe`` 在 evaluation / mining_session 评估帧形成点过滤；
    4. membership 构造失败时回退期末快照过滤（``get_universe(end, ...)``）并 warning，
       保证挖掘链路不崩。

    不做的事：manifest ``membership_hash`` 接线；动态过滤池的逐日 membership
    （``get_universe_membership`` 对 daily_default 等抛 ValueError，本函数捕获后回退期末快照）。
    """
    from factorzen.core.universe import get_universe, get_universe_membership
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.search.random_search import search_space_max_lookback

    if lookback_days is None:
        lookback_days = search_space_max_lookback()

    uni: list[str] | None = None
    membership: pl.DataFrame | None = None
    if universe:
        try:
            membership = get_universe_membership(start, end, universe)
            uni = membership["ts_code"].unique().to_list()
            if not uni and universe != "all_a":
                # membership 空但命名指数池：沿用 as-of 回退（成分数据未回补）
                uni = _universe_asof_fallback(universe, end)
                membership = None
        except Exception as exc:
            _LOG.warning(
                "universe membership 构造失败（%s: %s），回退期末快照过滤 "
                "get_universe(%s, %s)",
                type(exc).__name__, exc, end, universe,
            )
            uni = get_universe(end, universe)["ts_code"].to_list()
            if not uni and universe != "all_a":
                uni = _universe_asof_fallback(universe, end)
            membership = None

    ctx = FactorDataContext(start=start, end=end, required_data=["daily", "daily_basic"],
                            lookback_days=lookback_days, universe=uni)
    daily = ctx.daily.collect()
    basic = ctx.daily_basic.collect()
    if not basic.is_empty():
        daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
    # 基本面叶子（roe/margin/yoy，按公告日 PIT）+ 股东户数（ann_date PIT）+
    # 资金流/北向/两融/龙虎榜（日频；两融/龙虎榜 lag 在 attach_flows 内置）。
    # 物化路径 ExpressionFactor.compute 同样 attach（共用同一函数，防双路径漂移，陷阱#2）。
    from factorzen.daily.data.flows import attach_flows
    from factorzen.daily.data.pit import attach_fundamentals, attach_holders
    daily = attach_fundamentals(daily)
    daily = attach_holders(daily)
    daily = attach_flows(daily)

    if membership is not None:
        daily = _attach_in_universe(daily, membership, start)
    return daily


def run_mine(*, start: str, end: str, universe: str | None = None,
             n_trials: int = 200, top_k: int = 10, seed: int = 42,
             method: str = "random", holdout_ratio: float = 0.2,
             train_ratio: float = 0.7, decorr_threshold: float = 0.7,
             min_n_train: int = 5, dsr_alpha: float = DEFAULT_DSR_ALPHA,
             workers: int = 1, update_library: bool = True,
             library_orthogonal: bool = True,
             objective: str = "residual") -> dict:
    daily = prepare_mining_daily(start, end, universe)
    # 收尾自动 upsert 因子库（--no-library 关）；库根由 run_session 从 out_dir 推导
    # （workspace/mining_sessions → workspace/factor_library）。universe 落进记录溯源。
    # library_orthogonal：搜索期避开库内已覆盖方向（--no-library-orthogonal 关）。
    # objective：残差/裸 IC 挖掘目标（库空时 residual 自动退化 raw）。
    return run_session(daily, n_trials=n_trials, top_k=top_k, seed=seed, method=method,
                       holdout_ratio=holdout_ratio, train_ratio=train_ratio,
                       decorr_threshold=decorr_threshold, min_n_train=min_n_train,
                       dsr_alpha=dsr_alpha, eval_start=start, workers=workers,
                       update_library=update_library, library_universe=universe,
                       library_orthogonal=library_orthogonal, objective=objective)
