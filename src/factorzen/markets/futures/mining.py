"""期货挖掘入口：装配主力连续后复权帧 + 调用市场无关的 run_session。

daily 帧 = provider.fetch_bars（主力连续后复权 OHLC + 原始 vol/amount/oi + adj_factor +
mapping_ts_code）；派生列(vwap/log_vol/ret_1d/oi_chg)由 run_session 内部经
profile.factors.derived_columns 追加。与 crypto mining.py 同构（消除双路径漂移）。
"""
from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.discovery.export import alpha_cross_section_from_daily
from factorzen.discovery.mining_session import run_session
from factorzen.markets.base import MarketProfile


def build_futures_daily(
    provider: Any, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
) -> pl.DataFrame:
    """拉主力连续后复权帧（fetch_bars 已完成拼接+后复权，此处仅透传）。"""
    return provider.fetch_bars(symbols, start, end, freq)


def run_futures_mining(
    profile: MarketProfile,
    symbols: list[str] | None,
    start: str,
    end: str,
    *,
    n_trials: int,
    top_k: int,
    seed: int,
    method: str = "random",
    freq: str | None = None,
    out_dir: str = "workspace/mining_sessions",
    **session_kw: Any,
) -> dict:
    """商品期货因子挖掘：装配主力连续帧 → run_session(profile=futures)。"""
    freq = freq or profile.base_freq
    daily = build_futures_daily(profile.provider, symbols, start, end, freq)
    return run_session(
        daily,
        n_trials=n_trials,
        top_k=top_k,
        seed=seed,
        method=method,
        out_dir=out_dir,
        profile=profile,
        **session_kw,
    )


def validate_futures_expression(
    profile: MarketProfile,
    expression: str,
    symbols: list[str] | None,
    start: str,
    end: str,
    freq: str | None = None,
) -> dict:
    """单表达式防过拟合验证：bootstrap IC 95%CI + Deflated Sharpe（复用市场无关 ic_overfit_report）。"""
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.scoring import ic_overfit_report

    freq = freq or profile.base_freq
    daily = build_futures_daily(profile.provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    leaf_map = profile.factors.leaf_features()
    node = parse_expr(expression, leaf_map)
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    factor_df = sorted_daily.with_columns(
        evaluate_materialized(node, sorted_daily, leaf_map).alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"])
    return ic_overfit_report(factor_df, daily)


def export_futures_alpha(
    profile: MarketProfile,
    expression: str,
    symbols: list[str] | None,
    start: str,
    end: str,
    date: str,
    freq: str | None = None,
) -> pl.DataFrame:
    """计算期货表达式在 ``date`` 当日截面 α，返回 ``[ts_code, alpha]``。"""
    freq = freq or profile.base_freq
    daily = build_futures_daily(profile.provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    return alpha_cross_section_from_daily(
        expression, daily, date, leaf_map=profile.factors.leaf_features()
    )
