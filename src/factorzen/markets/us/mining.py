"""美股挖掘入口：装配 Yahoo 后复权日线帧 + 调用市场无关的 run_session。

daily 帧 = provider.fetch_bars（后复权 OHLC + 原始 vol + amount + adj_factor）；派生列
(vwap/log_vol/ret_1d)由 run_session 内部经 profile.factors.derived_columns 追加。
与 crypto/futures mining.py 同构（消除双路径漂移）。
"""
from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.config.settings import MINING_SESSIONS_DIR
from factorzen.discovery.export import alpha_cross_section_from_daily
from factorzen.discovery.mining_session import run_session
from factorzen.markets.base import MarketProfile


def build_us_daily(
    provider: Any, symbols: list[str] | None, start: str, end: str, freq: str = "daily"
) -> pl.DataFrame:
    """拉 Yahoo 后复权日线帧（fetch_bars 已完成复权，此处仅透传）。"""
    return provider.fetch_bars(symbols, start, end, freq)


def run_us_mining(
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
    out_dir: str = str(MINING_SESSIONS_DIR),
    **session_kw: Any,
) -> dict:
    """美股因子挖掘：装配后复权日线帧 → run_session(profile=us)。"""
    freq = freq or profile.base_freq
    daily = build_us_daily(profile.provider, symbols, start, end, freq)
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


def validate_us_expression(
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
    daily = build_us_daily(profile.provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    leaf_map = profile.factors.leaf_features()
    node = parse_expr(expression, leaf_map)
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    factor_df = sorted_daily.with_columns(
        evaluate_materialized(node, sorted_daily, leaf_map).alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"])
    return ic_overfit_report(factor_df, daily)


def export_us_alpha(
    profile: MarketProfile,
    expression: str,
    symbols: list[str] | None,
    start: str,
    end: str,
    date: str,
    freq: str | None = None,
) -> pl.DataFrame:
    """计算美股表达式在 ``date`` 当日截面 α，返回 ``[ts_code, alpha]``。"""
    freq = freq or profile.base_freq
    daily = build_us_daily(profile.provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    return alpha_cross_section_from_daily(
        expression, daily, date, leaf_map=profile.factors.leaf_features()
    )
