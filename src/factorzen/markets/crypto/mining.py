"""crypto 挖掘入口：装配 crypto daily 帧 + 调用市场无关的 run_session。

daily 帧 = bars(OHLCV) 左连 funding_rate 与 open_interest（crypto 特有叶子），
派生列(vwap/log_vol/ret_1d)由 run_session 内部经 profile.factors.derived_columns 追加。
"""
from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.config.settings import MINING_SESSIONS_DIR
from factorzen.discovery.export import alpha_cross_section_from_daily
from factorzen.discovery.mining_session import run_session
from factorzen.markets.base import MarketProfile


def build_crypto_daily(
    provider: Any, symbols: list[str], start: str, end: str, freq: str = "daily"
) -> pl.DataFrame:
    """拉 bars + funding + open_interest 并按 freq 对齐成挖掘帧。

    funding 缺失填 0(中性);OI:daily 保持 join+fill0 现行为,intraday 先按
    ts_code 前向填充(5 分钟粒度 OI 在细 bar 上是 asof 前值)再 fill 0。
    """
    bars = provider.fetch_bars(symbols, start, end, freq)
    if bars.is_empty():
        return bars
    funding = provider.fetch_funding(symbols, start, end, freq)
    oi = provider.fetch_open_interest(symbols, start, end, freq)
    daily = bars
    if not funding.is_empty():
        daily = daily.join(funding, on=["ts_code", "trade_date"], how="left")
    if not oi.is_empty():
        daily = daily.join(oi, on=["ts_code", "trade_date"], how="left")
    # 叶子列缺失则补 0(daily 现行为)
    for col in ("funding_rate", "open_interest"):
        if col not in daily.columns:
            daily = daily.with_columns(pl.lit(0.0).alias(col))
    daily = daily.sort(["ts_code", "trade_date"])
    if freq != "daily":  # intraday:5min OI 在细 bar 上按标的前向填充
        daily = daily.with_columns(pl.col("open_interest").forward_fill().over("ts_code"))
    return daily.with_columns(
        pl.col("funding_rate").fill_null(0.0), pl.col("open_interest").fill_null(0.0))


def run_crypto_mining(
    profile: MarketProfile,
    symbols: list[str],
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
    """crypto perps 因子挖掘：装配数据 → run_session(profile=crypto)。"""
    freq = freq or profile.base_freq
    provider = profile.provider
    assert hasattr(provider, "fetch_funding"), "run_crypto_mining 需 crypto profile(provider 缺 funding 扩展)"
    daily = build_crypto_daily(provider, symbols, start, end, freq)
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


def validate_crypto_expression(
    profile: MarketProfile,
    expression: str,
    symbols: list[str],
    start: str,
    end: str,
    freq: str | None = None,
) -> dict:
    """crypto 单表达式防过拟合验证：bootstrap IC 95%CI + Deflated Sharpe。

    返回 ``{ic_mean, ir, dsr_p, ci_lo, ci_hi, n}``（复用市场无关 ic_overfit_report）。
    """
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.scoring import ic_overfit_report

    freq = freq or profile.base_freq
    provider = profile.provider
    assert hasattr(provider, "fetch_funding"), "validate_crypto_expression 需 crypto profile(provider 缺 funding 扩展)"
    daily = build_crypto_daily(provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    leaf_map = profile.factors.leaf_features()
    node = parse_expr(expression, leaf_map)
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    factor_df = sorted_daily.with_columns(
        evaluate_materialized(node, sorted_daily, leaf_map).alias("factor_value")
    ).select(["trade_date", "ts_code", "factor_value"])
    return ic_overfit_report(factor_df, daily)


def export_crypto_alpha(
    profile: MarketProfile,
    expression: str,
    symbols: list[str],
    start: str,
    end: str,
    date: str,
    freq: str | None = None,
) -> pl.DataFrame:
    """计算 crypto 表达式在 ``date`` 当日截面 α，返回 ``[ts_code, alpha]``。

    注:``freq`` 透传给 build_crypto_daily;intraday 帧下 ``date`` 截面语义未定义
    (alpha_cross_section_from_daily 按当日 Date 匹配),intraday α 导出为已知限制。
    """
    freq = freq or profile.base_freq
    provider = profile.provider
    assert hasattr(provider, "fetch_funding"), "export_crypto_alpha 需 crypto profile(provider 缺 funding 扩展)"
    daily = build_crypto_daily(provider, symbols, start, end, freq)
    daily = profile.factors.derived_columns(daily)
    return alpha_cross_section_from_daily(
        expression, daily, date, leaf_map=profile.factors.leaf_features()
    )
