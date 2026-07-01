"""crypto 挖掘入口：装配 crypto daily 帧 + 调用市场无关的 run_session。

daily 帧 = bars(OHLCV) 左连 funding_rate 与 open_interest（crypto 特有叶子），
派生列(vwap/log_vol/ret_1d)由 run_session 内部经 profile.factors.derived_columns 追加。
"""
from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.discovery.export import alpha_cross_section_from_daily
from factorzen.discovery.mining_session import run_session
from factorzen.markets.base import MarketProfile
from factorzen.markets.crypto.provider import CryptoDataProvider


def build_crypto_daily(
    provider: CryptoDataProvider, symbols: list[str], start: str, end: str, freq: str = "daily"
) -> pl.DataFrame:
    """拉 bars + funding + open_interest 并对齐成挖掘用 daily 帧。

    funding/OI 缺失填 0.0（MVP：0 funding 为中性；OI 缺失后续可换前向填充）。
    """
    bars = provider.fetch_bars(symbols, start, end, freq)
    if bars.is_empty():
        return bars
    funding = provider.fetch_funding(symbols, start, end)
    oi = provider.fetch_open_interest(symbols, start, end)
    daily = bars
    if not funding.is_empty():
        daily = daily.join(funding, on=["ts_code", "trade_date"], how="left")
    if not oi.is_empty():
        daily = daily.join(oi, on=["ts_code", "trade_date"], how="left")
    # 保证 crypto 叶子列存在且无 null（未 join 到的填 0）
    for col in ("funding_rate", "open_interest"):
        if col not in daily.columns:
            daily = daily.with_columns(pl.lit(0.0).alias(col))
        else:
            daily = daily.with_columns(pl.col(col).fill_null(0.0))
    return daily.sort(["ts_code", "trade_date"])


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
    out_dir: str = "workspace/mining_sessions",
    **session_kw: Any,
) -> dict:
    """crypto perps 因子挖掘：装配数据 → run_session(profile=crypto)。"""
    provider = profile.provider
    assert isinstance(provider, CryptoDataProvider), "run_crypto_mining 需 crypto profile"
    daily = build_crypto_daily(provider, symbols, start, end, profile.base_freq)
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


def export_crypto_alpha(
    profile: MarketProfile,
    expression: str,
    symbols: list[str],
    start: str,
    end: str,
    date: str,
) -> pl.DataFrame:
    """计算 crypto 表达式在 ``date`` 当日截面 α，返回 ``[ts_code, alpha]``。"""
    provider = profile.provider
    assert isinstance(provider, CryptoDataProvider), "export_crypto_alpha 需 crypto profile"
    daily = build_crypto_daily(provider, symbols, start, end, profile.base_freq)
    daily = profile.factors.derived_columns(daily)
    return alpha_cross_section_from_daily(
        expression, daily, date, leaf_map=profile.factors.leaf_features()
    )
