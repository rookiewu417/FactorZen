"""crypto 风格因子（对标 A 股 Barra，但换成 crypto 可得的变量）。

每个函数签名 ``(daily, _aux=None) -> [trade_date, ts_code, factor_value]``，与
``risk/exposures.compute_exposures`` 期望的注册表 fn 兼容。``daily`` 为已加派生列的
crypto daily 帧（含 close/ret_1d/amount/vol/open_interest/funding_rate）。

窗口按 crypto 高频特性取较短值（对标 A 股 252/21，crypto 用 ~30/20）。
"""
from __future__ import annotations

from collections.abc import Callable

import polars as pl

_MOM_LONG = 30
_MOM_SKIP = 3
_VOL_WIN = 20
_LIQ_WIN = 20
_BETA_WIN = 30
_FUND_WIN = 20
_MKT_SYMBOL = "BTCUSDT"  # 市场基准（β 回归标的）


def _out(df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return df.with_columns(expr.alias("factor_value")).select(["trade_date", "ts_code", "factor_value"])


def crypto_size(daily: pl.DataFrame, _aux: object = None) -> pl.DataFrame:
    """size = ln(20 日 open_interest 均值)（持仓规模代理）。"""
    d = daily.sort(["ts_code", "trade_date"])
    oi_mean = pl.col("open_interest").rolling_mean(_LIQ_WIN, min_samples=5).over("ts_code")
    return _out(d, pl.when(oi_mean > 0).then(oi_mean.log()).otherwise(None))


def crypto_liquidity(daily: pl.DataFrame, _aux: object = None) -> pl.DataFrame:
    """liquidity = ln(20 日成交额均值)（换手/流动性代理，对标 A 股换手率）。"""
    d = daily.sort(["ts_code", "trade_date"])
    amt_mean = pl.col("amount").rolling_mean(_LIQ_WIN, min_samples=5).over("ts_code")
    return _out(d, pl.when(amt_mean > 0).then(amt_mean.log()).otherwise(None))


def crypto_momentum(daily: pl.DataFrame, _aux: object = None) -> pl.DataFrame:
    """momentum = ln(close[t-skip] / close[t-long])（跳过近 skip 期，对标 Barra 12-1）。"""
    d = daily.sort(["ts_code", "trade_date"])
    num = pl.col("close").shift(_MOM_SKIP).over("ts_code")
    den = pl.col("close").shift(_MOM_LONG).over("ts_code")
    return _out(d, pl.when((num > 0) & (den > 0)).then((num / den).log()).otherwise(None))


def crypto_volatility(daily: pl.DataFrame, _aux: object = None) -> pl.DataFrame:
    """volatility = 20 日 ret_1d 滚动标准差。"""
    d = daily.sort(["ts_code", "trade_date"])
    vol = pl.col("ret_1d").rolling_std(_VOL_WIN, min_samples=5).over("ts_code")
    return _out(d, vol)


def crypto_funding_carry(daily: pl.DataFrame, _aux: object = None) -> pl.DataFrame:
    """funding_carry = 20 日 funding_rate 均值（永续资金费 carry，crypto 特有）。"""
    d = daily.sort(["ts_code", "trade_date"])
    fc = pl.col("funding_rate").rolling_mean(_FUND_WIN, min_samples=5).over("ts_code")
    return _out(d, fc)


def crypto_btc_beta(daily: pl.DataFrame, _aux: object = None) -> pl.DataFrame:
    """btc_beta = ret_1d 对 BTC ret_1d 的 30 日滚动 β = cov(r, r_mkt)/var(r_mkt)。"""
    d = daily.sort(["ts_code", "trade_date"])
    mkt = (
        d.filter(pl.col("ts_code") == _MKT_SYMBOL)
        .select(["trade_date", pl.col("ret_1d").alias("_mkt_ret")])
    )
    d = d.join(mkt, on="trade_date", how="left")
    w = _BETA_WIN
    e_xy = (pl.col("ret_1d") * pl.col("_mkt_ret")).rolling_mean(w, min_samples=5).over("ts_code")
    e_x = pl.col("ret_1d").rolling_mean(w, min_samples=5).over("ts_code")
    e_m = pl.col("_mkt_ret").rolling_mean(w, min_samples=5).over("ts_code")
    e_mm = (pl.col("_mkt_ret") ** 2).rolling_mean(w, min_samples=5).over("ts_code")
    cov = e_xy - e_x * e_m
    var_m = e_mm - e_m * e_m
    beta = pl.when(var_m > 1e-12).then(cov / var_m).otherwise(None)
    return _out(d, beta)


CRYPTO_STYLE_REGISTRY: dict[str, Callable[..., pl.DataFrame]] = {
    "size": crypto_size,
    "liquidity": crypto_liquidity,
    "momentum": crypto_momentum,
    "volatility": crypto_volatility,
    "funding_carry": crypto_funding_carry,
    "btc_beta": crypto_btc_beta,
}
CRYPTO_STYLE_NAMES: list[str] = list(CRYPTO_STYLE_REGISTRY.keys())
