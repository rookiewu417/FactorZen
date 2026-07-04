"""MC0 Task 6: crypto FactorSet（叶子字典 + 派生列）。"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from factorzen.markets.base import FactorSet
from factorzen.markets.crypto.factors import CryptoFactorSet


def test_is_a_factorset():
    assert isinstance(CryptoFactorSet(), FactorSet)


def test_leaf_features_include_crypto_specific():
    fs = CryptoFactorSet()
    leaves = fs.leaf_features()
    # 价量叶子
    for name in ["close", "open", "high", "low", "vol", "amount", "vwap", "log_vol", "ret_1d"]:
        assert name in leaves
    # crypto 无复权：close 直接映射 close（非 close_adj）
    assert leaves["close"] == "close"
    # crypto 特有叶子
    assert "funding_rate" in leaves and "open_interest" in leaves


def test_basic_features():
    fs = CryptoFactorSet()
    assert fs.basic_features() == {"funding_rate", "open_interest"}


def test_derived_columns_ground_truth():
    fs = CryptoFactorSet()
    bars = pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 3 + ["ETHUSDT"] * 3,
        "trade_date": [date(2024, 1, i) for i in (1, 2, 3)] * 2,
        "close": [100.0, 110.0, 105.0, 50.0, 55.0, 60.0],
        "vol": [10.0, 20.0, 5.0, 100.0, 50.0, 80.0],
        "amount": [1000.0, 2200.0, 525.0, 5000.0, 2750.0, 4800.0],
    })
    out = fs.derived_columns(bars).sort(["ts_code", "trade_date"])
    btc = out.filter(pl.col("ts_code") == "BTCUSDT").sort("trade_date")
    # vwap = amount / vol
    assert btc["vwap"].to_list() == [100.0, 110.0, 105.0]
    # ret_1d = close.pct_change（首行 null）
    assert btc["ret_1d"][0] is None
    assert abs(btc["ret_1d"][1] - 0.1) < 1e-12
    assert abs(btc["ret_1d"][2] - (105.0 / 110.0 - 1)) < 1e-12
    # log_vol = ln(vol)，对拍 numpy
    np.testing.assert_allclose(btc["log_vol"].to_list(), np.log([10.0, 20.0, 5.0]), rtol=1e-12)
    # ret_1d 不跨标的泄漏：ETH 首行也是 null
    eth = out.filter(pl.col("ts_code") == "ETHUSDT").sort("trade_date")
    assert eth["ret_1d"][0] is None
    assert math.isclose(eth["ret_1d"][1], 0.1, rel_tol=1e-12)


def test_taker_buy_ratio_derived_and_guarded():
    fs = CryptoFactorSet()
    assert "taker_buy_ratio" in fs.leaf_features()
    bars = pl.DataFrame({
        "ts_code": ["BTCUSDT"] * 2, "trade_date": [1, 2],
        "close": [1.0, 2.0], "vol": [10.0, 0.0], "amount": [10.0, 0.0],
        "taker_buy_volume": [6.0, 3.0],
    })
    out = fs.derived_columns(bars)
    assert out["taker_buy_ratio"].to_list()[0] == 0.6
    assert out["taker_buy_ratio"][1] is None  # vol=0 → null 不除零


def test_taker_buy_ratio_null_when_source_missing():
    bars = pl.DataFrame({"ts_code": ["BTCUSDT"], "trade_date": [1],
                         "close": [1.0], "vol": [1.0], "amount": [1.0]})
    out = CryptoFactorSet().derived_columns(bars)  # ccxt 旧路径无 taker_buy_volume
    assert out["taker_buy_ratio"].to_list() == [None]
