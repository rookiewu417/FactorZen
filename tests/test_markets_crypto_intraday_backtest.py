"""intraday NAV 回测:手算 ground truth(收益/funding 逐 bar)+ 信号键上抛。"""
from datetime import date, datetime

import polars as pl
import pytest

from factorzen.markets.crypto.backtest import _coerce_signal_keys, simulate_crypto_nav
from factorzen.markets.crypto.costs import CryptoCostModel


def test_coerce_signal_keys_upcasts_date_for_intraday():
    w = pl.DataFrame({"ts_code": ["BTCUSDT"], "target_weight": [1.0]})
    out = _coerce_signal_keys({date(2026, 5, 1): w}, "1h")
    assert list(out.keys()) == [datetime(2026, 5, 1, 0, 0)]
    same = _coerce_signal_keys({date(2026, 5, 1): w}, "daily")
    assert list(same.keys()) == [date(2026, 5, 1)]  # daily 不动


def test_simulate_nav_hourly_ground_truth():
    # 单标的满仓多头,3 根 1h bar:100→110→99;第 2 根 bar 落 0.001 funding
    ts = [datetime(2026, 5, 1, h) for h in (0, 1, 2)]
    daily = pl.DataFrame({"ts_code": ["BTCUSDT"] * 3, "trade_date": ts,
                          "close": [100.0, 110.0, 99.0]})
    funding = pl.DataFrame({"ts_code": ["BTCUSDT"], "trade_date": [ts[1]],
                            "funding_rate": [0.001]})
    w = {ts[0]: pl.DataFrame({"ts_code": ["BTCUSDT"], "target_weight": [1.0]})}
    res = simulate_crypto_nav(w, daily, funding,
                              cost_model=CryptoCostModel(taker=0.0, slippage=0.0),
                              periods_per_year=8760)
    nets = res["nav"]["net_return"].to_list()
    # bar0=信号日无持仓;bar1: +10% - 0.001 funding = 0.099;bar2: -10%
    assert nets[0] == pytest.approx(0.0)
    assert nets[1] == pytest.approx(0.10 - 0.001)
    assert nets[2] == pytest.approx(-0.10)
    assert res["metrics"]["total_funding"] == pytest.approx(0.001)  # 仅 bar1 计提
