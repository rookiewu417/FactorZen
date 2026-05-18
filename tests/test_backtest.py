"""daily/evaluation/backtest.py 的单元测试。"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from daily.evaluation.backtest import BacktestResult, run_stratified_backtest


def _make_factor_price(n_dates: int = 60, n_stocks: int = 50, seed: int = 42):
    """生成合成因子+价格 DataFrame。"""
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]

    rows_factor = []
    rows_price = []
    last_close = {s: 10.0 + i for i, s in enumerate(stocks)}
    for idx, d in enumerate(dates):
        for s in stocks:
            if idx < n_dates - 1:
                rows_factor.append(
                    {
                        "trade_date": d,
                        "ts_code": s,
                        "factor_clean": float(rng.standard_normal()),
                    }
                )
            open_price = last_close[s] * (1.0 + float(rng.normal(0, 0.002)))
            close_price = open_price * (1.0 + float(rng.normal(0, 0.01)))
            rows_price.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "open": open_price,
                    "close": close_price,
                    "pre_close": last_close[s],
                    "pct_chg": (close_price / last_close[s] - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1_000_000.0,
                }
            )
            last_close[s] = close_price

    return pl.DataFrame(rows_factor), pl.DataFrame(rows_price)


def test_returns_backtest_result():
    factor_df, price_df = _make_factor_price()
    result = run_stratified_backtest(factor_df, price_df)
    assert isinstance(result, BacktestResult)


def test_factor_name_passed_through():
    factor_df, price_df = _make_factor_price()
    result = run_stratified_backtest(factor_df, price_df, factor_name="momentum")
    assert result.factor_name == "momentum"


def test_default_factor_name_is_empty():
    factor_df, price_df = _make_factor_price()
    result = run_stratified_backtest(factor_df, price_df)
    assert result.factor_name == ""


def test_summary_stats_has_portfolio_and_long_short():
    factor_df, price_df = _make_factor_price()
    n_groups = 5
    result = run_stratified_backtest(factor_df, price_df, n_groups=n_groups)
    assert "portfolio" in result.summary_stats
    assert "long_short" in result.summary_stats
    assert result.n_groups == n_groups


def test_nav_starts_near_one():
    factor_df, price_df = _make_factor_price()
    result = run_stratified_backtest(factor_df, price_df, n_groups=5)
    first_nav = result.nav.sort("trade_date")["nav"][0]
    assert first_nav == 1.0


def test_annual_return_is_finite():
    factor_df, price_df = _make_factor_price()
    result = run_stratified_backtest(factor_df, price_df, n_groups=5)
    for stats in result.summary_stats.values():
        assert np.isfinite(stats["ann_ret"])
        assert np.isfinite(stats["ann_vol"])


def test_summary_string_is_non_empty():
    factor_df, price_df = _make_factor_price()
    result = run_stratified_backtest(factor_df, price_df, n_groups=5)
    text = result.summary()
    assert "Portfolio" in text
    assert len(text) > 10
