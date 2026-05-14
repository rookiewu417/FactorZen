"""daily/evaluation/backtest.py 的单元测试。"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from daily.evaluation.backtest import BacktestResult, run_stratified_backtest


def _make_factor_ret(n_dates: int = 60, n_stocks: int = 50, seed: int = 42):
    """生成合成因子+收益 DataFrame。"""
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1 + i % 28) for i in range(n_dates)]
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]

    rows_factor = []
    rows_ret = []
    for d in dates:
        for s in stocks:
            rows_factor.append({
                "trade_date": d,
                "ts_code": s,
                "factor_clean": float(rng.standard_normal()),
            })
            rows_ret.append({
                "trade_date": d,
                "ts_code": s,
                "ret": float(rng.standard_normal() * 0.01),
            })

    return pl.DataFrame(rows_factor), pl.DataFrame(rows_ret)


def test_returns_backtest_result():
    factor_df, ret_df = _make_factor_ret()
    result = run_stratified_backtest(factor_df, ret_df)
    assert isinstance(result, BacktestResult)


def test_factor_name_passed_through():
    factor_df, ret_df = _make_factor_ret()
    result = run_stratified_backtest(factor_df, ret_df, factor_name="momentum")
    assert result.factor_name == "momentum"


def test_default_factor_name_is_empty():
    factor_df, ret_df = _make_factor_ret()
    result = run_stratified_backtest(factor_df, ret_df)
    assert result.factor_name == ""


def test_summary_stats_has_all_groups():
    factor_df, ret_df = _make_factor_ret()
    n_groups = 5
    result = run_stratified_backtest(factor_df, ret_df, n_groups=n_groups)
    for g in range(n_groups):
        assert g in result.summary_stats
    assert "long_short" in result.summary_stats


def test_nav_starts_near_one():
    factor_df, ret_df = _make_factor_ret()
    result = run_stratified_backtest(factor_df, ret_df, n_groups=5)
    # 第一个日期的 nav 应接近 (1 + first_ret)
    first_nav = result.nav.sort("trade_date").head(5)["nav"].to_numpy()
    assert all(0.5 < v < 2.0 for v in first_nav)


def test_annual_return_is_finite():
    factor_df, ret_df = _make_factor_ret()
    result = run_stratified_backtest(factor_df, ret_df, n_groups=5)
    for stats in result.summary_stats.values():
        assert np.isfinite(stats["ann_ret"])
        assert np.isfinite(stats["ann_vol"])


def test_summary_string_is_non_empty():
    factor_df, ret_df = _make_factor_ret()
    result = run_stratified_backtest(factor_df, ret_df, n_groups=5)
    text = result.summary()
    assert "Long-Short" in text
    assert len(text) > 10
