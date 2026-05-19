"""Tests for single-factor walk-forward summary integration."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_factor_price(n_dates: int = 40, n_stocks: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]
    factor_rows = []
    price_rows = []
    last_close = {code: 10.0 + idx for idx, code in enumerate(stocks)}
    for i, d in enumerate(dates):
        for code in stocks:
            if i < n_dates - 1:
                factor_rows.append(
                    {
                        "trade_date": d,
                        "ts_code": code,
                        "factor_clean": float(rng.normal()),
                    }
                )
            open_price = last_close[code] * (1.0 + float(rng.normal(0, 0.001)))
            close_price = open_price * (1.0 + float(rng.normal(0, 0.01)))
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "open": open_price,
                    "close": close_price,
                    "pre_close": last_close[code],
                    "pct_chg": (close_price / last_close[code] - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1e9,
                }
            )
            last_close[code] = close_price
    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)


def test_walk_forward_summary_marks_insufficient_data():
    from common.config_loader import RunConfig
    from daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    factor_df, price_df = _make_factor_price(n_dates=12)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        walk_forward={"train_days": 20, "test_days": 5, "step_days": 5, "embargo_days": 2},
    )

    summary, result = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert result is None
    assert summary == {"status": "insufficient_data", "n_folds": 0}


def test_walk_forward_summary_returns_oos_metrics_when_folds_exist():
    from common.config_loader import RunConfig
    from daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    factor_df, price_df = _make_factor_price(n_dates=36, n_stocks=80)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240228",
        backtest={"quantiles": 4},
        walk_forward={"train_days": 12, "test_days": 6, "step_days": 6, "embargo_days": 1},
    )

    summary, result = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert result is not None
    assert summary["status"] == "ok"
    assert summary["n_folds"] > 0
    assert summary["is_sharpe_mean"] == result.is_sharpe_mean
    assert summary["oos_sharpe_mean"] == result.oos_sharpe_mean
    assert summary["oos_sharpe_std"] == result.oos_sharpe_std
    assert summary["oos_max_dd"] == result.oos_max_dd
    assert summary["stability_ratio"] == result.stability_ratio
