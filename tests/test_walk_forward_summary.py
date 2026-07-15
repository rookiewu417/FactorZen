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
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    factor_df, price_df = _make_factor_price(n_dates=12)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        walk_forward={
            "enabled": True,
            "train_days": 20,
            "test_days": 5,
            "step_days": 5,
            "embargo_days": 2,
        },
    )

    summary, result = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert result is None
    assert summary["status"] == "insufficient_data"
    assert summary["n_folds"] == 0
    assert summary["requested_n_trials"] == 50
    assert summary["param_candidates"][-1] == {"top_n": 50}


def test_walk_forward_summary_returns_oos_metrics_when_folds_exist():
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    factor_df, price_df = _make_factor_price(n_dates=36, n_stocks=80)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240228",
        backtest={"quantiles": 4},
        walk_forward={
            "enabled": True,
            "train_days": 12,
            "test_days": 6,
            "step_days": 6,
            "embargo_days": 1,
        },
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


def test_walk_forward_summary_uses_top_n_candidates_from_n_trials(monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.backtest import PrecomputedWeightsStrategy
    from factorzen.daily.evaluation.walk_forward import WalkForwardResult
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    captured = {}

    def fake_run_walk_forward_search(**kwargs):
        captured["param_candidates"] = kwargs["param_candidates"]
        captured["strategy"] = kwargs["strategy_factory"]({"top_n": 10})
        return WalkForwardResult(
            folds=[],
            oos_returns=pl.DataFrame(),
            is_sharpe_mean=0.0,
            oos_sharpe_mean=0.0,
            oos_sharpe_std=0.0,
            oos_max_dd=0.0,
            stability_ratio=0.0,
        )

    monkeypatch.setattr(
        "factorzen.daily.evaluation.walk_forward_summary.run_walk_forward_search",
        fake_run_walk_forward_search,
    )
    factor_df, price_df = _make_factor_price(n_dates=12)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        backtest={"top_n": 10},
        walk_forward={"enabled": True, "n_trials": 4},
    )

    summary, _ = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert captured["param_candidates"] == [{"top_n": 10}]
    assert isinstance(captured["strategy"], PrecomputedWeightsStrategy)
    assert summary["requested_n_trials"] == 4


def test_walk_forward_summary_skips_runner_when_disabled(monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    def unexpected_runner(**_kwargs):
        raise AssertionError("disabled walk-forward must not invoke the runner")

    monkeypatch.setattr(
        "factorzen.daily.evaluation.walk_forward_summary.run_walk_forward_search",
        unexpected_runner,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
    )

    summary, result = run_quantile_walk_forward_summary(
        pl.DataFrame(),
        pl.DataFrame(),
        cfg,
        factor_name="momentum_20d",
    )

    assert summary == {"status": "disabled", "n_folds": 0}
    assert result is None


def test_walk_forward_optimized_path_matches_sequential_search():
    from factorzen.daily.evaluation.backtest import BacktestConfig, TopNLongOnlyStrategy
    from factorzen.daily.evaluation.walk_forward import WalkForwardSplitter, run_walk_forward_search

    factor_df, price_df = _make_factor_price(n_dates=32, n_stocks=40)
    splitter = WalkForwardSplitter(train_days=10, test_days=5, step_days=5, embargo_days=1)
    candidates = [{"top_n": 10}, {"top_n": 20}]
    cfg = BacktestConfig(max_abs_weight=0.05, max_participation_rate=1.0)

    def strategy_factory(params):
        return TopNLongOnlyStrategy(n=params["top_n"])

    sequential = run_walk_forward_search(
        strategy_factory=strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        param_candidates=candidates,
        config=cfg,
        factor_name="x",
        reuse_is_backtests=False,
        parallel_workers=1,
    )
    optimized = run_walk_forward_search(
        strategy_factory=strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        param_candidates=candidates,
        config=cfg,
        factor_name="x",
        reuse_is_backtests=True,
        parallel_workers=2,
    )

    assert optimized.oos_returns.equals(sequential.oos_returns)
    assert optimized.is_sharpe_mean == sequential.is_sharpe_mean
    assert optimized.oos_sharpe_mean == sequential.oos_sharpe_mean
    assert optimized.oos_sharpe_std == sequential.oos_sharpe_std
    assert optimized.oos_max_dd == sequential.oos_max_dd
    assert [fold.params for fold in optimized.folds] == [fold.params for fold in sequential.folds]
