"""Strategy backtest engine behavior tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from daily.evaluation.backtest import (
    BacktestConfig,
    BacktestContext,
    CostModel,
    FactorWeightedStrategy,
    OptimizerStrategy,
    QuantileLongShortStrategy,
    Strategy,
    StrategyBacktestResult,
    TopNLongOnlyStrategy,
    run_strategy_backtest,
)
from daily.evaluation.cost_models import SquareRootImpactCostModel


def _prices(
    *,
    day2_vol: float = 1000.0,
    day2_pct: float = 0.0,
    day2_amount: float = 1_000_000.0,
    day3_pct: float = 0.0,
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": day2_pct,
                "vol": day2_vol,
                "amount": day2_amount,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 11.0,
                "close": 12.0,
                "pre_close": 11.0,
                "pct_chg": day3_pct,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )


def _factor(values: list[tuple[date, str, float]] | None = None) -> pl.DataFrame:
    if values is None:
        values = [(date(2024, 1, 1), "000001.SZ", 1.0)]
    return pl.DataFrame(
        [
            {"trade_date": d, "ts_code": code, "factor_clean": value}
            for d, code, value in values
        ]
    )


class BuyOneStrategy(Strategy):
    name = "buy_one"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        assert context.signal_date < context.execution_date
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [1.0]})


class BadMissingColumnStrategy(Strategy):
    name = "bad_missing"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame({"ts_code": ["000001.SZ"], "weight": [1.0]})


class BadDuplicateStrategy(Strategy):
    name = "bad_duplicate"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame(
            {"ts_code": ["000001.SZ", "000001.SZ"], "target_weight": [0.5, 0.5]}
        )


class BadNaNStrategy(Strategy):
    name = "bad_nan"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [float("nan")]})


class ExitOnSecondSignalStrategy(Strategy):
    name = "exit_on_second_signal"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        target = 1.0 if context.signal_date == date(2024, 1, 1) else 0.0
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [target]})


def test_custom_strategy_runs_and_outputs_required_frames():
    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        _prices(),
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    assert isinstance(result, StrategyBacktestResult)
    assert {"gross_return", "cost", "borrow_cost", "net_return", "nav", "cash_weight"}.issubset(
        set(result.nav.columns)
    )
    assert {"weight", "market_value"}.issubset(set(result.positions.columns))
    assert {"prev_weight", "target_weight", "filled_delta_weight", "block_reason"}.issubset(
        set(result.trades.columns)
    )


def test_next_open_execution_ignores_same_day_return_and_starts_at_one():
    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        _prices(),
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    first_nav = result.nav.sort("trade_date")["nav"][0]
    second_nav = result.nav.sort("trade_date")["nav"][1]
    assert first_nav == pytest.approx(1.0)
    assert second_nav == pytest.approx(1.1)
    assert result.ret_definition == "open_to_close_with_overnight_carry"


def test_strategy_output_validation_errors_are_clear():
    with pytest.raises(ValueError, match="target_weight"):
        run_strategy_backtest(BadMissingColumnStrategy(), _factor(), _prices())
    with pytest.raises(ValueError, match="duplicate"):
        run_strategy_backtest(BadDuplicateStrategy(), _factor(), _prices())
    with pytest.raises(ValueError, match="finite"):
        run_strategy_backtest(BadNaNStrategy(), _factor(), _prices())


def test_suspended_stock_blocks_trade():
    result = run_strategy_backtest(BuyOneStrategy(), _factor(), _prices(day2_vol=0.0))
    trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.0)
    assert trade["block_reason"] == "suspended"


def test_limit_up_blocks_buy_but_allows_sell():
    result = run_strategy_backtest(BuyOneStrategy(), _factor(), _prices(day2_pct=9.9))
    buy_trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
    assert buy_trade["filled_delta_weight"] == pytest.approx(0.0)
    assert buy_trade["block_reason"] == "limit_up"


def test_limit_down_blocks_sell():
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
        ]
    )
    result = run_strategy_backtest(
        ExitOnSecondSignalStrategy(),
        factors,
        _prices(day3_pct=-9.9),
        config=BacktestConfig(max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    sell_trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
    assert sell_trade["filled_delta_weight"] == pytest.approx(0.0)
    assert sell_trade["block_reason"] == "limit_down"


def test_capacity_constraint_partially_fills_trade():
    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        _prices(day2_amount=100.0),
        config=BacktestConfig(initial_capital=100.0, max_participation_rate=0.1),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )
    trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.1)
    assert trade["block_reason"] == "capacity"


def test_cost_model_reduces_nav():
    config = BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0)
    free = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        _prices(),
        config=config,
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )
    costly = run_strategy_backtest(
        BuyOneStrategy(), _factor(), _prices(), config=config, cost_model=CostModel()
    )
    assert costly.nav["nav"][1] < free.nav["nav"][1]


def test_square_root_impact_uses_history_adv_for_trade_cost():
    price_rows = []
    for i, d in enumerate([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]):
        price_rows.extend(
            [
                {
                    "trade_date": d,
                    "ts_code": "000001.SZ",
                    "open": 10.0,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "pct_chg": 0.0,
                    "vol": 1000.0,
                    "amount": 1_000_000.0 if i == 0 else 100_000_000.0,
                },
                {
                    "trade_date": d,
                    "ts_code": "000002.SZ",
                    "open": 10.0,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "pct_chg": 0.0,
                    "vol": 1000.0,
                    "amount": 100_000_000.0,
                },
            ]
        )
    factor = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 1), "000002.SZ", 1.0),
        ]
    )

    class BuyBothStrategy(Strategy):
        name = "buy_both"

        def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "target_weight": [0.5, 0.5],
                }
            )

    result = run_strategy_backtest(
        BuyBothStrategy(),
        factor,
        pl.DataFrame(price_rows),
        config=BacktestConfig(max_participation_rate=1.0),
        cost_model=SquareRootImpactCostModel(alpha=0.1, fallback_adv=10_000_000.0),
    )

    trades = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).sort("ts_code")
    low_adv_cost = trades.filter(pl.col("ts_code") == "000001.SZ")["cost"][0]
    high_adv_cost = trades.filter(pl.col("ts_code") == "000002.SZ")["cost"][0]
    assert low_adv_cost > high_adv_cost


def test_quantile_long_short_strategy_selects_top_and_bottom_groups():
    factor = _factor(
        [
            (date(2024, 1, 1), "A", -2.0),
            (date(2024, 1, 1), "B", -1.0),
            (date(2024, 1, 1), "C", 1.0),
            (date(2024, 1, 1), "D", 2.0),
        ]
    )
    ctx = BacktestContext(
        signal_date=date(2024, 1, 1),
        execution_date=date(2024, 1, 2),
        factor_slice=factor,
        price_slice=pl.DataFrame(),
        current_positions=pl.DataFrame(),
        factor_col="factor_clean",
    )

    weights = QuantileLongShortStrategy(n_groups=2).generate_weights(ctx)
    assert dict(zip(weights["ts_code"], weights["target_weight"], strict=True)) == {
        "A": -0.5,
        "B": -0.5,
        "C": 0.5,
        "D": 0.5,
    }


def test_topn_long_only_strategy_weights_top_names_equally():
    factor = _factor(
        [
            (date(2024, 1, 1), "A", 1.0),
            (date(2024, 1, 1), "B", 3.0),
            (date(2024, 1, 1), "C", 2.0),
        ]
    )
    ctx = BacktestContext(
        signal_date=date(2024, 1, 1),
        execution_date=date(2024, 1, 2),
        factor_slice=factor,
        price_slice=pl.DataFrame(),
        current_positions=pl.DataFrame(),
        factor_col="factor_clean",
    )

    weights = TopNLongOnlyStrategy(n=2).generate_weights(ctx)
    assert dict(zip(weights["ts_code"], weights["target_weight"], strict=True)) == {
        "B": 0.5,
        "C": 0.5,
    }


def test_factor_weighted_strategy_supports_long_only_and_long_short():
    factor = _factor(
        [
            (date(2024, 1, 1), "A", -1.0),
            (date(2024, 1, 1), "B", 0.0),
            (date(2024, 1, 1), "C", 2.0),
        ]
    )
    ctx = BacktestContext(
        signal_date=date(2024, 1, 1),
        execution_date=date(2024, 1, 2),
        factor_slice=factor,
        price_slice=pl.DataFrame(),
        current_positions=pl.DataFrame(),
        factor_col="factor_clean",
    )

    long_only = FactorWeightedStrategy(long_only=True).generate_weights(ctx)
    long_short = FactorWeightedStrategy(long_only=False).generate_weights(ctx)

    assert long_only["target_weight"].min() >= 0
    assert long_only["target_weight"].sum() == pytest.approx(1.0)
    assert long_short["target_weight"].abs().sum() == pytest.approx(2.0)
    assert long_short["target_weight"].sum() == pytest.approx(0.0)


def test_optimizer_strategy_end_to_end():
    """OptimizerStrategy 端到端回测跑通，结果结构正确。"""
    from daily.optimization.mean_variance import MeanVarianceOptimizer

    # Build a multi-stock, multi-day dataset
    stocks = ["000001.SZ", "000002.SZ", "000003.SZ"]
    dates = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]
    price_rows = []
    for d in dates:
        for code in stocks:
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "open": 10.0,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "pct_chg": 0.0,
                    "vol": 1000.0,
                    "amount": 1_000_000.0,
                }
            )
    factor_rows = []
    for d in dates[:-1]:  # factor up to second-to-last date
        for j, code in enumerate(stocks):
            factor_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "factor_clean": float(j + 1),
                }
            )

    factor_df = pl.DataFrame(factor_rows)
    price_df = pl.DataFrame(price_rows)

    optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
    strategy = OptimizerStrategy(
        optimizer=optimizer,
        lookback_days=10,
        long_only=True,
        top_n=20,
    )
    cfg = BacktestConfig(max_abs_weight=0.5, max_gross_exposure=1.0, max_participation_rate=1.0)
    result = run_strategy_backtest(strategy, factor_df, price_df, config=cfg, factor_name="test")

    assert result.strategy_name == "optimizer_strategy"
    assert result.returns.height > 0
    assert "net_return" in result.returns.columns
    assert np.all(np.isfinite(result.returns["net_return"].to_numpy()))
