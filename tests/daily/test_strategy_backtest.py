"""Strategy backtest engine behavior tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from factorzen.daily.evaluation.backtest import (
    BacktestConfig,
    BacktestContext,
    CostModel,
    FactorWeightedStrategy,
    OptimizerStrategy,
    PrecomputedWeightsStrategy,
    QuantileLongShortStrategy,
    Strategy,
    StrategyBacktestResult,
    TopNLongOnlyStrategy,
    _compute_adv_20d,
    _precompute_adv_20d_by_date,
    _run_precomputed_weights_backtest_fast,
    _summary_stats,
    precompute_top_n_weights,
    run_strategy_backtest,
    trim_backtest_to_first_trade,
)
from factorzen.daily.evaluation.cost_models import SquareRootImpactCostModel


def _prices(
    *,
    day1_amount: float = 1_000_000.0,
    day2_open: float | None = 10.0,
    day2_vol: float = 1000.0,
    day2_pct: float = 0.0,
    day2_amount: float = 1_000_000.0,
    day3_open: float = 11.0,
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
                "amount": day1_amount,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": day2_open,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": day2_pct,
                "vol": day2_vol,
                "amount": day2_amount,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": day3_open,
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
        [{"trade_date": d, "ts_code": code, "factor_clean": value} for d, code, value in values]
    )


def test_precomputed_adv_matches_legacy_per_day_calculation():
    prices = pl.DataFrame(
        [
            {"trade_date": date(2024, 1, 1), "ts_code": "000001.SZ", "amount": 100.0},
            {"trade_date": date(2024, 1, 1), "ts_code": "000002.SZ", "amount": 50.0},
            {"trade_date": date(2024, 1, 2), "ts_code": "000001.SZ", "amount": 300.0},
            {"trade_date": date(2024, 1, 2), "ts_code": "000002.SZ", "amount": 0.0},
            {"trade_date": date(2024, 1, 3), "ts_code": "000001.SZ", "amount": 500.0},
            {"trade_date": date(2024, 1, 3), "ts_code": "000002.SZ", "amount": 150.0},
        ]
    )
    trade_dates = prices.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()

    adv_by_date = _precompute_adv_20d_by_date(prices, trade_dates)

    for idx, trade_date in enumerate(trade_dates):
        assert adv_by_date.get(trade_date, {}) == _compute_adv_20d(prices, trade_dates, idx)


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
        return pl.DataFrame({"ts_code": ["000001.SZ", "000001.SZ"], "target_weight": [0.5, 0.5]})


class BadNaNStrategy(Strategy):
    name = "bad_nan"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [float("nan")]})


class ExitOnSecondSignalStrategy(Strategy):
    name = "exit_on_second_signal"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        target = 1.0 if context.signal_date == date(2024, 1, 1) else 0.0
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [target]})


class CaptureCurrentPositionsStrategy(Strategy):
    name = "capture_current_positions"

    def __init__(self) -> None:
        self.captured_positions: pl.DataFrame | None = None

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        if context.execution_date == date(2024, 1, 3):
            self.captured_positions = context.current_positions
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [1.0]})


class ExitAfterCostNeutralEntryStrategy(Strategy):
    name = "exit_after_cost_neutral_entry"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        target = 1.0 / 1.01 if context.signal_date == date(2024, 1, 1) else 0.0
        return pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [target]})


class EqualTwoStockStrategy(Strategy):
    name = "equal_two_stock"

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "target_weight": [0.5, 0.5],
            }
        )


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


def test_backtest_lightweight_outputs_match_full_returns_and_summary():
    full = run_strategy_backtest(
        BuyOneStrategy(),
        _factor([(date(2024, 1, 1), "000001.SZ", 1.0), (date(2024, 1, 2), "000001.SZ", 1.0)]),
        _prices(),
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )
    light = run_strategy_backtest(
        BuyOneStrategy(),
        _factor([(date(2024, 1, 1), "000001.SZ", 1.0), (date(2024, 1, 2), "000001.SZ", 1.0)]),
        _prices(),
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    assert light.returns.equals(full.returns)
    assert light.nav.equals(full.nav)
    assert light.summary_stats == full.summary_stats
    assert light.positions.is_empty()
    assert light.trades.is_empty()


def test_precomputed_top_n_weights_match_top_n_strategy_backtest():
    factors = pl.DataFrame(
        [
            {"trade_date": date(2024, 1, 1), "ts_code": "000001.SZ", "factor_clean": 3.0},
            {"trade_date": date(2024, 1, 1), "ts_code": "000002.SZ", "factor_clean": 2.0},
            {"trade_date": date(2024, 1, 1), "ts_code": "000003.SZ", "factor_clean": 1.0},
            {"trade_date": date(2024, 1, 2), "ts_code": "000001.SZ", "factor_clean": 1.0},
            {"trade_date": date(2024, 1, 2), "ts_code": "000002.SZ", "factor_clean": 3.0},
            {"trade_date": date(2024, 1, 2), "ts_code": "000003.SZ", "factor_clean": 2.0},
        ]
    )
    prices = pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": code,
                "open": 10.0 + idx,
                "close": 10.1 + idx,
                "pre_close": 10.0 + idx,
                "pct_chg": 1.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
            for d in [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
            for idx, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"])
        ]
    )
    cfg = BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0)
    cost = CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0)

    generic = run_strategy_backtest(
        TopNLongOnlyStrategy(n=2),
        factors,
        prices,
        config=cfg,
        cost_model=cost,
    )
    precomputed = run_strategy_backtest(
        PrecomputedWeightsStrategy(precompute_top_n_weights(factors, top_n=2)),
        factors,
        prices,
        config=cfg,
        cost_model=cost,
    )
    fast = run_strategy_backtest(
        PrecomputedWeightsStrategy(precompute_top_n_weights(factors, top_n=2)),
        factors,
        prices,
        config=cfg,
        cost_model=cost,
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    assert precomputed.returns.equals(generic.returns)
    assert precomputed.nav.equals(generic.nav)
    assert precomputed.trades.equals(generic.trades)
    assert precomputed.summary_stats == generic.summary_stats
    assert fast.returns.equals(generic.returns)
    assert fast.nav.equals(generic.nav)
    assert fast.summary_stats == generic.summary_stats
    assert fast.positions.is_empty()
    assert fast.trades.is_empty()


def test_next_open_execution_starts_when_prior_signal_is_available():
    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        _prices(),
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    returns = result.returns.sort("trade_date")
    nav = result.nav.sort("trade_date")
    assert returns["trade_date"][0] == date(2024, 1, 2)
    assert returns["nav"][0] == pytest.approx(1.1)
    assert nav["trade_date"][0] == date(2024, 1, 1)
    assert nav["nav"][0] == pytest.approx(1.0)
    assert nav["trade_date"][1] == date(2024, 1, 2)
    assert nav["nav"][1] == pytest.approx(1.1)
    assert result.ret_definition == "open_to_close_with_overnight_carry"


def test_overnight_and_intraday_returns_are_compounded():
    prices = pl.DataFrame(
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
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 12.1,
                "close": 13.31,
                "pre_close": 11.0,
                "pct_chg": 21.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        factors,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000,
            max_participation_rate=1.0,
            fallback_adv=1_000_000.0,
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    day3 = result.returns.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
    assert day3["gross_return"] == pytest.approx((1.10 * 1.10) - 1.0)
    assert day3["net_return"] == pytest.approx((1.10 * 1.10) - 1.0)


def test_open_basis_trade_cost_is_scaled_to_prior_close_return_basis():
    prices = pl.DataFrame(
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
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 20.0,
                "close": 20.0,
                "pre_close": 10.0,
                "pct_chg": 100.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
        ]
    )

    result = run_strategy_backtest(
        ExitAfterCostNeutralEntryStrategy(),
        factors,
        prices,
        config=BacktestConfig(
            initial_capital=100.0,
            max_participation_rate=1.0,
            fallback_adv=1_000_000.0,
        ),
        cost_model=CostModel(commission=0.01, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    day3 = result.returns.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
    sell_trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
    assert day3["gross_return"] == pytest.approx(1.0)
    assert sell_trade["cost"] == pytest.approx(0.01)
    assert day3["cost"] == pytest.approx(0.02)
    assert day3["net_return"] == pytest.approx(0.98)


def test_current_positions_market_value_uses_open_nav_after_overnight_gap():
    prices = pl.DataFrame(
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
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 20.0,
                "close": 20.0,
                "pre_close": 10.0,
                "pct_chg": 100.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
        ]
    )
    strategy = CaptureCurrentPositionsStrategy()

    run_strategy_backtest(
        strategy,
        factors,
        prices,
        config=BacktestConfig(
            initial_capital=100.0,
            max_participation_rate=1.0,
            fallback_adv=1_000_000.0,
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    assert strategy.captured_positions is not None
    position = strategy.captured_positions.row(0, named=True)
    assert position["weight"] == pytest.approx(1.0)
    assert position["market_value"] == pytest.approx(200.0)


def test_positions_are_recorded_as_close_weights_after_intraday_drift():
    prices = pl.DataFrame(
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
                "trade_date": date(2024, 1, 1),
                "ts_code": "000002.SZ",
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
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000002.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "factor_clean": 1.0,
            },
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000002.SZ",
                "factor_clean": 1.0,
            },
        ]
    )

    result = run_strategy_backtest(
        EqualTwoStockStrategy(),
        factors,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000,
            max_participation_rate=1.0,
            fallback_adv=1_000_000.0,
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    positions = result.positions.filter(pl.col("trade_date") == date(2024, 1, 2))
    weights = dict(zip(positions["ts_code"].to_list(), positions["weight"].to_list(), strict=True))
    assert weights["000001.SZ"] == pytest.approx(0.55 / 1.05)
    assert weights["000002.SZ"] == pytest.approx(0.50 / 1.05)


def test_strategy_output_validation_errors_are_clear():
    with pytest.raises(ValueError, match="target_weight"):
        run_strategy_backtest(BadMissingColumnStrategy(), _factor(), _prices())
    with pytest.raises(ValueError, match="duplicate"):
        run_strategy_backtest(BadDuplicateStrategy(), _factor(), _prices())
    with pytest.raises(ValueError, match="finite"):
        run_strategy_backtest(BadNaNStrategy(), _factor(), _prices())


def test_trim_backtest_to_first_trade_recomputes_cached_summary():
    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        _prices(),
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )
    leading = pl.DataFrame(
        [
            {
                "trade_date": date(2023, 12, 29),
                "gross_return": 0.0,
                "cost": 0.0,
                "borrow_cost": 0.0,
                "net_return": 0.0,
                "nav": 1.0,
                "cash_weight": 1.0,
                "turnover": 0.0,
            }
        ]
    )
    cached = StrategyBacktestResult(
        factor_name=result.factor_name,
        strategy_name=result.strategy_name,
        n_groups=result.n_groups,
        returns=pl.concat([leading, result.returns]),
        nav=pl.concat(
            [
                leading.select(
                    [
                        "trade_date",
                        "gross_return",
                        "cost",
                        "borrow_cost",
                        "net_return",
                        "nav",
                        "cash_weight",
                    ]
                ),
                result.nav,
            ]
        ),
        positions=result.positions,
        trades=result.trades,
        summary_stats={"long_short": {"sharpe": -999.0}},
        config=result.config,
        frequency=result.frequency,
        ret_definition=result.ret_definition,
    )

    trimmed = trim_backtest_to_first_trade(cached)

    assert trimmed.returns["trade_date"][0] == date(2024, 1, 2)
    assert trimmed.summary_stats["long_short"]["sharpe"] != -999.0


def test_suspended_stock_blocks_trade():
    result = run_strategy_backtest(BuyOneStrategy(), _factor(), _prices(day2_open=None))
    trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.0)
    assert trade["block_reason"] == "missing_price"


def test_limit_up_blocks_buy_but_allows_sell():
    result = run_strategy_backtest(
        BuyOneStrategy(), _factor(), _prices(day2_open=11.0, day2_pct=9.9)
    )
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
        _prices(day3_open=9.9, day3_pct=-9.9),
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
        _prices(day1_amount=100.0),
        config=BacktestConfig(initial_capital=100.0, max_participation_rate=0.1),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )
    trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.1)
    assert trade["block_reason"] == "capacity"


def test_capacity_constraint_uses_open_nav_value_after_overnight_gap():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 20.0,
                "close": 20.0,
                "pre_close": 10.0,
                "pct_chg": 100.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 2), "000001.SZ", 1.0),
        ]
    )

    result = run_strategy_backtest(
        ExitOnSecondSignalStrategy(),
        factors,
        prices,
        config=BacktestConfig(
            initial_capital=100.0,
            max_participation_rate=1.0,
            fallback_adv=None,
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.filter(pl.col("trade_date") == date(2024, 1, 3)).row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(-0.5)
    assert trade["block_reason"] == "capacity"


def test_next_open_buy_is_not_blocked_by_same_day_close_limit():
    prices = pl.DataFrame(
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
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        prices,
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(1.0)
    assert trade["block_reason"] == ""


def test_next_open_buy_is_blocked_when_open_is_at_limit_up():
    prices = pl.DataFrame(
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
                "open": 11.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        prices,
        config=BacktestConfig(initial_capital=1_000_000, max_participation_rate=1.0),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.0)
    assert trade["block_reason"] == "limit_up"


def test_capacity_uses_trailing_adv_not_execution_day_amount():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor(),
        prices,
        config=BacktestConfig(initial_capital=100.0, max_participation_rate=0.1),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.1)
    assert trade["block_reason"] == "capacity"


def test_capacity_trailing_adv_ignores_nan_amounts():
    prices = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": np.nan,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 100_000_000.0,
            },
        ]
    )

    result = run_strategy_backtest(
        BuyOneStrategy(),
        _factor([(date(2024, 1, 2), "000001.SZ", 1.0)]),
        prices,
        config=BacktestConfig(initial_capital=100.0, max_participation_rate=0.1),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
    )

    trade = result.trades.sort("trade_date").row(0, named=True)
    assert trade["filled_delta_weight"] == pytest.approx(0.1)
    assert trade["block_reason"] == "capacity"


def test_summary_total_cost_uses_period_basis_return_drag():
    returns = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "gross_return": 0.0,
                "cost": 0.02,
                "borrow_cost": 0.0,
                "net_return": -0.02,
                "nav": 0.98,
                "cash_weight": 1.0,
                "turnover": 0.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "gross_return": 0.0,
                "cost": 0.03,
                "borrow_cost": 0.0,
                "net_return": -0.03,
                "nav": 0.9506,
                "cash_weight": 1.0,
                "turnover": 0.0,
            },
        ]
    )
    trades = pl.DataFrame(
        [
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000001.SZ",
                "prev_weight": 0.0,
                "target_weight": 1.0,
                "filled_delta_weight": 1.0,
                "turnover": 1.0,
                "cost": 999.0,
                "block_reason": "",
            }
        ]
    )

    stats = _summary_stats(returns, trades)

    assert stats["portfolio"]["total_cost"] == pytest.approx(0.05)


def test_summary_max_drawdown_includes_initial_nav():
    returns = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "gross_return": [-0.10],
            "cost": [0.0],
            "borrow_cost": [0.0],
            "net_return": [-0.10],
            "nav": [0.90],
            "cash_weight": [0.0],
            "turnover": [0.0],
        }
    )
    trades = pl.DataFrame(
        schema={
            "trade_date": pl.Date,
            "ts_code": pl.Utf8,
            "prev_weight": pl.Float64,
            "target_weight": pl.Float64,
            "filled_delta_weight": pl.Float64,
            "turnover": pl.Float64,
            "cost": pl.Float64,
            "block_reason": pl.Utf8,
        }
    )

    stats = _summary_stats(returns, trades)

    assert stats["portfolio"]["max_dd"] == pytest.approx(-0.10)


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
        config=BacktestConfig(max_participation_rate=100.0),
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


def test_quantile_long_short_thin_cross_section_returns_empty():
    """N < n_groups 时不得建单腿裸头寸，应 flat（#7）。"""
    factor = _factor(
        [
            (date(2024, 1, 1), "A", 1.0),
            (date(2024, 1, 1), "B", 2.0),
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

    weights = QuantileLongShortStrategy(n_groups=10).generate_weights(ctx)
    assert weights.height == 0
    assert weights.columns == ["ts_code", "target_weight"]


def test_quantile_long_short_one_empty_leg_returns_empty():
    """分组后 long 或 short 任一为空（近常数/退化分桶）→ flat，禁止裸多/裸空（#7）。"""
    # N=3、n_groups=10：rank 分组最多落到 0/3/6，填不满 top 组 → 旧实现只剩 short 腿
    factor = _factor(
        [
            (date(2024, 1, 1), "A", 1.0),
            (date(2024, 1, 1), "B", 2.0),
            (date(2024, 1, 1), "C", 3.0),
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

    weights = QuantileLongShortStrategy(n_groups=10).generate_weights(ctx)
    assert weights.height == 0, "单腿退化截面必须 flat，不得裸空/裸多"


def test_quantile_long_short_sufficient_cross_section_zero_regression():
    """N ≥ n_groups 且两腿齐全时保持等权 long/short（#7 零回归）。"""
    # 10 只、n_groups=5 → 每组 2 只，top/bottom 各 2
    factor = _factor(
        [(date(2024, 1, 1), f"S{i}", float(i)) for i in range(10)]
    )
    ctx = BacktestContext(
        signal_date=date(2024, 1, 1),
        execution_date=date(2024, 1, 2),
        factor_slice=factor,
        price_slice=pl.DataFrame(),
        current_positions=pl.DataFrame(),
        factor_col="factor_clean",
    )

    weights = QuantileLongShortStrategy(n_groups=5).generate_weights(ctx)
    wmap = dict(zip(weights["ts_code"], weights["target_weight"], strict=True))
    # bottom: S0,S1 short -0.5 each; top: S8,S9 long +0.5 each
    assert wmap == {
        "S0": -0.5,
        "S1": -0.5,
        "S8": 0.5,
        "S9": 0.5,
    }
    assert sum(w for w in wmap.values() if w > 0) == pytest.approx(1.0)
    assert sum(w for w in wmap.values() if w < 0) == pytest.approx(-1.0)


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


def test_fast_path_validates_target_weights_like_slow_path():
    """快速路径必须像慢路径一样校验 target_weight，而不是静默放行(Fix 3)。

    慢路径每次 generate_weights() 后都过 _validate_target_weights；
    PrecomputedWeightsStrategy 走快路径时此前完全跳过这层校验，非法权重
    会静默传播成垃圾 NAV。两种非法输入都应抛出和慢路径一致的清晰错误。
    """
    fast_kwargs = {
        "collect_positions": False,
        "collect_trades": False,
        "include_context_positions": False,
    }

    nan_weights = {
        date(2024, 1, 1): pl.DataFrame(
            {"ts_code": ["000001.SZ"], "target_weight": [float("nan")]}
        ),
    }
    with pytest.raises(ValueError, match="finite"):
        run_strategy_backtest(
            PrecomputedWeightsStrategy(nan_weights),
            _factor(),
            _prices(),
            **fast_kwargs,
        )

    oversized_weights = {
        date(2024, 1, 1): pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [5.0]}),
    }
    with pytest.raises(ValueError, match="max_abs_weight"):
        run_strategy_backtest(
            PrecomputedWeightsStrategy(oversized_weights),
            _factor(),
            _prices(),
            **fast_kwargs,
        )


def test_fast_path_charges_borrow_cost_on_short_position():
    """快速路径满仓做空时必须按 borrow_annual 扣息，闭式解验证（Fix 1）。

    2 天、单只股票、target_weight=-1.0（满仓做空），价格全程持平
    （gross_return=0、trade_cost=0），唯一的收益拖累应是融券利息：
    net_return = -short_exposure * borrow_rate_per_period。
    """
    prices = pl.DataFrame(
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
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    weights_by_date = {
        date(2024, 1, 1): pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [-1.0]}),
    }
    factors = _factor([(date(2024, 1, 1), "000001.SZ", 1.0)])
    cfg = BacktestConfig(
        initial_capital=1_000_000, max_participation_rate=1.0, fallback_adv=1_000_000.0
    )
    cost_model = CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0.10)

    fast = run_strategy_backtest(
        PrecomputedWeightsStrategy(weights_by_date),
        factors,
        prices,
        config=cfg,
        cost_model=cost_model,
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    expected_borrow_cost = 1.0 * cost_model.borrow_rate_per_period("daily")
    day2 = fast.nav.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
    assert day2["borrow_cost"] == pytest.approx(expected_borrow_cost)
    assert day2["net_return"] == pytest.approx(-expected_borrow_cost)
    assert day2["nav"] == pytest.approx(1.0 - expected_borrow_cost)


def test_fast_path_borrow_cost_matches_slow_path():
    """快速路径的融券扣息须和慢路径数值一致（Fix 1 慢/快对照）。"""
    prices = pl.DataFrame(
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
                "close": 10.5,
                "pre_close": 10.0,
                "pct_chg": 5.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 3),
                "ts_code": "000001.SZ",
                "open": 10.5,
                "close": 10.4,
                "pre_close": 10.5,
                "pct_chg": -1.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    weights_by_date = {
        date(2024, 1, 1): pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [-0.5]}),
    }
    factors = _factor([(date(2024, 1, 1), "000001.SZ", 1.0)])
    cfg = BacktestConfig(
        initial_capital=1_000_000, max_participation_rate=1.0, fallback_adv=1_000_000.0
    )
    cost_model = CostModel(commission=0.0005, stamp_tax=0.001, slippage=0.0005, borrow_annual=0.085)

    slow = run_strategy_backtest(
        PrecomputedWeightsStrategy(weights_by_date),
        factors,
        prices,
        config=cfg,
        cost_model=cost_model,
    )
    fast = run_strategy_backtest(
        PrecomputedWeightsStrategy(weights_by_date),
        factors,
        prices,
        config=cfg,
        cost_model=cost_model,
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    assert fast.nav["borrow_cost"].to_list() == pytest.approx(slow.nav["borrow_cost"].to_list())
    assert fast.nav["nav"].to_list() == pytest.approx(slow.nav["nav"].to_list())
    # 卫生检查：确实有非零融券成本被扣除（不是两条路径都恰好为 0 而巧合相等）
    assert any(v > 0 for v in fast.nav["borrow_cost"].to_list())


def test_borrow_cost_is_daily_regardless_of_frequency():
    """融券是每日持有成本：回测循环恒按日迭代，融券成本必须按【日】费率计提，
    不随 frequency(weekly/monthly) 放大。

    历史 bug：monthly 下每个交易日按 21 天融券费计提，一个交易日就被扣 21 天利息，
    全年累计高估约 21x（weekly 约 5x），足以把盈利的市场中性因子在 Tear Sheet 上
    翻成亏损。慢路径(通用循环)与快路径(precomputed fast)都必须正确。
    """
    prices = pl.DataFrame(
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
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    weights_by_date = {
        date(2024, 1, 1): pl.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [-1.0]}),
    }
    factors = _factor([(date(2024, 1, 1), "000001.SZ", 1.0)])
    cost_model = CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0.10)
    # 满仓做空持有一个交易日，应恰好扣 annual/252，与 frequency 无关
    expected_daily_borrow = 1.0 * cost_model.borrow_rate_per_period("daily")

    for frequency in ("weekly", "monthly"):
        cfg = BacktestConfig(
            initial_capital=1_000_000,
            max_participation_rate=1.0,
            fallback_adv=1_000_000.0,
            frequency=frequency,
        )
        # collect_positions 默认 True -> 慢路径（通用循环）
        slow = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            factors,
            prices,
            config=cfg,
            cost_model=cost_model,
        )
        # collect 全关 -> 快路径（_run_precomputed_weights_backtest_fast）
        fast = run_strategy_backtest(
            PrecomputedWeightsStrategy(weights_by_date),
            factors,
            prices,
            config=cfg,
            cost_model=cost_model,
            collect_positions=False,
            collect_trades=False,
            include_context_positions=False,
        )
        slow_day2 = slow.nav.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
        fast_day2 = fast.nav.filter(pl.col("trade_date") == date(2024, 1, 2)).row(0, named=True)
        assert slow_day2["borrow_cost"] == pytest.approx(expected_daily_borrow), (
            f"慢路径 {frequency} 融券应按日计提"
        )
        assert fast_day2["borrow_cost"] == pytest.approx(expected_daily_borrow), (
            f"快路径 {frequency} 融券应按日计提"
        )


def test_fast_path_handles_missing_open_price_without_crashing():
    """快速路径 open 为 None 时不应崩溃，该股票当天判定不可交易，其余股票正常（Fix 4）。"""
    weights_by_date = {
        date(2024, 1, 1): pl.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "target_weight": [0.5, 0.5]}
        ),
    }
    prices = pl.DataFrame(
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
                "trade_date": date(2024, 1, 1),
                "ts_code": "000002.SZ",
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
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000002.SZ",
                "open": None,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            },
        ]
    )
    factors = _factor(
        [
            (date(2024, 1, 1), "000001.SZ", 1.0),
            (date(2024, 1, 1), "000002.SZ", 1.0),
        ]
    )

    fast = run_strategy_backtest(
        PrecomputedWeightsStrategy(weights_by_date),
        factors,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000, max_participation_rate=1.0, fallback_adv=1_000_000.0
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        collect_positions=False,
        collect_trades=False,
        include_context_positions=False,
    )

    day2_nav = fast.nav.filter(pl.col("trade_date") == date(2024, 1, 2))["nav"][0]
    # 000002.SZ 因缺 open 不可交易（贡献 0），000001.SZ 满仓一半 + 10% 涨幅
    assert day2_nav == pytest.approx(1.05)


def test_fast_path_handles_missing_pre_close_without_crashing():
    """快速路径内部函数 pre_close 为 None(open 有效)时不应崩溃，该股票当天判定不可交易（Fix 4）。

    ``_prepare_price_df`` 会把逐行 ``pre_close=None`` 兜底填成同行 ``open``
    （"用今日开盘价近似昨收"），因此公开入口 ``run_strategy_backtest`` 无法构造
    出"pre_close=None 且 open 有效"这种组合传到快路径——这里直接调用内部函数
    ``_run_precomputed_weights_backtest_fast``，绕开 ``_prepare_price_df`` 的
    兜底，独立验证 pre_close 的 None 保护分支（与测试文件既有的
    ``_compute_adv_20d`` 等私有函数直测惯例一致）。
    """
    weights_by_date = {
        date(2024, 1, 1): pl.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "target_weight": [0.5, 0.5]}
        ),
    }
    trade_dates = [date(2024, 1, 1), date(2024, 1, 2)]
    price = pl.DataFrame(
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
                "overnight_ret": 0.0,
                "intraday_ret": 0.0,
            },
            {
                "trade_date": date(2024, 1, 1),
                "ts_code": "000002.SZ",
                "open": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
                "overnight_ret": 0.0,
                "intraday_ret": 0.0,
            },
            {
                "trade_date": date(2024, 1, 2),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
                "overnight_ret": 0.0,
                "intraday_ret": 0.1,
            },
            {
                # pre_close=None（绕过 _prepare_price_df 的 fill_null(open) 兜底），open 仍有效
                "trade_date": date(2024, 1, 2),
                "ts_code": "000002.SZ",
                "open": 10.0,
                "close": 11.0,
                "pre_close": None,
                "pct_chg": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
                "overnight_ret": 0.0,
                "intraday_ret": 0.1,
            },
        ]
    )

    result = _run_precomputed_weights_backtest_fast(
        strategy=PrecomputedWeightsStrategy(weights_by_date),
        price=price,
        trade_dates=trade_dates,
        config=BacktestConfig(
            initial_capital=1_000_000, max_participation_rate=1.0, fallback_adv=1_000_000.0
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        factor_name="test",
    )

    day2_nav = result.nav.filter(pl.col("trade_date") == date(2024, 1, 2))["nav"][0]
    # 000002.SZ 因缺 pre_close 不可交易（贡献 0），000001.SZ 满仓一半 + 10% 涨幅
    assert day2_nav == pytest.approx(1.05)


def test_optimizer_strategy_end_to_end():
    """OptimizerStrategy 端到端回测跑通，结果结构正确。"""
    from factorzen.daily.optimization.mean_variance import MeanVarianceOptimizer

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
