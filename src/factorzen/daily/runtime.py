"""Build daily runtime objects from validated declarative research configuration."""

from __future__ import annotations

from factorzen.config.research import RunConfig, StrategySpec
from factorzen.daily.evaluation.backtest import BacktestConfig as RuntimeBacktestConfig
from factorzen.daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel
from factorzen.daily.evaluation.strategy_registry import build_strategy
from factorzen.daily.preprocessing.pipeline import PreprocessingPipeline


def build_preprocessing_pipeline(config: RunConfig) -> PreprocessingPipeline:
    """Build the configured preprocessing implementation."""
    return PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        outlier_method=config.preprocessing.outlier,
        normalizer_method=config.preprocessing.normalizer,
        neutralize=config.preprocessing.neutralize,
    )


def build_runtime_backtest_config(
    config: RunConfig,
    factor_col: str = "factor_clean",
    frequency: str = "daily",
    strategy_spec: StrategySpec | None = None,
) -> RuntimeBacktestConfig:
    """Build the daily backtest implementation config."""
    return RuntimeBacktestConfig(
        factor_col=factor_col,
        frequency=frequency,
        max_abs_weight=(
            strategy_spec.max_abs_weight
            if strategy_spec is not None and strategy_spec.max_abs_weight is not None
            else config.backtest.max_abs_weight
        ),
        rebalance_threshold=(
            strategy_spec.rebalance_threshold
            if strategy_spec is not None and strategy_spec.rebalance_threshold is not None
            else config.backtest.rebalance_threshold
        ),
        strategy_type=strategy_spec.type if strategy_spec is not None else None,
        strategy_params=dict(strategy_spec.params) if strategy_spec is not None else {},
        cost_model=(
            strategy_spec.cost_model
            if strategy_spec is not None and strategy_spec.cost_model is not None
            else config.backtest.cost_model
        ),
        alpha=(
            strategy_spec.alpha
            if strategy_spec is not None and strategy_spec.alpha is not None
            else config.backtest.alpha
        ),
        fallback_adv=(
            strategy_spec.fallback_adv
            if strategy_spec is not None and strategy_spec.fallback_adv is not None
            else config.backtest.fallback_adv
        ),
    )


def build_cost_model(config: RunConfig, strategy_spec: StrategySpec | None = None):
    """Build the configured transaction-cost implementation."""
    cost_model = (
        strategy_spec.cost_model
        if strategy_spec and strategy_spec.cost_model
        else config.backtest.cost_model
    )
    if cost_model == "square_root_impact":
        return SquareRootImpactCostModel(
            alpha=(
                strategy_spec.alpha
                if strategy_spec and strategy_spec.alpha is not None
                else config.backtest.alpha
            ),
            fallback_adv=(
                strategy_spec.fallback_adv
                if strategy_spec and strategy_spec.fallback_adv is not None
                else config.backtest.fallback_adv
            ),
        )
    return LinearCostModel()


def build_backtest_strategy(config: RunConfig):
    """Build the primary configured strategy."""
    strategies = build_backtest_strategies(config)
    return strategies[config.backtest.primary or next(iter(strategies))]


def build_backtest_strategies(config: RunConfig):
    """Build all named runtime strategies."""
    strategies = {}
    for spec in config.backtest.strategy_specs:
        strategy = build_strategy(spec.type, spec.params)
        strategy.name = spec.name
        strategies[spec.name] = strategy
    return strategies
