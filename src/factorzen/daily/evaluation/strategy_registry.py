"""Backtest strategy registry and dynamic construction."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from factorzen.daily.evaluation.backtest import (
    FactorWeightedStrategy,
    OptimizerStrategy,
    QuantileLongShortStrategy,
    Strategy,
    TopNLongOnlyStrategy,
)


def _build_topn(params: dict[str, Any]) -> Strategy:
    return TopNLongOnlyStrategy(
        n=int(params.get("top_n", params.get("n", 50))),
        factor_col=str(params.get("factor_col", "factor_clean")),
    )


def _build_quantile_long_short(params: dict[str, Any]) -> Strategy:
    return QuantileLongShortStrategy(
        n_groups=int(params.get("quantiles", params.get("n_groups", 10))),
        factor_col=str(params.get("factor_col", "factor_clean")),
    )


def _build_factor_weighted(params: dict[str, Any]) -> Strategy:
    return FactorWeightedStrategy(
        long_only=bool(params.get("long_only", False)),
        gross_exposure=float(params.get("gross_exposure", 2.0)),
        long_exposure=float(params.get("long_exposure", 1.0)),
        factor_col=str(params.get("factor_col", "factor_clean")),
    )


def _build_optimizer_strategy(params: dict[str, Any]) -> Strategy:
    optimizer_name = str(params.get("optimizer", "mean_variance"))
    if optimizer_name != "mean_variance":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    from factorzen.daily.optimization.base import OptimizerConstraints
    from factorzen.daily.optimization.mean_variance import MeanVarianceOptimizer

    constraints = OptimizerConstraints(
        max_weight=float(params.get("max_weight", params.get("max_abs_weight", 0.1))),
        min_weight=float(params.get("min_weight", 0.0)),
        gross_exposure=float(params.get("gross_exposure", 1.0)),
        net_exposure=float(params.get("net_exposure", 1.0)),
        turnover_limit=(
            float(params["turnover_limit"]) if params.get("turnover_limit") is not None else None
        ),
    )
    return OptimizerStrategy(
        optimizer=MeanVarianceOptimizer(risk_aversion=float(params.get("risk_aversion", 1.0))),
        lookback_days=int(params.get("lookback_days", 60)),
        factor_col=str(params.get("factor_col", "factor_clean")),
        cov_estimator=str(params.get("cov_estimator", "ledoit_wolf")),
        constraints=constraints,
        long_only=bool(params.get("long_only", True)),
        top_n=int(params["top_n"]) if params.get("top_n") is not None else None,
    )


_BUILTIN_BUILDERS = {
    "topn_long_only": _build_topn,
    "quantile_long_short": _build_quantile_long_short,
    "factor_weighted": _build_factor_weighted,
    "optimizer_strategy": _build_optimizer_strategy,
}


def _load_strategy_class(type_name: str) -> type[Strategy]:
    module_name, _, class_name = type_name.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Unknown strategy type: {type_name}")
    module = import_module(module_name)
    strategy_cls = getattr(module, class_name)
    if not issubclass(strategy_cls, Strategy):
        raise TypeError(f"{type_name} is not a Strategy subclass")
    return strategy_cls


def build_strategy(type_name: str, params: dict[str, Any] | None = None) -> Strategy:
    """Build a strategy from a built-in type name or dotted class path."""
    params = dict(params or {})
    if type_name in _BUILTIN_BUILDERS:
        return _BUILTIN_BUILDERS[type_name](params)

    strategy_cls = _load_strategy_class(type_name)
    if hasattr(strategy_cls, "from_config"):
        return strategy_cls.from_config(params)
    return strategy_cls(**params)
