"""Single-factor walk-forward summary helpers."""

from __future__ import annotations

from typing import Any

import polars as pl

from factorzen.core.config_loader import (
    RunConfig,
    build_runtime_backtest_config,
    build_top_n_candidate_params,
)
from factorzen.daily.evaluation.backtest import Strategy, TopNLongOnlyStrategy
from factorzen.daily.evaluation.walk_forward import (
    WalkForwardResult,
    WalkForwardSplitter,
    run_walk_forward_search,
)


def summarize_walk_forward_result(
    result: WalkForwardResult,
    *,
    requested_n_trials: int | None = None,
    param_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a WalkForwardResult to a JSON-serializable metadata summary."""
    if not result.folds:
        summary: dict[str, Any] = {"status": "insufficient_data", "n_folds": 0}
    else:
        summary = {
            "status": "ok",
            "n_folds": len(result.folds),
            "is_sharpe_mean": result.is_sharpe_mean,
            "oos_sharpe_mean": result.oos_sharpe_mean,
            "oos_sharpe_std": result.oos_sharpe_std,
            "oos_max_dd": result.oos_max_dd,
            "stability_ratio": result.stability_ratio,
        }
    if requested_n_trials is not None:
        summary["requested_n_trials"] = requested_n_trials
    if param_candidates is not None:
        summary["param_candidates"] = param_candidates
    return summary


def run_quantile_walk_forward_summary(
    factor_df: pl.DataFrame,
    price_df: pl.DataFrame,
    config: RunConfig,
    *,
    factor_name: str,
    frequency: str = "daily",
) -> tuple[dict[str, Any], WalkForwardResult | None]:
    """Run top-N walk-forward search and return summary plus result.

    Each fold searches deterministic top_n candidates on the IS window and
    evaluates the selected candidate on the OOS window.
    """
    splitter = WalkForwardSplitter(
        train_days=config.walk_forward.train_days,
        test_days=config.walk_forward.test_days,
        step_days=config.walk_forward.step_days,
        embargo_days=config.walk_forward.embargo_days,
    )

    def strategy_factory(params: dict[str, Any]) -> Strategy:
        return TopNLongOnlyStrategy(
            n=int(params.get("top_n", config.backtest.top_n)),
            factor_col="factor_clean",
        )

    param_candidates = build_top_n_candidate_params(config)
    result = run_walk_forward_search(
        strategy_factory=strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        param_candidates=param_candidates,
        config=build_runtime_backtest_config(config, factor_col="factor_clean", frequency=frequency),
        factor_name=factor_name,
        seed=config.seed,
    )
    summary = summarize_walk_forward_result(
        result,
        requested_n_trials=config.walk_forward.n_trials,
        param_candidates=param_candidates,
    )
    if summary["status"] == "insufficient_data":
        return summary, None
    return summary, result
