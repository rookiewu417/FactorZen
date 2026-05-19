"""Single-factor walk-forward summary helpers."""

from __future__ import annotations

from typing import Any

import polars as pl

from common.config_loader import RunConfig, build_runtime_backtest_config
from daily.evaluation.backtest import QuantileLongShortStrategy, Strategy
from daily.evaluation.walk_forward import WalkForwardResult, WalkForwardSplitter, run_walk_forward


def summarize_walk_forward_result(result: WalkForwardResult) -> dict[str, Any]:
    """Convert a WalkForwardResult to a JSON-serializable metadata summary."""
    if not result.folds:
        return {"status": "insufficient_data", "n_folds": 0}
    return {
        "status": "ok",
        "n_folds": len(result.folds),
        "is_sharpe_mean": result.is_sharpe_mean,
        "oos_sharpe_mean": result.oos_sharpe_mean,
        "oos_sharpe_std": result.oos_sharpe_std,
        "oos_max_dd": result.oos_max_dd,
        "stability_ratio": result.stability_ratio,
    }


def run_quantile_walk_forward_summary(
    factor_df: pl.DataFrame,
    price_df: pl.DataFrame,
    config: RunConfig,
    *,
    factor_name: str,
    frequency: str = "daily",
) -> tuple[dict[str, Any], WalkForwardResult | None]:
    """Run quantile long-short walk-forward and return metadata summary plus result."""
    splitter = WalkForwardSplitter(
        train_days=config.walk_forward.train_days,
        test_days=config.walk_forward.test_days,
        step_days=config.walk_forward.step_days,
        embargo_days=config.walk_forward.embargo_days,
    )

    def strategy_factory(_params: dict[str, Any]) -> Strategy:
        return QuantileLongShortStrategy(
            n_groups=config.backtest.quantiles,
            factor_col="factor_clean",
        )

    result = run_walk_forward(
        strategy_factory=strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        config=build_runtime_backtest_config(config, factor_col="factor_clean", frequency=frequency),
        factor_name=factor_name,
        params={},
        seed=config.seed,
    )
    summary = summarize_walk_forward_result(result)
    if summary["status"] == "insufficient_data":
        return summary, None
    return summary, result
