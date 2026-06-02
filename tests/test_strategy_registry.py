"""Tests for pluggable backtest strategy construction."""

from __future__ import annotations

import sys
from textwrap import dedent


def test_builtin_strategy_registry_builds_supported_strategies():
    from factorzen.daily.evaluation.backtest import (
        FactorWeightedStrategy,
        OptimizerStrategy,
        QuantileLongShortStrategy,
        TopNLongOnlyStrategy,
    )
    from factorzen.daily.evaluation.strategy_registry import build_strategy

    topn = build_strategy("topn_long_only", {"top_n": 12})
    quantile = build_strategy("quantile_long_short", {"quantiles": 4})
    weighted = build_strategy(
        "factor_weighted",
        {"long_only": True, "gross_exposure": 1.0, "long_exposure": 0.8},
    )

    assert isinstance(topn, TopNLongOnlyStrategy)
    assert topn.n == 12
    assert isinstance(quantile, QuantileLongShortStrategy)
    assert quantile.n_groups == 4
    assert isinstance(weighted, FactorWeightedStrategy)
    assert weighted.long_only is True
    assert weighted.long_exposure == 0.8

    optimizer = build_strategy(
        "optimizer_strategy",
        {
            "optimizer": "mean_variance",
            "risk_aversion": 2.0,
            "lookback_days": 40,
            "cov_estimator": "ledoit_wolf",
            "long_only": True,
            "top_n": 80,
            "max_weight": 0.08,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
        },
    )

    assert isinstance(optimizer, OptimizerStrategy)
    assert optimizer.lookback_days == 40
    assert optimizer.cov_estimator == "ledoit_wolf"
    assert optimizer.long_only is True
    assert optimizer.top_n == 80
    assert optimizer.constraints.max_weight == 0.08


def test_strategy_registry_imports_custom_strategy_from_dotted_path(tmp_path, monkeypatch):
    module_path = tmp_path / "custom_strategy.py"
    module_path.write_text(
        dedent(
            """
            import polars as pl

            from factorzen.daily.evaluation.backtest import Strategy


            class CustomStrategy(Strategy):
                name = "custom"

                def __init__(self, multiplier: int) -> None:
                    self.multiplier = multiplier

                @classmethod
                def from_config(cls, config):
                    return cls(multiplier=config["multiplier"])

                def generate_weights(self, context):
                    return pl.DataFrame(
                        {"ts_code": [], "target_weight": []},
                        schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
                    )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("custom_strategy", None)

    from factorzen.daily.evaluation.strategy_registry import build_strategy

    strategy = build_strategy("custom_strategy.CustomStrategy", {"multiplier": 3})

    assert strategy.name == "custom"
    assert strategy.multiplier == 3
