"""Tests for common.config_loader module."""

from __future__ import annotations

import pytest


def test_load_valid_config(tmp_path):
    from factorzen.core.config_loader import load_run_config

    yaml_content = "factor: momentum_20d\nstart: '20230101'\nend: '20241231'\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    config = load_run_config(p)
    assert config.factor == "momentum_20d"
    assert config.benchmark is None
    assert config.seed is None  # optional default


def test_load_config_with_seed(tmp_path):
    from factorzen.core.config_loader import load_run_config

    yaml_content = "factor: reversal\nstart: '20230101'\nend: '20241231'\nseed: 99\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    config = load_run_config(p)
    assert config.seed == 99


def test_invalid_outlier_method(tmp_path):
    import pydantic

    from factorzen.core.config_loader import load_run_config

    yaml_content = (
        "factor: x\nstart: '20230101'\nend: '20241231'\npreprocessing:\n  outlier: invalid_method\n"
    )
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_run_config(p)


def test_default_preprocessing():
    from factorzen.core.config_loader import RunConfig

    cfg = RunConfig(factor="x", start="20230101", end="20241231")
    assert cfg.preprocessing.outlier == "mad"
    assert cfg.preprocessing.normalizer == "zscore"


def test_walk_forward_is_disabled_by_default_and_can_be_enabled():
    from factorzen.core.config_loader import RunConfig

    default_cfg = RunConfig(factor="x", start="20230101", end="20241231")
    enabled_cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        walk_forward={"enabled": True},
    )

    assert default_cfg.walk_forward.enabled is False
    assert enabled_cfg.walk_forward.enabled is True


def test_build_preprocessing_pipeline_from_run_config():
    from factorzen.core.config_loader import RunConfig, build_preprocessing_pipeline

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        preprocessing={
            "outlier": "winsorize",
            "normalizer": "rank_uniform",
            "neutralize": True,
        },
    )

    pipeline = build_preprocessing_pipeline(cfg)

    assert pipeline.outlier_method == "winsorize"
    assert pipeline.normalizer_method == "rank_uniform"
    assert pipeline.neutralize is True


def test_build_runtime_backtest_config_from_run_config():
    from factorzen.core.config_loader import RunConfig, build_runtime_backtest_config

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={
            "quantiles": 7,
            "max_abs_weight": 0.2,
            "rebalance_threshold": 0.15,
        },
    )

    runtime = build_runtime_backtest_config(cfg, factor_col="factor_clean", frequency="weekly")

    assert runtime.factor_col == "factor_clean"
    assert runtime.frequency == "weekly"
    assert runtime.max_abs_weight == 0.2
    assert runtime.rebalance_threshold == 0.15


def test_build_cost_model_from_run_config():
    from factorzen.core.config_loader import RunConfig, build_cost_model
    from factorzen.daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel

    linear_cfg = RunConfig(factor="x", start="20230101", end="20241231")
    assert isinstance(build_cost_model(linear_cfg), LinearCostModel)

    impact_cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={"cost_model": "square_root_impact"},
    )
    assert isinstance(build_cost_model(impact_cfg), SquareRootImpactCostModel)


def test_default_benchmark_is_derived_from_universe():
    from factorzen.core.config_loader import default_benchmark_for_universe

    assert default_benchmark_for_universe("csi300") == "000300.SH"
    assert default_benchmark_for_universe("csi500") == "000905.SH"
    assert default_benchmark_for_universe("csi800") == "000906.SH"
    assert default_benchmark_for_universe("unknown") == "000300.SH"


def test_build_backtest_strategy_uses_top_n():
    from factorzen.core.config_loader import RunConfig, build_backtest_strategy
    from factorzen.daily.evaluation.backtest import TopNLongOnlyStrategy

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={"top_n": 17},
    )

    strategy = build_backtest_strategy(cfg)

    assert isinstance(strategy, TopNLongOnlyStrategy)
    assert strategy.n == 17


def test_backtest_config_supports_multiple_named_strategies():
    from factorzen.core.config_loader import RunConfig

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={
            "primary": "topn_50",
            "strategies": [
                {"name": "topn_50", "type": "topn_long_only", "params": {"top_n": 50}},
                {
                    "name": "quantile_ls_5",
                    "type": "quantile_long_short",
                    "params": {"quantiles": 5},
                },
            ],
        },
    )

    assert cfg.backtest.primary == "topn_50"
    assert [strategy.name for strategy in cfg.backtest.strategies] == [
        "topn_50",
        "quantile_ls_5",
    ]
    assert cfg.backtest.strategies[1].params == {"quantiles": 5}


def test_strategy_spec_can_override_runtime_backtest_settings():
    from factorzen.core.config_loader import RunConfig, build_runtime_backtest_config

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={
            "strategies": [
                {
                    "name": "topn_20_tight",
                    "type": "topn_long_only",
                    "params": {"top_n": 20},
                    "max_abs_weight": 0.03,
                    "rebalance_threshold": 0.2,
                    "cost_model": "square_root_impact",
                }
            ],
        },
    )

    spec = cfg.backtest.strategy_specs[0]
    assert spec.max_abs_weight == 0.03
    assert spec.rebalance_threshold == 0.2
    assert spec.cost_model == "square_root_impact"

    runtime = build_runtime_backtest_config(cfg, strategy_spec=spec)
    assert runtime.strategy_type == "topn_long_only"
    assert runtime.strategy_params == {"top_n": 20}


def test_legacy_backtest_config_exposes_default_strategy_spec():
    from factorzen.core.config_loader import RunConfig

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={"top_n": 17, "quantiles": 7},
    )

    assert cfg.backtest.strategy_specs[0].name == "topn_17"
    assert cfg.backtest.strategy_specs[0].type == "topn_long_only"
    assert cfg.backtest.strategy_specs[0].params == {"top_n": 17}
    assert cfg.backtest.primary == "topn_17"


def test_build_backtest_strategies_returns_named_runtime_strategies():
    from factorzen.core.config_loader import RunConfig, build_backtest_strategies
    from factorzen.daily.evaluation.backtest import QuantileLongShortStrategy, TopNLongOnlyStrategy

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={
            "strategies": [
                {"name": "topn_12", "type": "topn_long_only", "params": {"top_n": 12}},
                {"name": "ls_4", "type": "quantile_long_short", "params": {"quantiles": 4}},
            ],
        },
    )

    strategies = build_backtest_strategies(cfg)

    assert list(strategies) == ["topn_12", "ls_4"]
    assert isinstance(strategies["topn_12"], TopNLongOnlyStrategy)
    assert strategies["topn_12"].n == 12
    assert isinstance(strategies["ls_4"], QuantileLongShortStrategy)
    assert strategies["ls_4"].n_groups == 4


def test_default_all_strategy_specs_include_builtin_suite():
    from factorzen.core.config_loader import default_all_strategy_specs

    specs = default_all_strategy_specs()

    assert [spec.name for spec in specs] == [
        "topn_50",
        "quantile_ls_5",
        "factor_weighted_ls",
        "optimizer_mv_long_only",
    ]
    assert [spec.type for spec in specs] == [
        "topn_long_only",
        "quantile_long_short",
        "factor_weighted",
        "optimizer_strategy",
    ]
    assert specs[3].params["optimizer"] == "mean_variance"
    assert specs[3].params["long_only"] is True


def test_build_top_n_candidate_params_uses_n_trials():
    from factorzen.core.config_loader import RunConfig, build_top_n_candidate_params

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={"top_n": 10},
        walk_forward={"n_trials": 4},
    )

    assert build_top_n_candidate_params(cfg) == [
        {"top_n": 10},
    ]


def test_build_top_n_candidate_params_skips_weights_above_cap():
    from factorzen.core.config_loader import RunConfig, build_top_n_candidate_params

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={"top_n": 50, "max_abs_weight": 0.1},
        walk_forward={"n_trials": 5},
    )

    assert build_top_n_candidate_params(cfg) == [
        {"top_n": 10},
        {"top_n": 20},
        {"top_n": 30},
        {"top_n": 40},
        {"top_n": 50},
    ]


def test_default_daily_research_config_top_n_override_updates_primary_strategy():
    from factorzen.core.config_loader import (
        build_default_daily_research_config,
        build_run_config_from_dict,
    )

    base = build_default_daily_research_config(
        factor="x",
        start="20230101",
        end="20241231",
    ).model_dump()

    cfg = build_run_config_from_dict(base, overrides=["backtest.top_n=30"])

    assert cfg.backtest.primary == "topn_30"
    primary = cfg.backtest.strategy_specs[0]
    assert primary.name == "topn_30"
    assert primary.params["top_n"] == 30
