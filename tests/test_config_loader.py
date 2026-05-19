"""Tests for common.config_loader module."""
from __future__ import annotations

import pytest


def test_load_valid_config(tmp_path):
    from common.config_loader import load_run_config

    yaml_content = "factor: momentum_20d\nstart: '20230101'\nend: '20241231'\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    config = load_run_config(p)
    assert config.factor == "momentum_20d"
    assert config.seed is None  # optional default


def test_load_config_with_seed(tmp_path):
    from common.config_loader import load_run_config

    yaml_content = "factor: reversal\nstart: '20230101'\nend: '20241231'\nseed: 99\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    config = load_run_config(p)
    assert config.seed == 99


def test_invalid_outlier_method(tmp_path):
    import pydantic

    from common.config_loader import load_run_config

    yaml_content = (
        "factor: x\nstart: '20230101'\nend: '20241231'\n"
        "preprocessing:\n  outlier: invalid_method\n"
    )
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_run_config(p)


def test_default_preprocessing():
    from common.config_loader import RunConfig

    cfg = RunConfig(factor="x", start="20230101", end="20241231")
    assert cfg.preprocessing.outlier == "mad"
    assert cfg.preprocessing.normalizer == "zscore"


def test_build_preprocessing_pipeline_from_run_config():
    from common.config_loader import RunConfig, build_preprocessing_pipeline

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
    from common.config_loader import RunConfig, build_runtime_backtest_config

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
    from common.config_loader import RunConfig, build_cost_model
    from daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel

    linear_cfg = RunConfig(factor="x", start="20230101", end="20241231")
    assert isinstance(build_cost_model(linear_cfg), LinearCostModel)

    impact_cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20241231",
        backtest={"cost_model": "square_root_impact"},
    )
    assert isinstance(build_cost_model(impact_cfg), SquareRootImpactCostModel)
