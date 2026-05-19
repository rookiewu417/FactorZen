"""Tests for common.config_loader module."""
from __future__ import annotations

import pytest


def test_load_valid_config(tmp_path):
    from common.config_loader import load_run_config

    yaml_content = "factor: momentum_20d\nstart: '20230101'\nend: '20241231'\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content)
    config = load_run_config(p)
    assert config.factor == "momentum_20d"
    assert config.seed is None  # optional default


def test_load_config_with_seed(tmp_path):
    from common.config_loader import load_run_config

    yaml_content = "factor: reversal\nstart: '20230101'\nend: '20241231'\nseed: 99\n"
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content)
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
    p.write_text(yaml_content)
    with pytest.raises(pydantic.ValidationError):
        load_run_config(p)


def test_default_preprocessing():
    from common.config_loader import RunConfig

    cfg = RunConfig(factor="x", start="20230101", end="20241231")
    assert cfg.preprocessing.outlier == "mad"
    assert cfg.preprocessing.normalizer == "zscore"
