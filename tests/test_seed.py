"""Tests for common.seed module."""
from __future__ import annotations

import numpy as np


def test_set_global_seed_returns_dict():
    from common.seed import set_global_seed

    result = set_global_seed(42)
    assert result["seed"] == 42


def test_seed_reproducibility():
    """固定种子两次采样结果相同。"""
    from common.seed import set_global_seed

    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_get_optuna_sampler_reproducible():
    """同种子采样器产生相同建议。"""
    import optuna

    from common.seed import get_optuna_sampler
    sampler1 = get_optuna_sampler(42)
    sampler2 = get_optuna_sampler(42)
    study = optuna.create_study(sampler=sampler1)
    trial1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    study2 = optuna.create_study(sampler=sampler2)
    trial2 = study2.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    assert trial1.params == trial2.params
