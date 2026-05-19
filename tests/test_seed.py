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


def test_get_optuna_sampler_not_none():
    from common.seed import get_optuna_sampler

    sampler = get_optuna_sampler(42)
    assert sampler is not None
