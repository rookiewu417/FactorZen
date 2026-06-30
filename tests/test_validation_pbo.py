import numpy as np


def test_pbo_noise_near_half():
    """纯噪声候选池：IS 最优在 OOS 无优势 → PBO ≈ 0.5。"""
    from factorzen.validation.pbo import compute_pbo
    rng = np.random.default_rng(0)
    perf = rng.normal(0, 1, (20, 200))
    pbo = compute_pbo(perf, n_splits=10)
    assert 0.3 < pbo < 0.7


def test_pbo_one_dominant_low():
    """一个候选全程显著最优 → IS 最优 = OOS 最优 → PBO 低。"""
    from factorzen.validation.pbo import compute_pbo
    rng = np.random.default_rng(0)
    perf = rng.normal(0, 1, (20, 200))
    perf[0] += 3.0  # 候选0 全程领先
    pbo = compute_pbo(perf, n_splits=10)
    assert pbo < 0.2


def test_pbo_too_small_returns_nan():
    from factorzen.validation.pbo import compute_pbo
    assert np.isnan(compute_pbo(np.zeros((1, 100)), n_splits=10))
