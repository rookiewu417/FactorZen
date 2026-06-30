import numpy as np


def test_positive_ic_ci_above_zero():
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    rng = np.random.default_rng(0)
    ic = rng.normal(0.05, 0.02, 250)  # 明显正 IC
    lo, hi = block_bootstrap_ic_ci(ic, seed=1)
    assert lo > 0 and hi > lo


def test_noise_ic_ci_straddles_zero():
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    rng = np.random.default_rng(0)
    ic = rng.normal(0.0, 0.05, 250)  # 噪声 IC
    lo, hi = block_bootstrap_ic_ci(ic, seed=1)
    assert lo < 0 < hi


def test_too_short_returns_nan():
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci
    lo, hi = block_bootstrap_ic_ci(np.array([0.1, 0.2]), block_size=10)
    assert np.isnan(lo) and np.isnan(hi)
