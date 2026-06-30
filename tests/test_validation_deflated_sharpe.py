import numpy as np


def test_strong_sharpe_significant():
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    # 高 IR、长样本、少试验 → 应显著
    dsr, p = deflated_sharpe(sharpe=0.15, n_trials=5, n_obs=500, sharpe_variance=0.0025)
    assert dsr > 0.95 and p < 0.05


def test_noise_sharpe_not_significant():
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    # IR≈0 → 不显著
    dsr, p = deflated_sharpe(sharpe=0.0, n_trials=100, n_obs=500, sharpe_variance=0.0025)
    assert p > 0.05


def test_more_trials_tightens():
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    # 同样观测 Sharpe，更多试验 → DSR 下降（多重检验收紧）
    dsr_few, _ = deflated_sharpe(0.12, n_trials=5, n_obs=500, sharpe_variance=0.0025)
    dsr_many, _ = deflated_sharpe(0.12, n_trials=1000, n_obs=500, sharpe_variance=0.0025)
    assert dsr_many < dsr_few


def test_expected_max_sharpe_grows_with_trials():
    from factorzen.validation.deflated_sharpe import expected_max_sharpe
    assert expected_max_sharpe(0.0025, 1000) > expected_max_sharpe(0.0025, 10)
