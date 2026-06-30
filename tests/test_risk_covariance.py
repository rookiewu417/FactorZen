"""协方差/特质风险测试：验证数学性质（对称、半正定、全正）。"""
import numpy as np


def test_factor_covariance_symmetric_psd():
    from factorzen.risk.covariance import estimate_factor_covariance

    rng = np.random.default_rng(0)
    fr = rng.standard_normal((120, 5))  # (T=120, K=5)
    cov = estimate_factor_covariance(fr, half_life=60, nw_lags=2)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-10)  # 对称
    assert np.linalg.eigvalsh(cov).min() >= -1e-8  # 半正定


def test_specific_risk_positive():
    from factorzen.risk.covariance import estimate_specific_risk

    rng = np.random.default_rng(0)
    resid = rng.standard_normal((120, 8))  # (T=120, N=8)
    sr = estimate_specific_risk(resid, half_life=60, shrinkage=0.3)
    assert sr.shape == (8,)
    assert (sr > 0).all()  # 特质风险全正


def test_eigenvector_adjustment_symmetric_same_shape():
    from factorzen.risk.covariance import eigenvector_adjustment

    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    cov = a @ a.T  # 半正定对称
    adj = eigenvector_adjustment(cov, n_simulations=200, seed=1)
    assert adj.shape == (4, 4)
    # eigenvector_adjustment 内部执行 (A + A.T) / 2，对称性达机器精度，与
    # factor_covariance 对称断言保持一致
    assert np.allclose(adj, adj.T, atol=1e-10)


def test_covariance_too_short_returns_identity():
    from factorzen.risk.covariance import estimate_factor_covariance

    cov = estimate_factor_covariance(np.zeros((1, 3)), half_life=60)
    assert cov.shape == (3, 3)
    # T < 2 时 estimate_factor_covariance 明确返回单位阵（见 covariance.py L40-42）
    assert np.allclose(cov, np.eye(3))
