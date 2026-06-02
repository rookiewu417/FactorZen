"""组合优化器单元测试。"""
from __future__ import annotations

import numpy as np

from factorzen.daily.optimization.base import OptimizerConstraints
from factorzen.daily.optimization.covariance import (
    ewma_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)
from factorzen.daily.optimization.max_sharpe import MaxSharpeOptimizer
from factorzen.daily.optimization.mean_variance import MeanVarianceOptimizer
from factorzen.daily.optimization.risk_parity import RiskParityOptimizer


def _equal_cov(n: int, sigma: float = 0.01) -> np.ndarray:
    """n 个独立资产的对角协方差矩阵（等波动率）。"""
    return np.eye(n) * sigma**2


def _default_cons(n: int, max_weight: float = 1.0) -> OptimizerConstraints:
    return OptimizerConstraints(max_weight=max_weight, min_weight=0.0, gross_exposure=1.0, net_exposure=1.0)


class TestMeanVarianceOptimizer:
    def test_single_asset_full_weight(self):
        """单资产时权重应为 1.0。"""
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        mu = np.array([0.01])
        cov = np.array([[0.0001]])
        cons = _default_cons(1)
        w = opt.solve(mu, cov, cons)
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 0.05

    def test_max_weight_respected(self):
        """max_weight 约束必须严格生效。"""
        opt = MeanVarianceOptimizer(risk_aversion=0.1)
        n = 5
        mu = np.ones(n) * 0.01
        mu[0] = 0.1  # 第一个资产预期收益最高
        cov = _equal_cov(n)
        cons = _default_cons(n, max_weight=0.3)
        w = opt.solve(mu, cov, cons)
        assert np.all(w <= 0.3 + 1e-6), f"max_weight 违反: {w}"

    def test_weights_nonnegative(self):
        """long_only 情形下权重应非负。"""
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        n = 4
        mu = np.array([0.01, 0.02, -0.01, 0.005])
        cov = _equal_cov(n)
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert np.all(w >= -1e-6), f"负权重: {w}"

    def test_fallback_on_infeasible(self):
        """不可行问题时应返回有效的 fallback 权重。"""
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        n = 3
        mu = np.ones(n) * 0.01
        # 构造奇异协方差（不一定触发失败，但结果应合法）
        cov = np.zeros((n, n))
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert len(w) == n
        assert np.all(np.isfinite(w))


class TestRiskParityOptimizer:
    def test_equal_vol_equal_weight(self):
        """等波动率资产时风险平价退化为等权。"""
        opt = RiskParityOptimizer()
        n = 4
        cov = _equal_cov(n, sigma=0.01)
        mu = np.ones(n) * 0.01
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert len(w) == n
        np.testing.assert_allclose(
            w, np.full(n, 1.0 / n), atol=0.02, err_msg="等波动率时风险平价应接近等权"
        )

    def test_high_vol_lower_weight(self):
        """高波动率资产应获得更低权重。"""
        opt = RiskParityOptimizer()
        n = 2
        cov = np.diag([0.01**2, 0.03**2])  # 第二个波动率 3 倍
        mu = np.ones(n) * 0.01
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert w[0] > w[1], f"高波动资产权重应更低: {w}"

    def test_weights_sum_to_one(self):
        """权重之和应约为 1。"""
        opt = RiskParityOptimizer()
        n = 5
        cov = _equal_cov(n)
        mu = np.ones(n) * 0.01
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert abs(w.sum() - 1.0) < 0.01


class TestMaxSharpeOptimizer:
    def test_concentrates_on_high_return(self):
        """高预期收益资产权重应显著更高。"""
        opt = MaxSharpeOptimizer()
        n = 3
        mu = np.array([0.001, 0.01, 0.001])
        cov = _equal_cov(n)
        cons = _default_cons(n, max_weight=0.9)
        w = opt.solve(mu, cov, cons)
        assert w[1] > w[0] and w[1] > w[2], f"高收益资产权重应最高: {w}"

    def test_negative_returns_fallback(self):
        """全部预期收益非正时应返回合法 fallback。"""
        opt = MaxSharpeOptimizer()
        n = 3
        mu = np.array([-0.01, -0.02, -0.005])
        cov = _equal_cov(n)
        cons = _default_cons(n)
        w = opt.solve(mu, cov, cons)
        assert len(w) == n
        assert np.all(np.isfinite(w))


class TestCovarianceEstimators:
    def test_sample_covariance_shape(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, (100, 5))
        cov = sample_covariance(returns)
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T), "协方差矩阵应对称"

    def test_ewma_covariance_shape(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, (60, 4))
        cov = ewma_covariance(returns, halflife=20)
        assert cov.shape == (4, 4)
        assert np.allclose(cov, cov.T)

    def test_ledoit_wolf_psd(self):
        """Ledoit-Wolf 协方差矩阵应为正半定。"""
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, (50, 5))
        cov = ledoit_wolf_shrinkage(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10), f"协方差矩阵含负特征值: {eigenvalues.min()}"
