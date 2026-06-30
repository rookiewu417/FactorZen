# tests/test_portfolio_optimizer.py
import numpy as np

from factorzen.portfolio.constraints import ConstraintConfig
from factorzen.portfolio.optimizer import OptimizeResult, optimize_portfolio
from factorzen.risk.exposures import ExposureMatrix


class _RiskResult:
    """手搓最小 RiskModelResult（仅优化器需要的 3 字段）。"""
    def __init__(self, n=6, k=3):
        rng = np.random.default_rng(1)
        names = ["size", "ind_A", "ind_B"]
        mat = rng.standard_normal((n, k))
        mat[:, 1] = [1, 1, 1, 0, 0, 0]
        mat[:, 2] = [0, 0, 0, 1, 1, 1]
        self.factor_exposures = ExposureMatrix([f"{i}" for i in range(n)], names, mat)
        F = rng.standard_normal((k, k))
        self.factor_covariance = F @ F.T * 0.01
        self.specific_risk = np.full(n, 0.1)
        self.factor_names = names


def test_optimize_returns_optimal_weights():
    r = _RiskResult()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = optimize_portfolio(alpha, r, risk_aversion=1.0,
                             constraint_config=ConstraintConfig(w_max=0.3))
    assert isinstance(res, OptimizeResult)
    assert res.status == "optimal"
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert (res.weights <= 0.3 + 1e-6).all()
    assert res.objective_value is not None


def test_infeasible_returns_none_not_garbage():
    """矛盾约束(w_max 太小无法满仓) → infeasible，weights=None，不返回垃圾。"""
    r = _RiskResult(n=6)
    alpha = np.ones(6) * 0.05
    # 6 只股票，单票上限 0.1 → 最多 0.6 < 1.0 满仓 → infeasible
    res = optimize_portfolio(alpha, r, constraint_config=ConstraintConfig(w_max=0.1))
    assert res.weights is None
    assert res.status != "optimal"          # infeasible/unbounded 等
