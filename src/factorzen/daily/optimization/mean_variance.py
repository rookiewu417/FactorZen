"""均值-方差优化器（cvxpy 实现）。"""
from __future__ import annotations

import numpy as np

from factorzen.core.logger import get_logger
from factorzen.daily.optimization.base import OptimizerConstraints, PortfolioOptimizer

logger = get_logger(__name__)


class MeanVarianceOptimizer(PortfolioOptimizer):
    """均值-方差优化：max wᵀμ - λ/2 · wᵀΣw。

    Args:
        risk_aversion: 风险厌恶系数 λ，越大越保守。
    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        self.risk_aversion = risk_aversion

    def solve(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizerConstraints,
    ) -> np.ndarray:
        import cvxpy as cp

        n = len(expected_returns)

        # Fallback weights
        fallback = (
            constraints.prev_weights
            if len(constraints.prev_weights) == n
            else np.full(n, 1.0 / n)
        )

        w = cp.Variable(n)
        objective = cp.Maximize(
            expected_returns @ w - (self.risk_aversion / 2) * cp.quad_form(w, cov_matrix)
        )

        cons = [
            w >= constraints.min_weight,
            w <= constraints.max_weight,
            cp.sum(w) <= constraints.net_exposure,
            cp.norm1(w) <= constraints.gross_exposure,
        ]
        if constraints.turnover_limit is not None and len(constraints.prev_weights) == n:
            cons.append(cp.norm1(w - constraints.prev_weights) <= constraints.turnover_limit)

        prob = cp.Problem(objective, cons)
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                result = np.array(w.value)
                result = np.clip(result, constraints.min_weight, constraints.max_weight)
                return result
        except Exception as e:
            logger.warning(f"MeanVarianceOptimizer 求解失败: {e}，回退等权/上期权重")
        return fallback
