"""最大夏普比率优化器（cvxpy 参数化变换）。"""
from __future__ import annotations

import numpy as np

from factorzen.core.logger import get_logger
from factorzen.daily.optimization.base import (
    OptimizerConstraints,
    PortfolioOptimizer,
    unsupported_constraint_warnings,
)

logger = get_logger(__name__)


class MaxSharpeOptimizer(PortfolioOptimizer):
    """最大夏普比率（切线组合）。

    用 Markowitz 参数化变换将分式规划转为 QP：
        令 y = w / (μᵀw), 求解 min yᵀΣy, s.t. μᵀy=1, y>=0
    """

    def solve(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizerConstraints,
    ) -> np.ndarray:
        import cvxpy as cp

        n = len(expected_returns)
        unsupported = unsupported_constraint_warnings(constraints)
        if unsupported:
            logger.warning(
                "MaxSharpeOptimizer 不施加以下约束（静默忽略，净暴露恒归一到 1、换手不受限）："
                "%s；如需这些约束请改用 MeanVarianceOptimizer",
                unsupported,
            )
        fallback = (
            constraints.prev_weights
            if len(constraints.prev_weights) == n
            else np.full(n, 1.0 / n)
        )

        # 如果预期收益全非正，回退等权
        if np.all(expected_returns <= 0):
            return fallback

        y = cp.Variable(n, nonneg=True)
        objective = cp.Minimize(cp.quad_form(y, cov_matrix))
        cons = [
            expected_returns @ y == 1.0,
            y >= constraints.min_weight * cp.sum(y),
            y <= constraints.max_weight * cp.sum(y),
        ]
        prob = cp.Problem(objective, cons)
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None:
                y_val = np.maximum(y.value, 0)
                total = y_val.sum()
                if total > 1e-8:
                    w = y_val / total
                    w = np.clip(w, constraints.min_weight, constraints.max_weight)
                    s = w.sum()
                    if s > 1e-8:
                        return w / s
        except Exception as e:
            logger.warning(f"MaxSharpeOptimizer 求解失败: {e}，回退等权/上期权重")
        return fallback
