"""风险平价优化器（scipy 迭代求解）。"""
from __future__ import annotations

import numpy as np

from common.logger import get_logger
from daily.optimization.base import OptimizerConstraints, PortfolioOptimizer

logger = get_logger(__name__)


class RiskParityOptimizer(PortfolioOptimizer):
    """风险平价：各资产风险贡献相等。

    使用 scipy.optimize.minimize 求解：
        min Σ_i Σ_j (RC_i - RC_j)²
    where RC_i = w_i · (Σw)_i / wᵀΣw
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        expected_returns: np.ndarray,  # 风险平价不使用预期收益
        cov_matrix: np.ndarray,
        constraints: OptimizerConstraints,
    ) -> np.ndarray:
        from scipy.optimize import minimize

        n = len(expected_returns)
        fallback = (
            constraints.prev_weights
            if len(constraints.prev_weights) == n
            else np.full(n, 1.0 / n)
        )

        def _risk_contributions(w: np.ndarray) -> np.ndarray:
            portfolio_var = w @ cov_matrix @ w
            if portfolio_var <= 0:
                return np.zeros(n)
            marginal = cov_matrix @ w
            return w * marginal / portfolio_var

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contributions(w)
            diff = rc[:, None] - rc[None, :]
            return float(np.sum(diff**2))

        w0 = np.full(n, 1.0 / n)
        bounds = [(max(constraints.min_weight, 1e-6), constraints.max_weight)] * n
        scipy_cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        try:
            result = minimize(
                _objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_cons,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            if result.success or result.fun < 1e-6:
                w = np.clip(result.x, max(constraints.min_weight, 0), constraints.max_weight)
                total = w.sum()
                if total > 1e-8:
                    return w / total
        except Exception as e:
            logger.warning(f"RiskParityOptimizer 求解失败: {e}，回退等权")
        return fallback
