"""组合优化器：因子风险模型形式的 mean-variance QP（cvxpy/CLARABEL）。"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from factorzen.portfolio.constraints import build_constraints


@dataclass
class OptimizeResult:
    weights: np.ndarray | None
    status: str
    objective_value: float | None
    solve_seconds: float


def _psd(F: np.ndarray) -> np.ndarray:
    """对称化 + 特征值 clip 到 ≥0，保证 cvxpy quad_form 的 PSD 要求。"""
    F = (F + F.T) / 2.0
    vals, vecs = np.linalg.eigh(F)
    vals = np.clip(vals, 0.0, None)
    return (vecs * vals) @ vecs.T


def optimize_portfolio(alpha, risk_result, *, risk_aversion: float = 1.0,
                       constraint_config, solver: str = "CLARABEL") -> OptimizeResult:
    X = risk_result.factor_exposures.matrix          # (n, k)
    F = _psd(np.asarray(risk_result.factor_covariance))  # (k, k) PSD
    D = np.asarray(risk_result.specific_risk)        # (n,) std
    n = X.shape[0]
    w = cp.Variable(n)
    factor_var = cp.quad_form(X.T @ w, F)
    spec_var = cp.sum_squares(cp.multiply(D, w))
    objective = cp.Maximize(alpha @ w - risk_aversion * (factor_var + spec_var))
    cons = build_constraints(w, exposures=risk_result.factor_exposures, config=constraint_config)
    prob = cp.Problem(objective, cons)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=getattr(cp, solver))
    except cp.error.SolverError:
        return OptimizeResult(None, "solver_error", None, time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    if prob.status != "optimal" or w.value is None:
        return OptimizeResult(None, prob.status, None, dt)
    return OptimizeResult(np.asarray(w.value), prob.status, float(prob.value), dt)
