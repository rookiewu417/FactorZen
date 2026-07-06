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
    """因子形式 mean-variance QP：max alpha@w − risk_aversion * portfolio_variance。

    注意：这里的 ``risk_aversion`` 缩放约定与 ``daily/optimization/mean_variance.py``
    **不同**——那边的目标函数是 ``wᵀμ − (risk_aversion/2)·wᵀΣw``（含 1/2），这里没有
    这个 1/2。两者是各自独立维护的组合构建实现（见项目 CLAUDE.md「命名空间分离」），
    同一个 ``risk_aversion`` 数值在两边对应的实际风险惩罚强度相差 2 倍，不能跨模块
    直接套用调参经验。
    """
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
    except (cp.error.SolverError, cp.error.DCPError):
        # SolverError：求解器本身失败；DCPError：传入的协方差矩阵在 prob.solve()
        # 阶段被判定非 PSD(例如 _psd() 特征值裁剪后仍残留浮点级别的非PSD残差)。
        # 两者都不是"程序错误"，而是"这组输入解不出来"，统一按 infeasible/error
        # 既有设计处理：不返垃圾、不让异常未捕获地往外抛崩掉整条 pipeline。
        return OptimizeResult(None, "error", None, time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    # optimal_inaccurate(CLARABEL AlmostSolved)也是可用解，必须接受——否则丢成全零
    # 兜底后，下游 sim(_SUCCESS_OPT_STATUSES 含 optimal_inaccurate)会把它当真实清仓
    # 信号执行。与 daily/optimization/mean_variance 及集成测试口径一致。
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return OptimizeResult(None, prob.status, None, dt)
    return OptimizeResult(np.asarray(w.value), prob.status, float(prob.value), dt)
