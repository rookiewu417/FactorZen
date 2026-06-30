import cvxpy as cp
import numpy as np

from factorzen.portfolio.constraints import ConstraintConfig, build_constraints
from factorzen.risk.exposures import ExposureMatrix


def _exposures(n=6, k=3):
    # factor_names: 1 风格(size) + 2 行业(ind_A/ind_B)
    names = ["size", "ind_A", "ind_B"]
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]   # 前3只 A 行业（已中心化）
    mat[:, 2] = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]   # 后3只 B 行业（已中心化）
    return ExposureMatrix(codes=[f"{i}" for i in range(n)], factor_names=names, matrix=mat)


def _solve_with(constraints_fn):
    exp = _exposures()
    w = cp.Variable(exp.n_stocks)
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    cons = constraints_fn(w, exp)
    prob = cp.Problem(cp.Maximize(alpha @ w), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob, w


def test_box_and_budget():
    cfg = ConstraintConfig(w_max=0.3)
    prob, w = _solve_with(lambda w, e: build_constraints(w, exposures=e, config=cfg))
    assert prob.status == "optimal"
    assert abs(w.value.sum() - 1.0) < 1e-6           # budget Σw=1
    assert (w.value >= -1e-7).all() and (w.value <= 0.3 + 1e-6).all()  # box


def test_industry_neutral_to_zero():
    # 行业中性到 0：组合在 ind_A/ind_B 暴露 == 0
    cfg = ConstraintConfig(w_max=0.5, neutral_factors=["ind_A", "ind_B"])
    prob, w = _solve_with(lambda w, e: build_constraints(w, exposures=e, config=cfg))
    assert prob.status == "optimal"
    exp = _exposures()
    ind_cols = [1, 2]
    neutral_exp = exp.matrix[:, ind_cols].T @ w.value
    assert np.abs(neutral_exp).max() < 1e-5          # 中性暴露≈0


def test_turnover_constraint():
    prev = np.array([1/6] * 6)
    cfg = ConstraintConfig(w_max=1.0, turnover_budget=0.2, prev_weights=prev)
    prob, w = _solve_with(lambda w, e: build_constraints(w, exposures=e, config=cfg))
    assert prob.status == "optimal"
    assert np.abs(w.value - prev).sum() < 0.2 + 1e-5  # L1 换手 ≤ budget
