import cvxpy as cp
import numpy as np

from factorzen.portfolio.constraints import ConstraintConfig, build_constraints
from factorzen.risk.exposures import ExposureMatrix


def _exposures(n=9, k=4):
    # factor_names: 1 风格(size) + 3 行业(ind_A/ind_B/ind_C)，已去均值
    # 每组 3 只股票；去均值后本组暴露 +2/3、其余 -1/3，满足列和为 0
    names = ["size", "ind_A", "ind_B", "ind_C"]
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [2/3,  2/3,  2/3, -1/3, -1/3, -1/3, -1/3, -1/3, -1/3]  # A 组
    mat[:, 2] = [-1/3, -1/3, -1/3,  2/3,  2/3,  2/3, -1/3, -1/3, -1/3]  # B 组
    mat[:, 3] = [-1/3, -1/3, -1/3, -1/3, -1/3, -1/3,  2/3,  2/3,  2/3]  # C 组
    return ExposureMatrix(codes=[f"{i}" for i in range(n)], factor_names=names, matrix=mat)


def _solve_with(constraints_fn):
    exp = _exposures()
    w = cp.Variable(exp.n_stocks)
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01, 0.07, 0.04, 0.06])
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
    # 3 个行业列已去均值，中性约束到 0 在 long_only+Σw=1 下可行
    # 真正验证多个独立中性约束：ind_A/ind_B/ind_C 的组合暴露 X_ind.T@w 均 ≈ 0
    cfg = ConstraintConfig(w_max=0.5, neutral_factors=["ind_A", "ind_B", "ind_C"])
    prob, w = _solve_with(lambda w, e: build_constraints(w, exposures=e, config=cfg))
    assert prob.status == "optimal"
    exp = _exposures()
    ind_cols = [1, 2, 3]
    neutral_exp = exp.matrix[:, ind_cols].T @ w.value
    assert np.abs(neutral_exp).max() < 1e-5          # 全部行业暴露 ≈ 0


def test_turnover_constraint():
    prev = np.array([1/9] * 9)
    cfg = ConstraintConfig(w_max=1.0, turnover_budget=0.2, prev_weights=prev)
    prob, w = _solve_with(lambda w, e: build_constraints(w, exposures=e, config=cfg))
    assert prob.status == "optimal"
    assert np.abs(w.value - prev).sum() < 0.2 + 1e-5  # L1 换手 ≤ budget


def test_infeasible_when_w_max_too_tight():
    # n * w_max < 1.0 → 满仓 Σw=1 与 box 约束矛盾，问题不可行
    n = 6
    w_max = 0.1  # 6 * 0.1 = 0.6 < 1.0
    names = ["size"]
    mat = np.zeros((n, 1))
    exp = ExposureMatrix(codes=[f"{i}" for i in range(n)], factor_names=names, matrix=mat)
    cfg = ConstraintConfig(w_max=w_max)
    w = cp.Variable(n)
    cons = build_constraints(w, exposures=exp, config=cfg)
    prob = cp.Problem(cp.Maximize(cp.sum(w)), cons)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ("infeasible", "infeasible_inaccurate")
