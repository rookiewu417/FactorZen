import cvxpy as cp
import numpy as np

from factorzen.portfolio.constraints import ConstraintConfig, build_constraints
from factorzen.risk.exposures import ExposureMatrix


def _exposures(n=12, k=5):
    # factor_names: 1 风格(size) + 4 行业(ind_A/ind_B/ind_C/ind_D)，已去均值
    # 每组 3 只股票（共 4 组 12 只）；去均值后本组暴露 +3/4、其余 -1/4，列和为 0
    # 关键性质：ind_A+ind_B+ind_C = -ind_D ≠ 0，故 ind_A/B/C 三列线性独立（秩=3）
    names = ["size", "ind_A", "ind_B", "ind_C", "ind_D"]
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((n, k))
    # ind_A: 股票 0-2 属 A 组
    mat[:, 1] = [-1/4]*12
    mat[0:3, 1] = 3/4
    # ind_B: 股票 3-5 属 B 组
    mat[:, 2] = [-1/4]*12
    mat[3:6, 2] = 3/4
    # ind_C: 股票 6-8 属 C 组
    mat[:, 3] = [-1/4]*12
    mat[6:9, 3] = 3/4
    # ind_D: 股票 9-11 属 D 组
    mat[:, 4] = [-1/4]*12
    mat[9:12, 4] = 3/4
    return ExposureMatrix(codes=[f"{i}" for i in range(n)], factor_names=names, matrix=mat)


def _solve_with(constraints_fn):
    exp = _exposures()
    w = cp.Variable(exp.n_stocks)
    alpha = np.array([0.10, 0.05, 0.02, 0.08, 0.03, 0.01, 0.07, 0.04, 0.06,
                      0.09, 0.11, 0.03])
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
    # 4 个行业组（n=12），仅对 ind_A/B/C 施加中性约束（不含 ind_D）
    # 由于 ind_A+ind_B+ind_C = -ind_D ≠ 0，三列线性独立（秩=3），构成 3 个真正独立约束
    cfg = ConstraintConfig(w_max=0.5, neutral_factors=["ind_A", "ind_B", "ind_C"])
    prob, w = _solve_with(lambda w, e: build_constraints(w, exposures=e, config=cfg))
    assert prob.status == "optimal"
    exp = _exposures()
    ind_cols = [1, 2, 3]   # size=0, ind_A=1, ind_B=2, ind_C=3, ind_D=4
    neutral_exp = exp.matrix[:, ind_cols].T @ w.value
    assert np.abs(neutral_exp).max() < 1e-5          # 3 个独立行业暴露均 ≈ 0


def test_turnover_constraint():
    prev = np.array([1/12] * 12)
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


# ── 生产形态 raw one-hot 行业列测试（Fix 2：钉死真实行为 + 验证等权基准修复）────────


def _raw_onehot_exposures(n_industries=4, stocks_per_ind=3):
    """构造生产形态 raw one-hot 行业暴露矩阵（取值 0/1，未去均值）。"""
    n = n_industries * stocks_per_ind
    ind_names = [f"ind_{chr(65 + i)}" for i in range(n_industries)]  # ind_A/B/C/D
    mat = np.zeros((n, n_industries))
    for i in range(n_industries):
        mat[i * stocks_per_ind:(i + 1) * stocks_per_ind, i] = 1.0
    return ExposureMatrix(
        codes=[f"{i}" for i in range(n)],
        factor_names=ind_names,
        matrix=mat,
    ), ind_names


def test_raw_onehot_industry_neutral_no_bench_is_infeasible():
    """复现 Critical 根因：生产形态 raw one-hot 行业列 + neutral=所有 ind_ + bench=None
    (target=0) + long_only + Σw=1 → infeasible。

    数学根因：Σ_{k} (X_{ind_k}.T @ w) = Σw = 1，但所有行业暴露=0 要求 Σw=0，矛盾。
    """
    exp, ind_names = _raw_onehot_exposures()
    cfg = ConstraintConfig(
        w_max=1.0,
        neutral_factors=ind_names,
        benchmark_weights=None,   # target = 0（绝对零暴露）
        long_only=True,
    )
    w = cp.Variable(exp.n_stocks)
    cons = build_constraints(w, exposures=exp, config=cfg)
    prob = cp.Problem(cp.Maximize(cp.sum(w)), cons)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ("infeasible", "infeasible_inaccurate"), (
        f"期望 infeasible，实际 {prob.status}（raw one-hot + target=0 必不可行）"
    )


def test_raw_onehot_industry_neutral_with_equal_bench_is_feasible():
    """同 raw one-hot 行业列，传等权基准 → target = 等权行业暴露（各行业 1/n_ind），
    long_only + Σw=1 下可行（验证 Fix 1：CLI 传 bench_weights=np.full(n, 1/n)）。
    """
    exp, ind_names = _raw_onehot_exposures()
    n = exp.n_stocks
    bench_weights = np.full(n, 1.0 / n)
    cfg = ConstraintConfig(
        w_max=0.15,               # 单票上限 15%（等权 ≈ 8.3%，留有余量）
        neutral_factors=ind_names,
        benchmark_weights=bench_weights,
        long_only=True,
    )
    w = cp.Variable(n)
    alpha = np.arange(n, dtype=float) / n
    cons = build_constraints(w, exposures=exp, config=cfg)
    prob = cp.Problem(cp.Maximize(alpha @ w), cons)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status in ("optimal", "optimal_inaccurate"), (
        f"期望 optimal，实际 {prob.status}（raw one-hot + 等权基准应可行）"
    )
    # 验证行业暴露确实对齐基准（各行业总权重 ≈ 等权基准暴露）
    target = exp.matrix.T @ bench_weights
    actual = exp.matrix.T @ w.value
    np.testing.assert_allclose(actual, target, atol=1e-4,
                               err_msg="行业暴露未能对齐等权基准")
