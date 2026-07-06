# tests/test_portfolio_optimizer.py
import cvxpy as cp
import numpy as np

import factorzen.portfolio.optimizer as optimizer_mod
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


def test_optimize_accepts_optimal_inaccurate(monkeypatch):
    """CLARABEL 因数值精度返回 'optimal_inaccurate'（AlmostSolved）时 w.value 非 None，
    必须当可用解返回、而非丢弃成全零兜底——否则下游 sim(_SUCCESS_OPT_STATUSES 含
    optimal_inaccurate)会把'全零仓位'当真实清仓信号执行。与 daily/optimization 及集成
    测试(test_integration_mine_export_validate 断言 Σw≈1)口径对齐。"""
    orig_solve = cp.Problem.solve

    def fake_solve(self, *args, **kwargs):
        orig_solve(self, *args, **kwargs)  # 真实求解 → w.value 是可行解，status='optimal'
        self._status = "optimal_inaccurate"  # 降级模拟 AlmostSolved
        return self.value

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)

    r = _RiskResult()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = optimize_portfolio(alpha, r, risk_aversion=1.0,
                             constraint_config=ConstraintConfig(w_max=0.3))
    assert res.status == "optimal_inaccurate"
    assert res.weights is not None, "optimal_inaccurate 的可用解不应被丢弃"
    assert abs(res.weights.sum() - 1.0) < 1e-2, "应返回真实权重(Σw≈1)，而非全零兜底"


def test_non_psd_covariance_dcp_error_returns_error_status_not_raises(monkeypatch):
    """非PSD协方差矩阵在 prob.solve() 阶段抛 cp.error.DCPError(而非 SolverError)。

    真实场景:因子高度共线/协方差病态时，_psd() 的特征值裁剪可能仍残留浮点
    级别的非PSD残差。这里直接绕过 _psd()（monkeypatch 成恒等函数），喂一个
    对称但有负特征值的协方差矩阵，模拟"裁剪失效"。optimize_portfolio 必须
    捕获 DCPError 并返回 status="error"，而不是让异常未捕获地往外抛、崩掉
    整条 pipeline。
    """
    monkeypatch.setattr(optimizer_mod, "_psd", lambda F: F)

    r = _RiskResult()  # n=6, k=3
    # 对称但非PSD(含负特征值) —— 模拟 _psd() 裁剪后仍残留的病态残差
    bad_cov = np.diag([1.0, 1.0, -1.0])
    assert np.allclose(bad_cov, bad_cov.T)
    assert (np.linalg.eigvalsh(bad_cov) < 0).any()
    r.factor_covariance = bad_cov

    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])

    # 先确认这个协方差矩阵真的会在 solve() 阶段触发 DCPError（而不是别的异常/
    # 在更早阶段就报错），否则这个回归测试本身没有判别力。
    import factorzen.portfolio.constraints as constraints_mod
    X = r.factor_exposures.matrix
    w = cp.Variable(X.shape[0])
    factor_var = cp.quad_form(X.T @ w, bad_cov)
    spec_var = cp.sum_squares(cp.multiply(r.specific_risk, w))
    objective = cp.Maximize(alpha @ w - (factor_var + spec_var))
    cons = constraints_mod.build_constraints(
        w, exposures=r.factor_exposures, config=ConstraintConfig(w_max=0.3))
    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.DCPError:
        pass
    else:
        raise AssertionError("测试前提失效: bad_cov 未能在 solve() 阶段触发 DCPError")

    res = optimize_portfolio(alpha, r, constraint_config=ConstraintConfig(w_max=0.3))
    assert isinstance(res, OptimizeResult)
    assert res.weights is None
    assert res.status == "error"
