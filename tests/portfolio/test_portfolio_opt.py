"""test_portfolio_constraints.py：无 module docstring 的测试。
test_portfolio_optimizer.py：无 module docstring 的测试。
test_portfolio_pipeline.py：无 module docstring 的测试。
test_portfolio_report.py：Tests for portfolio_report.py — M7 成果展示页。
test_integration_portfolio_sim.py：集成测试：M4 组合构建（run_portfolio）→ M7 模拟交易（run_portfolio_simulation）贯通。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import cvxpy as cp
import numpy as np
import polars as pl
import pytest

import factorzen.portfolio.optimizer as optimizer_mod
from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.portfolio.constraints import ConstraintConfig, build_constraints
from factorzen.portfolio.optimizer import OptimizeResult, optimize_portfolio
from factorzen.reports.portfolio_report import generate_portfolio_report
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult
from factorzen.sim.engine import run_portfolio_simulation


# ==== 来自 test_portfolio_constraints.py ====
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

# ==== 来自 test_portfolio_optimizer.py ====
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

# ==== 来自 test_portfolio_pipeline.py ====
def _risk_result__pipeline(n=6, k=3):
    rng = np.random.default_rng(2)
    names = ["size", "ind_A", "ind_B"]
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [1, 1, 1, 0, 0, 0]
    mat[:, 2] = [0, 0, 0, 1, 1, 1]
    F = rng.standard_normal((k, k))
    F = F @ F.T * 0.01
    return RiskModelResult(
        factor_exposures=ExposureMatrix([f"{i:06d}.SZ" for i in range(n)], names, mat),
        factor_covariance=F, specific_risk=np.full(n, 0.1), factor_names=names)

def test_run_portfolio_writes_products(tmp_path: Path):
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4, out_dir=str(tmp_path), run_id="t1")
    run_dir = Path(res["run_dir"])
    for f in ["weights.parquet", "attribution.csv", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f
    assert res["status"] == "optimal"
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "optimal" and "objective" in m

def test_run_portfolio_attribution_placeholder_flag(tmp_path: Path):
    """建仓时 stock_returns=zeros + factor_returns_latest={} → manifest 含占位标注。"""
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.zeros(6),       # 建仓时点全 0
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={},        # 无因子收益
                        risk_aversion=1.0, w_max=0.4,
                        out_dir=str(tmp_path), run_id="placeholder")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["return_attribution_available"] is False, (
        "stock_returns=zeros + factor_returns={}  时应标注归因不可用"
    )
    assert "return_attribution_note" in m and m["return_attribution_note"] is not None

def test_run_portfolio_attribution_available_when_returns_provided(tmp_path: Path):
    """传入真实收益时 return_attribution_available=True，note=None。"""
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4,
                        out_dir=str(tmp_path), run_id="real_returns")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["return_attribution_available"] is True
    assert m["return_attribution_note"] is None

def test_run_portfolio_records_signal_date(tmp_path: Path):
    """run_portfolio 传 signal_date 后，manifest.json 应记录该字段供 M7 sim 对齐。"""
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.zeros(6),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={},
                        risk_aversion=1.0, w_max=0.4,
                        out_dir=str(tmp_path), run_id="sig_date",
                        signal_date="2024-12-31")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m.get("signal_date") == "2024-12-31", (
        f"manifest.json 应含 signal_date='2024-12-31', 实际: {m.get('signal_date')!r}"
    )

def test_run_portfolio_infeasible_does_not_crash(tmp_path: Path):
    """w_max=0.1 → n*w_max=0.6 < 1，违反 Σw=1，优化器返回 infeasible，pipeline 不崩溃。"""
    rr = _risk_result__pipeline()  # n=6 stocks
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    # w_max=0.1 with 6 stocks: max feasible sum = 0.6 < 1 → infeasible
    res = run_portfolio(
        alpha, rr,
        codes=rr.factor_exposures.codes,
        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
        sectors=["A", "A", "A", "B", "B", "B"],
        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
        risk_aversion=1.0, w_max=0.1, out_dir=str(tmp_path), run_id="infeasible",
    )
    run_dir = Path(res["run_dir"])

    # 4 产物文件必须存在
    for f in ["weights.parquet", "attribution.csv", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f"missing: {f}"

    # status 非 optimal
    assert res["status"] != "optimal"

    # weights.parquet 的 target_weight 全 0
    df_w = pl.read_parquet(run_dir / "weights.parquet")
    assert (df_w["target_weight"] == 0.0).all(), "infeasible 时 target_weight 应全 0"

    # attribution.csv 为空（0 行）
    df_a = pl.read_csv(run_dir / "attribution.csv")
    assert len(df_a) == 0, "infeasible 时 attribution 应为空"

    # manifest.json status 非 optimal
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] != "optimal"

def test_run_portfolio_manifest_has_reproducibility_fields(tmp_path: Path):
    """manifest.json 应含 command/git_dirty/pixi_lock_sha256/schema_version（复用 core.experiment 的
    build_manifest_base，而非各自手写精简版 manifest）。"""
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4, out_dir=str(tmp_path), run_id="repro1")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())

    assert m["schema_version"] == "1"
    assert isinstance(m["git_dirty"], bool)
    assert isinstance(m["pixi_lock_sha256"], str) and m["pixi_lock_sha256"]
    assert isinstance(m["command"], list) and m["command"]
    assert m.get("git_sha")
    # 原有字段不应回归丢失
    assert m["run_id"] == "repro1"
    assert "duration_seconds" in m

def test_run_portfolio_manifest_command_override(tmp_path: Path):
    """显式传 command 时应原样记录，供复现当时具体怎么跑的。"""
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4, out_dir=str(tmp_path), run_id="repro2",
                        command=["fz", "portfolio", "build", "--w-max", "0.4"])
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["command"] == ["fz", "portfolio", "build", "--w-max", "0.4"]

def test_run_portfolio_warns_when_turnover_set_without_prev_weights(tmp_path: Path, capsys):
    """L2：给了 turnover_budget 但无 prev_weights → 换手约束静默丢弃，须告警。"""
    rr = _risk_result__pipeline()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                  stock_returns=np.zeros(6), sectors=["A"] * 6,
                  factor_returns_latest={}, risk_aversion=1.0, w_max=0.4,
                  turnover_budget=0.3, out_dir=str(tmp_path), run_id="tw")
    err = capsys.readouterr().err
    assert "turnover" in err and "不生效" in err

# ==== 来自 test_portfolio_report.py ====
@pytest.fixture()
def base_metrics() -> dict:
    return {
        "ann_ret": 0.12,
        "ann_vol": 0.18,
        "sharpe": 0.67,
        "max_dd": -0.15,
        "ann_turnover": 3.2,
        "total_cost": 0.01,
    }

@pytest.fixture()
def attribution_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "type": ["brinson_allocation", "factor_return"],
            "key": ["银行", "size"],
            "value": [0.01, 0.005],
        }
    )

@pytest.fixture()
def risk_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "metric": ["total_risk", "factor_risk", "specific_risk"],
            "value": [0.18, 0.15, 0.10],
        }
    )

@pytest.fixture()
def manifest() -> dict:
    return {"n_holdings": 87, "status": "optimal"}

# ── core rendering test (from brief) ─────────────────────────────────────

def test_generate_portfolio_report_html_has_sections(
    base_metrics, attribution_df, risk_df, manifest
):
    """复刻 brief 里要求的断言，不依赖真实回测。"""
    html = generate_portfolio_report(
        None,
        metrics=base_metrics,
        attribution_df=attribution_df,
        risk_summary_df=risk_df,
        portfolio_manifest=manifest,
    )
    assert isinstance(html, str) and len(html) > 500

    # 关键 section 存在
    assert "sharpe" in html.lower() or "夏普" in html
    assert "0.67" in html or "67" in html  # 绩效数值渲染进去
    assert "总风险" in html or "total_risk" in html or "0.18" in html
    # merged from deleted attribution/manifest smoke tests
    assert "银行" in html or "brinson" in html.lower() or "归因" in html
    assert "87" in html or "optimal" in html or "持仓" in html

# ── additional coverage ───────────────────────────────────────────────────

def test_html_has_doctype(base_metrics):
    """输出是合法 HTML 文档，以 DOCTYPE 开头。"""
    html = generate_portfolio_report(None, metrics=base_metrics)
    assert html.strip().startswith("<!DOCTYPE") or html.strip().startswith("<!")

def test_no_chart_when_sim_result_none(base_metrics):
    """sim_result=None 时不应有 base64 图表字符串（无 <img src='data:image）。"""
    html = generate_portfolio_report(None, metrics=base_metrics)
    # If there are no charts, there should be no base64 img tags
    assert 'data:image/png;base64' not in html

def test_capability_overview_cards_present(base_metrics):
    """修复5：能力总览卡按能力命名展示，不暴露内部里程碑代号（M1-M6）。

    展示页是对外可见正文，项目约定（CLAUDE.md）要求对外不暴露 M0-M7 里程碑代号；
    旧版直接把 "M1 因子挖掘".."M6 多 Agent 协作" 写进卡片文案，违反该约定。
    """
    html = generate_portfolio_report(None, metrics=base_metrics)
    # 按能力命名的文案应存在
    assert any(
        token in html
        for token in ("因子挖掘", "防过拟合", "风险模型", "组合优化", "Agent", "模拟交易")
    )
    # 内部里程碑代号字样不应再出现在对外可见正文中
    for milestone_token in ("M1", "M2", "M3", "M4", "M5", "M6"):
        assert milestone_token not in html, (
            f"展示页正文不应出现内部里程碑代号 {milestone_token!r}（对外应按能力命名）"
        )

# ==== 来自 test_integration_portfolio_sim.py ====
def _risk_result__integration(n: int = 6, k: int = 3) -> RiskModelResult:
    """与 tests/test_portfolio_pipeline.py::_risk_result__integration 同构（n=6 时行业各占一半）。"""
    rng = np.random.default_rng(2)
    names = ["size", "ind_A", "ind_B"]
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [1, 1, 1, 0, 0, 0]
    mat[:, 2] = [0, 0, 0, 1, 1, 1]
    F = rng.standard_normal((k, k))
    F = F @ F.T * 0.01
    return RiskModelResult(
        factor_exposures=ExposureMatrix([f"{i:06d}.SZ" for i in range(n)], names, mat),
        factor_covariance=F, specific_risk=np.full(n, 0.1), factor_names=names)

def _fake_daily(codes: list[str], start: str = "20230101", end: str = "20230228") -> pl.DataFrame:
    """构造 mock 日线数据（不连接真实数据源），真正使用 start/end（覆盖整段回测窗口）。"""
    start_d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()
    dates = pl.date_range(start_d, end_d, "1d", eager=True)
    rng = np.random.default_rng(0)
    rows = []
    for c in codes:
        for dt in dates:
            rows.append({
                "trade_date": dt, "ts_code": c,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "pre_close": 10.0, "change": 0.0, "pct_chg": float(rng.normal(0, 1)),
                # amount 需远大于 BacktestConfig 默认 initial_capital(1e8) ×
                # max_participation_rate(0.05) 隐含的 ADV 门槛，否则 fast path 会按
                # 20 日 ADV 参与率把单日调仓幅度限制到远小于目标权重（真实的流动性约束
                # 行为，非 bug），导致建仓后长期停留在接近全现金的状态，测不出本测试要
                # 验证的"跳过 infeasible 信号"效果。
                "vol": 1e6, "amount": 1e10,
            })
    return pl.DataFrame(rows)

def _build_portfolio(rr: RiskModelResult, *, w_max: float, out_dir: str, run_id: str,
                     signal_date: str) -> dict:
    codes = rr.factor_exposures.codes
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    return run_portfolio(
        alpha, rr, codes=codes,
        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
        sectors=["A", "A", "A", "B", "B", "B"],
        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
        risk_aversion=1.0, w_max=w_max, out_dir=out_dir, run_id=run_id,
        signal_date=signal_date,
    )

def test_portfolio_build_to_sim_happy_path(tmp_path: Path):
    """run_portfolio() 真实落盘的 run_dir 直接喂给 run_portfolio_simulation()：
    不抛异常，产出非空 nav 与完整 metrics 字段（含 total_cost/ann_turnover）。
    """
    rr = _risk_result__integration()
    codes = rr.factor_exposures.codes
    build_res = _build_portfolio(
        rr, w_max=0.4, out_dir=str(tmp_path / "portfolios"), run_id="link_happy",
        signal_date="2023-01-05",
    )
    assert build_res["status"] == "optimal"

    daily = _fake_daily(codes, start="20230101", end="20230228")
    sim_res = run_portfolio_simulation(
        [build_res["run_dir"]], daily, out_dir=str(tmp_path / "sim"), run_id="sim_happy",
    )

    run_dir = Path(sim_res["run_dir"])
    for f in ["nav.parquet", "metrics.json", "manifest.json"]:
        assert (run_dir / f).exists(), f"missing: {f}"

    nav_df = pl.read_parquet(run_dir / "nav.parquet")
    assert not nav_df.is_empty(), "串联后 nav 不应为空"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    for k in ("ann_ret", "ann_vol", "sharpe", "max_dd", "avg_turnover", "total_cost", "ann_turnover"):
        assert k in metrics, f"metrics.json 缺少字段: {k}"

    for k in ("run_dir", "sharpe", "max_dd", "ann_ret"):
        assert k in sim_res, f"run_portfolio_simulation 返回值缺少字段: {k}"

    sim_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert sim_manifest["n_signals"] == 1

def test_portfolio_build_infeasible_status_not_treated_as_valid_signal(tmp_path: Path):
    """一个 optimal + 一个 infeasible（w_max 过小导致约束不可行）的 run_dir 一起喂给 sim：
    infeasible 那次 run_portfolio() 会把全零持仓兜底写盘（status != optimal），
    sim 必须跳过它、不能当成"清仓"信号执行——否则有效仓位会被这个假信号错误抹平。
    """
    rr = _risk_result__integration()  # n=6
    codes = rr.factor_exposures.codes

    valid = _build_portfolio(
        rr, w_max=0.4, out_dir=str(tmp_path / "portfolios"), run_id="valid1",
        signal_date="2023-01-05",
    )
    assert valid["status"] == "optimal"

    # w_max=0.1 时 6 只股票的最大可行仓位 = 0.6 < 1，Σw=1 无法满足 → infeasible。
    infeasible = _build_portfolio(
        rr, w_max=0.1, out_dir=str(tmp_path / "portfolios"), run_id="infeasible1",
        signal_date="2023-01-20",
    )
    assert infeasible["status"] != "optimal"
    # infeasible 的兜底权重必须全零（否则下面的行为验证会失去意义）。
    infeasible_w = pl.read_parquet(Path(infeasible["run_dir"]) / "weights.parquet")
    assert (infeasible_w["target_weight"] == 0.0).all()

    daily = _fake_daily(codes, start="20230101", end="20230228")
    sim_res = run_portfolio_simulation(
        [valid["run_dir"], infeasible["run_dir"]], daily,
        out_dir=str(tmp_path / "sim_skip"), run_id="sim_skip",
    )

    nav_df = pl.read_parquet(Path(sim_res["run_dir"]) / "nav.parquet")
    assert not nav_df.is_empty(), "有效信号（valid1）应正常执行，nav 不应整体为空"

    # 若 infeasible 的全零兜底权重被误当有效清仓信号执行，2023-01-21 起 cash_weight
    # 会跳升到接近 1.0（全部清仓为现金）；跳过后应继续持有 valid1 建立的仓位，
    # cash_weight 应保持在低位（组合优化约束 Σw=1，建仓后接近满仓）。
    after = nav_df.filter(pl.col("trade_date") >= date(2023, 1, 21))
    assert after.height > 0
    assert (after["cash_weight"] < 0.5).all(), (
        "infeasible run 的全零兜底持仓被当成了有效清仓信号执行（cash_weight 跳升到接近 1.0）"
    )

def test_portfolio_build_only_infeasible_run_raises(tmp_path: Path):
    """所有 run_dir 都是 infeasible 兜底时，sim 应彻底找不到有效信号并明确报错，
    而不是静默产出一份"看似正常但其实全零仓位"的净值曲线。
    """
    rr = _risk_result__integration()
    codes = rr.factor_exposures.codes

    infeasible = _build_portfolio(
        rr, w_max=0.1, out_dir=str(tmp_path / "portfolios"), run_id="only_bad",
        signal_date="2023-01-05",
    )
    assert infeasible["status"] != "optimal"

    daily = _fake_daily(codes, start="20230101", end="20230228")
    with pytest.raises(ValueError, match="no portfolio weights"):
        run_portfolio_simulation(
            [infeasible["run_dir"]], daily,
            out_dir=str(tmp_path / "sim_only_bad"), run_id="only_bad_sim",
        )

