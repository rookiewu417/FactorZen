# M4 · 组合优化与归因 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 α 信号通过带约束凸优化（cvxpy，用 M3 因子风险模型）变成目标组合，并用 Brinson + 风险因子归因解释收益来源。

**Architecture:** 新建 `portfolio/`（优化器 + 约束）+ `attribution/`（Brinson + 风险因子归因）+ pipeline + CLI，复用 M3 风险模型。**与现有 `daily/optimization/`（单因子研究流，全 Σ 矩阵、无中性、静默 fallback）命名空间分离**——M4 是组合构建流（因子形式 QP、M3 中性约束、显式求解状态），接口/风险形式/归因方法均不兼容，故不改现有代码。

**Tech Stack:** Python 3.10–3.12 · **cvxpy（已依赖，CLARABEL solver）** · numpy · polars · 复用 M3 `RiskModel`/`RiskModelResult`/`ExposureMatrix`。

## Global Constraints

- **与现有 `daily/optimization/` 分离**：M4 新建 `portfolio/` + `attribution/`，**不改** `daily/optimization/`（其 `MeanVarianceOptimizer` 用全 Σ 矩阵、`solve()→ndarray` 静默 fallback、`OptimizerConstraints` 无中性，与 M4 不兼容）。可复用 `daily/evaluation/attribution.py` 的 `aggregate_positions_to_sectors`（辅助）、借鉴 `mean_variance.py` 的 cvxpy norm1 写法。
- **cvxpy 已依赖**（无新增）；solver 用 **`cp.CLARABEL`**（项目已用，非 ECOS）。
- **风险项 = M3 因子模型形式**：`risk(w) = cp.quad_form(X.T @ w, F) + cp.sum_squares(cp.multiply(D, w))`，X=`result.factor_exposures.matrix (n,k)`、F=`result.factor_covariance (k,k)`、D=`result.specific_risk (n,)` 标准差。**F 须 PSD**——求解前 `F = (F + F.T)/2` 对称化 + 特征值 clip 到 ≥0。
- **行业/风格列区分**：`factor_names` = 8 风格名（`size/value/momentum/volatility/liquidity/quality/growth/leverage`，无前缀）+ 行业列（`ind_` 前缀）。中性约束用 `mask = [n.startswith("ind_") for n in factor_names]` 选列。
- **求解稳定性（验收核心）**：cvxpy `prob.status` 非 `"optimal"`（含 `"optimal_inaccurate"` 视情况）→ `OptimizeResult.weights = None` + 记录 status，**绝不返回垃圾权重**（与现有静默 fallback 相反）。
- **归因守恒**：Brinson `Σ(配置+选股+交互) ≈ active_return`；风险因子归因因子贡献与 M3 `decompose_risk` 一致（跨函数验证）。两种归因口径不同、并列不对账。
- **环境**：`pixi run pytest` / `ruff check`；polars 1.41.2；cvxpy 1.9.1。
- **提交**：conventional commits；作者 `rookiewu417 <1007372080@qq.com>`；每 task 只 `git add` 自己的文件（工作区有无关 M0 改动，**绝不** `-A`）。
- **测试**：全 mock 离线（小 universe + 构造 α/F/X/D），cvxpy 确定性；避免恒真（值断言、约束满足容差、守恒跨函数验证、infeasible 反例）。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/portfolio/__init__.py` | 包标记 | 1 |
| `src/factorzen/portfolio/constraints.py` | `ConstraintConfig` + `build_constraints`（box/budget/中性/换手） | 1 |
| `src/factorzen/portfolio/optimizer.py` | `optimize_portfolio` + `OptimizeResult`（因子形式 QP + 显式 status） | 2 |
| `src/factorzen/attribution/__init__.py` | 包标记 | 3 |
| `src/factorzen/attribution/risk_attribution.py` | `risk_factor_attribution`（M3 因子收益/风险归因） | 3 |
| `src/factorzen/attribution/brinson.py` | `brinson_attribution`（M4 版，股票级输入 + 守恒） | 4 |
| `src/factorzen/pipelines/portfolio_build.py` | `run_portfolio`（拉数据/M3 → 优化 → 归因 → 落盘）+ `compute_sector_returns` + `fetch_index_weights` | 5 |
| `src/factorzen/cli/main.py`（改） | `fz portfolio build` 子命令 | 6 |
| `tests/test_portfolio_*.py` / `test_attribution_m4_*.py` | 各 task 测试 | 各 |

---

## Task 1: 约束构造（constraints.py）

**Files:**
- Create: `src/factorzen/portfolio/__init__.py`（空）, `src/factorzen/portfolio/constraints.py`
- Test: `tests/test_portfolio_constraints.py`

**Interfaces:**
- Consumes: `ExposureMatrix`（`risk/exposures.py`，`.matrix (n,k)`/`.factor_names`）；cvxpy
- Produces: `ConstraintConfig`（dataclass）；`build_constraints(w, *, exposures, config) -> list`（cvxpy 约束列表）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_portfolio_constraints.py
import cvxpy as cp
import numpy as np

from factorzen.portfolio.constraints import ConstraintConfig, build_constraints
from factorzen.risk.exposures import ExposureMatrix


def _exposures(n=6, k=3):
    # factor_names: 1 风格(size) + 2 行业(ind_A/ind_B)
    names = ["size", "ind_A", "ind_B"]
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [1, 1, 1, 0, 0, 0]   # 前3只 A 行业
    mat[:, 2] = [0, 0, 0, 1, 1, 1]   # 后3只 B 行业
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
```

- [ ] **Step 2: 跑测试确认失败** → `pixi run pytest tests/test_portfolio_constraints.py -v` → FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 constraints.py**

```python
# src/factorzen/portfolio/constraints.py
"""组合优化约束构造（cvxpy）：box / budget / 行业风格中性 / 换手。"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class ConstraintConfig:
    w_max: float = 0.05
    long_only: bool = True
    neutral_factors: list[str] | None = None      # 要中性的 exposure 列名（风格/行业）
    benchmark_weights: np.ndarray | None = None    # 中性目标基准权重（None → 中性到 0）
    turnover_budget: float | None = None
    prev_weights: np.ndarray | None = None


def build_constraints(w, *, exposures, config: ConstraintConfig) -> list:
    """返回 cvxpy 约束列表。w 为 cp.Variable(n_stocks)。"""
    cons = [cp.sum(w) == 1.0]                       # budget 全额
    if config.long_only:
        cons.append(w >= 0.0)
    cons.append(w <= config.w_max)                  # box 个股上限
    # 行业/风格中性：选定列暴露 == benchmark 暴露（或 0）
    if config.neutral_factors:
        idx = [exposures.factor_names.index(n) for n in config.neutral_factors
               if n in exposures.factor_names]
        if idx:
            X_s = exposures.matrix[:, idx]          # (n, len(idx))
            target = (X_s.T @ config.benchmark_weights
                      if config.benchmark_weights is not None
                      else np.zeros(len(idx)))
            cons.append(X_s.T @ w == target)
    # 换手：L1
    if config.turnover_budget is not None and config.prev_weights is not None:
        cons.append(cp.norm1(w - config.prev_weights) <= config.turnover_budget)
    return cons
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_portfolio_constraints.py -v
pixi run ruff check src/factorzen/portfolio/ tests/test_portfolio_constraints.py
git add src/factorzen/portfolio/__init__.py src/factorzen/portfolio/constraints.py tests/test_portfolio_constraints.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(portfolio): 约束构造(box/budget/行业风格中性/换手)"
```

---

## Task 2: 优化器（optimizer.py）

**Files:**
- Create: `src/factorzen/portfolio/optimizer.py`
- Test: `tests/test_portfolio_optimizer.py`

**Interfaces:**
- Consumes: `RiskModelResult`（`risk/model.py`：`.factor_exposures.matrix`/`.factor_covariance`/`.specific_risk`）；`build_constraints`/`ConstraintConfig`（Task 1）
- Produces: `OptimizeResult`（weights/status/objective_value/solve_seconds）；`optimize_portfolio(alpha, risk_result, *, risk_aversion=1.0, constraint_config, solver="CLARABEL") -> OptimizeResult`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_portfolio_optimizer.py
import numpy as np

from factorzen.portfolio.constraints import ConstraintConfig
from factorzen.portfolio.optimizer import OptimizeResult, optimize_portfolio
from factorzen.risk.exposures import ExposureMatrix


class _RiskResult:
    """手搓最小 RiskModelResult（仅优化器需要的 3 字段）。"""
    def __init__(self, n=6, k=3):
        rng = np.random.default_rng(1)
        names = ["size", "ind_A", "ind_B"]
        mat = rng.standard_normal((n, k))
        mat[:, 1] = [1, 1, 1, 0, 0, 0]; mat[:, 2] = [0, 0, 0, 1, 1, 1]
        self.factor_exposures = ExposureMatrix([f"{i}" for i in range(n)], names, mat)
        F = rng.standard_normal((k, k)); self.factor_covariance = F @ F.T * 0.01
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
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 optimizer.py**

```python
# src/factorzen/portfolio/optimizer.py
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
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_portfolio_optimizer.py -v
pixi run ruff check src/factorzen/portfolio/optimizer.py tests/test_portfolio_optimizer.py
git add src/factorzen/portfolio/optimizer.py tests/test_portfolio_optimizer.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(portfolio): 因子形式 mean-variance QP(显式 status, infeasible 不返垃圾)"
```

---

## Task 3: 风险因子归因（risk_attribution.py）

**Files:**
- Create: `src/factorzen/attribution/__init__.py`（空）, `src/factorzen/attribution/risk_attribution.py`
- Test: `tests/test_attribution_risk.py`

**Interfaces:**
- Consumes: `RiskModel.decompose_risk(weights, result)`（`risk/model.py`，dict）；`RiskModelResult`
- Produces: `RiskAttributionResult`（factor_return_contrib/factor_risk_contrib/specific_return/specific_risk）；`risk_factor_attribution(weights, risk_result, factor_returns_latest) -> RiskAttributionResult`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_attribution_risk.py
import math
import numpy as np

from factorzen.attribution.risk_attribution import RiskAttributionResult, risk_factor_attribution
from factorzen.risk.exposures import ExposureMatrix


class _RiskResult:
    def __init__(self):
        names = ["size", "value"]
        X = np.array([[1.0, 0.5], [0.8, -0.3], [-0.2, 1.1]])
        self.factor_exposures = ExposureMatrix(["A", "B", "C"], names, X)
        self.factor_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        self.specific_risk = np.array([0.10, 0.15, 0.20])
        self.factor_names = names


def test_return_attribution_conserves():
    """因子收益贡献 + 特异 ≈ 组合收益(因子模型口径)。"""
    r = _RiskResult()
    w = np.array([0.5, 0.3, 0.2])
    factor_ret = {"size": 0.02, "value": -0.01}   # 最新一期因子收益
    stock_ret = np.array([0.03, 0.01, -0.02])      # 个股实际收益
    res = risk_factor_attribution(w, r, factor_ret, stock_returns=stock_ret)
    assert isinstance(res, RiskAttributionResult)
    # 组合收益 = Σ 因子贡献 + 特异
    port_ret = float(w @ stock_ret)
    total_attrib = sum(res.factor_return_contrib.values()) + res.specific_return
    assert math.isclose(total_attrib, port_ret, rel_tol=1e-9)
    # 因子收益贡献 = 组合暴露 × 因子收益
    Xw = r.factor_exposures.matrix.T @ w
    assert math.isclose(res.factor_return_contrib["size"], Xw[0] * 0.02, rel_tol=1e-9)


def test_risk_contrib_matches_m3_decompose():
    """风险贡献与 M3 decompose_risk 一致(跨函数验证,非恒真)。"""
    from factorzen.risk.model import RiskModel
    from factorzen.risk.model import RiskModelResult
    r = _RiskResult()
    rr = RiskModelResult(factor_exposures=r.factor_exposures, factor_covariance=r.factor_covariance,
                         specific_risk=r.specific_risk, factor_names=r.factor_names)
    w = np.array([0.5, 0.3, 0.2])
    res = risk_factor_attribution(w, rr, {"size": 0.0, "value": 0.0},
                                  stock_returns=np.zeros(3))
    m3 = RiskModel().decompose_risk(w, rr)
    assert math.isclose(res.factor_risk_contrib["size"], m3["size"], rel_tol=1e-9)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 risk_attribution.py**

```python
# src/factorzen/attribution/risk_attribution.py
"""风险因子归因：基于 M3，把组合收益/风险分解到风格因子 + 特异。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorzen.risk.model import RiskModel


@dataclass
class RiskAttributionResult:
    factor_return_contrib: dict[str, float]   # 各因子收益贡献 = (Xᵀw)_j × f_j
    factor_risk_contrib: dict[str, float]     # 各因子风险贡献(M3 MCR)
    specific_return: float
    specific_risk: float


def risk_factor_attribution(weights, risk_result, factor_returns_latest: dict,
                            *, stock_returns) -> RiskAttributionResult:
    """收益归因：因子贡献 = 暴露×因子收益；特异 = 组合实际收益 − Σ因子贡献。
    风险归因：复用 M3 decompose_risk。"""
    w = np.asarray(weights)
    X = risk_result.factor_exposures.matrix       # (n, k)
    names = risk_result.factor_names
    Xw = X.T @ w                                  # (k,) 组合因子暴露
    factor_ret_contrib = {names[j]: float(Xw[j] * factor_returns_latest.get(names[j], 0.0))
                          for j in range(len(names))}
    port_ret = float(w @ np.asarray(stock_returns))
    specific_return = port_ret - sum(factor_ret_contrib.values())
    # 风险贡献复用 M3
    decomp = RiskModel().decompose_risk(w, risk_result)
    factor_risk_contrib = {n: float(decomp.get(n, 0.0)) for n in names}
    return RiskAttributionResult(
        factor_return_contrib=factor_ret_contrib, factor_risk_contrib=factor_risk_contrib,
        specific_return=specific_return, specific_risk=float(decomp.get("specific_risk", 0.0)))
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_attribution_risk.py -v
pixi run ruff check src/factorzen/attribution/ tests/test_attribution_risk.py
git add src/factorzen/attribution/__init__.py src/factorzen/attribution/risk_attribution.py tests/test_attribution_risk.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(attribution): 风险因子归因(因子收益贡献守恒 + M3 风险分解)"
```

---

## Task 4: Brinson 归因（brinson.py）

**Files:**
- Create: `src/factorzen/attribution/brinson.py`
- Test: `tests/test_attribution_brinson.py`

**Interfaces:**
- Produces: `BrinsonResult`（allocation/selection/total_excess）；`brinson_attribution(port_weights, bench_weights, sector_returns, sectors) -> BrinsonResult`（股票级输入）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_attribution_brinson.py
import math
import numpy as np

from factorzen.attribution.brinson import BrinsonResult, brinson_attribution


def test_brinson_conserves_to_excess():
    """配置 + 选股 ≈ 组合超额收益。"""
    # 4 只股票，2 行业(A/B)，单期
    port_w = np.array([0.4, 0.1, 0.4, 0.1])
    bench_w = np.array([0.25, 0.25, 0.25, 0.25])
    sectors = ["A", "A", "B", "B"]
    stock_ret = np.array([0.05, 0.03, -0.01, 0.02])
    res = brinson_attribution(port_w, bench_w, stock_ret, sectors)
    assert isinstance(res, BrinsonResult)
    port_ret = float(port_w @ stock_ret)
    bench_ret = float(bench_w @ stock_ret)
    excess = port_ret - bench_ret
    total = sum(res.allocation.values()) + sum(res.selection.values())
    assert math.isclose(total, excess, rel_tol=1e-9, abs_tol=1e-12)


def test_pure_allocation():
    """组合行业内选股与基准一致、仅行业权重不同 → 全是配置效应。"""
    port_w = np.array([0.3, 0.3, 0.2, 0.2])    # A 行业超配
    bench_w = np.array([0.25, 0.25, 0.25, 0.25])
    sectors = ["A", "A", "B", "B"]
    stock_ret = np.array([0.04, 0.04, 0.01, 0.01])  # 行业内同收益(无选股差异)
    res = brinson_attribution(port_w, bench_w, stock_ret, sectors)
    assert abs(sum(res.selection.values())) < 1e-9   # 选股效应≈0
    assert abs(sum(res.allocation.values())) > 1e-6   # 配置效应非0
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 brinson.py**

```python
# src/factorzen/attribution/brinson.py
"""Brinson 归因（M4 版，股票级输入）：单期 BHB 配置效应 + 选股效应(交互归入选股)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrinsonResult:
    allocation: dict[str, float]   # 各行业配置效应
    selection: dict[str, float]    # 各行业选股效应(含交互)
    total_excess: float


def brinson_attribution(port_weights, bench_weights, stock_returns, sectors) -> BrinsonResult:
    """单期 Brinson。各行业:
    配置 = (w_p − w_b)·(r_b_sector − r_b_total)
    选股 = w_p·(r_p_sector − r_b_sector)   ← 交互项归入选股
    守恒: Σ(配置+选股) = port_ret − bench_ret。"""
    w_p = np.asarray(port_weights); w_b = np.asarray(bench_weights)
    r = np.asarray(stock_returns); secs = list(sectors)
    uniq = sorted(set(secs))
    r_b_total = float(w_b @ r)
    allocation, selection = {}, {}
    for s in uniq:
        m = np.array([x == s for x in secs])
        wp_s = float(w_p[m].sum()); wb_s = float(w_b[m].sum())
        r_p_s = float(w_p[m] @ r[m] / wp_s) if wp_s > 1e-12 else 0.0   # 组合行业内收益
        r_b_s = float(w_b[m] @ r[m] / wb_s) if wb_s > 1e-12 else 0.0   # 基准行业内收益
        allocation[s] = (wp_s - wb_s) * (r_b_s - r_b_total)
        selection[s] = wp_s * (r_p_s - r_b_s)
    total_excess = float(w_p @ r) - r_b_total
    return BrinsonResult(allocation=allocation, selection=selection, total_excess=total_excess)
```
> 守恒推导：Σ[配置+选股] = Σ[(wp−wb)(rb_s−rb) + wp(rp_s−rb_s)] = Σ wp·rp_s − Σ wb·rb_s − rb·Σ(wp−wb)。Σwp=Σwb=1 故末项=0；Σwp·rp_s = port_ret，Σwb·rb_s = bench_ret。✓ 测试以此为守恒断言。

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_attribution_brinson.py -v
pixi run ruff check src/factorzen/attribution/brinson.py tests/test_attribution_brinson.py
git add src/factorzen/attribution/brinson.py tests/test_attribution_brinson.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(attribution): Brinson 配置/选股归因(股票级,守恒)"
```

---

## Task 5: Pipeline（portfolio_build.py）

**Files:**
- Create: `src/factorzen/pipelines/portfolio_build.py`
- Test: `tests/test_portfolio_pipeline.py`

**Interfaces:**
- Consumes: `optimize_portfolio`/`ConstraintConfig`（Task 1/2）；`risk_factor_attribution`（Task 3）；`brinson_attribution`（Task 4）
- Produces: `run_portfolio(alpha, risk_result, *, codes, stock_returns, sectors, bench_weights=None, prev_weights=None, risk_aversion=1.0, neutral_factors=None, turnover_budget=None, w_max=0.05, out_dir, run_id=None) -> dict`（`{run_dir, status, n_holdings, objective}`）；落 `weights.parquet`/`attribution.csv`/`risk_summary.csv`/`manifest.json`

- [ ] **Step 1: 写失败测试（mock，注入 risk_result + α）**

```python
# tests/test_portfolio_pipeline.py
import json
from pathlib import Path
import numpy as np

from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult


def _risk_result(n=6, k=3):
    rng = np.random.default_rng(2)
    names = ["size", "ind_A", "ind_B"]
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [1, 1, 1, 0, 0, 0]; mat[:, 2] = [0, 0, 0, 1, 1, 1]
    F = rng.standard_normal((k, k)); F = F @ F.T * 0.01
    return RiskModelResult(
        factor_exposures=ExposureMatrix([f"{i:06d}.SZ" for i in range(n)], names, mat),
        factor_covariance=F, specific_risk=np.full(n, 0.1), factor_names=names)


def test_run_portfolio_writes_products(tmp_path: Path):
    rr = _risk_result()
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
```

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 portfolio_build.py**（拉/接 risk_result → optimize → 归因 → 落盘；含 `compute_sector_returns` 辅助。`fetch_index_weights` 真实数据用，测试注入故 MVP 可后置）

```python
# src/factorzen/pipelines/portfolio_build.py
"""组合构建 pipeline：α + M3 风险模型 → 优化 → 归因 → 落盘。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.attribution.brinson import brinson_attribution
from factorzen.attribution.risk_attribution import risk_factor_attribution
from factorzen.portfolio.constraints import ConstraintConfig
from factorzen.portfolio.optimizer import optimize_portfolio


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_portfolio(alpha, risk_result, *, codes, stock_returns, sectors,
                  factor_returns_latest, bench_weights=None, prev_weights=None,
                  risk_aversion=1.0, neutral_factors=None, turnover_budget=None,
                  w_max=0.05, out_dir="workspace/portfolios", run_id=None) -> dict:
    t0 = time.perf_counter()
    cfg = ConstraintConfig(w_max=w_max, neutral_factors=neutral_factors,
                           benchmark_weights=bench_weights,
                           turnover_budget=turnover_budget, prev_weights=prev_weights)
    opt = optimize_portfolio(alpha, risk_result, risk_aversion=risk_aversion,
                             constraint_config=cfg)
    rid = run_id or "portfolio"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    w = opt.weights if opt.weights is not None else np.zeros(len(codes))

    pl.DataFrame({"ts_code": codes, "target_weight": w.tolist(),
                  "prev_weight": (prev_weights.tolist() if prev_weights is not None
                                  else [0.0] * len(codes))}).write_parquet(run_dir / "weights.parquet")

    # 归因（仅 optimal 时有意义）
    attrib_rows = []
    if opt.weights is not None:
        ra = risk_factor_attribution(w, risk_result, factor_returns_latest,
                                     stock_returns=np.asarray(stock_returns))
        for k, v in ra.factor_return_contrib.items():
            attrib_rows.append({"type": "factor_return", "key": k, "value": v})
        attrib_rows.append({"type": "specific_return", "key": "specific", "value": ra.specific_return})
        bench = bench_weights if bench_weights is not None else np.full(len(codes), 1.0 / len(codes))
        br = brinson_attribution(w, bench, np.asarray(stock_returns), sectors)
        for s, v in br.allocation.items():
            attrib_rows.append({"type": "brinson_allocation", "key": s, "value": v})
        for s, v in br.selection.items():
            attrib_rows.append({"type": "brinson_selection", "key": s, "value": v})
    pl.DataFrame(attrib_rows if attrib_rows else {"type": [], "key": [], "value": []}) \
        .write_csv(run_dir / "attribution.csv")

    # 风险摘要（复用 M3 decompose）
    from factorzen.risk.model import RiskModel
    risk_rows = []
    if opt.weights is not None:
        decomp = RiskModel().decompose_risk(w, risk_result)
        risk_rows = [{"metric": k, "value": float(v)} for k, v in decomp.items()]
    pl.DataFrame(risk_rows if risk_rows else {"metric": [], "value": []}) \
        .write_csv(run_dir / "risk_summary.csv")

    manifest = {"run_id": rid, "status": opt.status, "objective": opt.objective_value,
                "n_holdings": int((w > 1e-6).sum()), "risk_aversion": risk_aversion,
                "w_max": w_max, "neutral_factors": neutral_factors,
                "turnover_budget": turnover_budget,
                "turnover": (float(np.abs(w - prev_weights).sum()) if prev_weights is not None else None),
                "git_sha": _git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"run_dir": str(run_dir), "status": opt.status,
            "n_holdings": manifest["n_holdings"], "objective": opt.objective_value}


def compute_sector_returns(daily: pl.DataFrame, stocks: pl.DataFrame) -> pl.DataFrame:
    """行业等权收益：daily(pct_chg) + stocks(industry) → [trade_date, sector, ret]。"""
    j = daily.join(stocks.select(["ts_code", "industry"]), on="ts_code")
    return (j.group_by(["trade_date", "industry"])
            .agg((pl.col("pct_chg") / 100.0).mean().alias("ret"))
            .rename({"industry": "sector"}))
```

- [ ] **Step 4: 跑测试通过 + ruff + 提交**

```bash
pixi run pytest tests/test_portfolio_pipeline.py -v
pixi run ruff check src/factorzen/pipelines/portfolio_build.py tests/test_portfolio_pipeline.py
git add src/factorzen/pipelines/portfolio_build.py tests/test_portfolio_pipeline.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(portfolio): run_portfolio pipeline(优化+归因+落盘) + 行业收益聚合"
```

---

## Task 6: CLI `fz portfolio build` + 收尾

**Files:**
- Modify: `src/factorzen/cli/main.py`（顶层 `portfolio` 命令组 + `_cmd_portfolio_build`）
- Test: `tests/test_portfolio_cli.py`
- Modify: `README.md`（核心能力补「组合」行）

**Interfaces:**
- Consumes: `run_portfolio`（Task 5）；`get_universe`/`loader`/`RiskModel.build`（拉数据+风险模型）；`build_parser` 仿 `fz risk`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_portfolio_cli.py
def test_parser_has_portfolio_build():
    from factorzen.cli.main import build_parser
    p = build_parser()
    args = p.parse_args(["portfolio", "build", "--start", "20230101", "--end", "20241231",
                         "--universe", "csi500", "--alpha-file", "a.parquet", "--lam", "2.0"])
    assert args.command == "portfolio"
    assert args.portfolio_command == "build"
    assert args.alpha_file == "a.parquet"
    assert args.lam == 2.0
    assert callable(args.func)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL（`AttributeError`）

- [ ] **Step 3: 接入 CLI**

`build_parser()` 加顶层 `portfolio` 组（仿 `fz risk`）：
```python
    portfolio = sub.add_parser("portfolio", help="Portfolio construction & attribution")
    pf_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    p_build = pf_sub.add_parser("build", help="Build optimized portfolio + attribution")
    p_build.add_argument("--start", required=True)
    p_build.add_argument("--end", required=True)
    p_build.add_argument("--universe", default="all_a")
    p_build.add_argument("--alpha-file", required=True, dest="alpha_file",
                         help="α 信号文件(parquet/csv: 列 ts_code + alpha)")
    p_build.add_argument("--lam", type=float, default=1.0, dest="lam", help="风险厌恶系数")
    p_build.add_argument("--w-max", type=float, default=0.05, dest="w_max")
    p_build.add_argument("--turnover", type=float, default=None)
    p_build.add_argument("--industry-neutral", action="store_true", dest="industry_neutral")
    p_build.set_defaults(func=_cmd_portfolio_build)
```
模块顶层加 handler（延迟 import，仿 `_cmd_risk_build`）：
```python
def _cmd_portfolio_build(args: argparse.Namespace) -> int:
    import numpy as np
    import polars as pl
    from factorzen.core import loader
    from factorzen.core.universe import get_universe
    from factorzen.pipelines.portfolio_build import run_portfolio
    from factorzen.risk.model import RiskModel
    stocks = get_universe(args.end, args.universe)
    uni = stocks["ts_code"].to_list()
    daily = loader.fetch_daily(args.start, args.end).filter(pl.col("ts_code").is_in(uni))
    daily_basic = loader.fetch_daily_basic(args.start, args.end).filter(pl.col("ts_code").is_in(uni))
    risk_result = RiskModel().build(daily, daily_basic, stocks, args.start, args.end)
    codes = risk_result.factor_exposures.codes
    # α：从 --alpha-file 读取截面信号(ts_code + alpha)，对齐 codes 顺序(缺失填 0)
    adf = (pl.read_parquet(args.alpha_file) if args.alpha_file.endswith(".parquet")
           else pl.read_csv(args.alpha_file))
    amap = dict(zip(adf["ts_code"].to_list(), adf["alpha"].to_list()))
    alpha = np.array([float(amap.get(c, 0.0)) for c in codes])
    neutral = [n for n in risk_result.factor_names if n.startswith("ind_")] if args.industry_neutral else None
    res = run_portfolio(alpha, risk_result, codes=codes,
                        stock_returns=np.zeros(len(codes)), sectors=list(stocks["industry"]),
                        factor_returns_latest={}, risk_aversion=args.lam, w_max=args.w_max,
                        neutral_factors=neutral, turnover_budget=args.turnover)
    print(f"[portfolio] status={res['status']} holdings={res['n_holdings']} → {res['run_dir']}")
    return 0
```
> `--alpha-file` 是 α 信号的通用注入口（`ts_code + alpha` 两列），与上游解耦——单因子值、合成信号、挖掘因子都可导出成此格式喂入。`stock_returns` 真实数据下应传持仓期实际收益（驱动归因），MVP/smoke 可传 0（仅看权重+风险分解）。

- [ ] **Step 4: 跑测试通过**

Run: `pixi run pytest tests/test_portfolio_cli.py -v` → PASS

- [ ] **Step 5: 全量质量门**

```bash
pixi run pytest tests/test_portfolio_*.py tests/test_attribution_*.py -q   # M4 全测试绿
pixi run ruff check src/factorzen/portfolio/ src/factorzen/attribution/ src/factorzen/pipelines/portfolio_build.py
pixi run pytest tests/test_optimizer.py tests/test_attribution.py -q       # 现有回归（M4 没破坏 daily/optimization）
```

- [ ] **Step 6: README + 提交**

README「核心能力」表加："| 组合 | 带约束凸优化(cvxpy mean-variance + 行业/风格中性 + 换手)、Brinson + 风险因子归因，`fz portfolio build` |"。
```bash
git add src/factorzen/cli/main.py tests/test_portfolio_cli.py README.md
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(portfolio): fz portfolio build CLI + README"
```

---

## 收尾验收（全部 task 完成后）

- [ ] `pixi run pytest tests/test_portfolio_*.py tests/test_attribution_risk.py tests/test_attribution_brinson.py -q` 全绿
- [ ] `pixi run pytest tests/test_optimizer.py tests/test_attribution.py -q` 现有回归绿（M4 没改 `daily/optimization/`）
- [ ] `pixi run ruff check src/factorzen/portfolio/ src/factorzen/attribution/ src/factorzen/pipelines/portfolio_build.py tests/test_portfolio_*.py` 0 errors
- [ ] **求解稳定（验收核心）**：infeasible → `weights=None` + 显式 status（Task 2 断言，非恒真）
- [ ] **约束满足**：box/budget/中性（暴露≈0/target）/换手（L1≤budget）误差 < 容差（Task 1 断言）
- [ ] **归因守恒**：Brinson `Σ(配置+选股)≈超额`；风险因子贡献与 M3 decompose 一致（Task 3/4 断言）
- [ ] 真实数据 smoke（手动）：`fz portfolio build --start 20230101 --end 20241231 --universe csi500 --alpha <因子> --industry-neutral` → 产 weights/attribution/manifest，status=optimal
- [ ] `git status --short` 干净（只 M4 相关入库，未带 M0）
- [ ] 更新本 plan 追加完成记录 + memory roadmap（M4 完成）

---

*M4 完成后，FactorZen 拥有"α → 带约束凸优化组合 → Brinson + 风险因子归因"的组合构建链路，与 M3 风险模型天然衔接、与现有单因子研究流（daily/optimization）分离，求解稳定、收益来源可解释。*
