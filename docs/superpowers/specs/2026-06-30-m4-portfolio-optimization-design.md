# M4 · 组合优化与归因 — 设计文档

> 状态：设计讨论完成（2026-06-30），待用户复核 → 转实现计划。
> 上游：[FactorZen 升级计划](../../FactorZen-升级计划.md) 的里程碑 **M4**，依赖 M3（风险模型）。
> 定位：把"验证过的 α 信号"通过**带约束凸优化**变成可执行的目标组合，并用 **Brinson + 风险因子归因**解释组合收益来源。

---

## 1. 目标与定位

`α 信号 → 目标组合权重（cvxpy 约束优化） → 可解释收益来源（归因）`。这是研究流水线里 **Tear Sheet 之后的下一层**：

- **Tear Sheet（reports/，现有）**：单因子研究——"这个因子有没有预测力"（IC / 分层回测 / walk-forward）。无持仓、无归因。
- **M4（本里程碑）**：组合构建 + 归因——"拿信号该建什么组合、组合赚的钱从哪来"。产**目标权重** + **归因**（组合收益 → 因子 vs 行业 vs 特异）。

二者不重叠、不替代，是前后两个阶段。M4 是 FactorZen **第一个该用现成凸优化库**的里程碑（`cvxpy>=1.4` 已是项目依赖，非新增）。

### 1.1 已拍板决策（讨论结论）

| 决策 | 选择 |
|---|---|
| 优化目标 | **均值-方差** `max αᵀw − λ·wᵀΣw`（Σ 用 M3 风险模型） |
| 约束 | box(个股上限) + budget(全额) + **行业/风格中性**(M3 exposures) + **换手**(上期权重)；跟踪误差 defer |
| 归因 | **Brinson**(配置/选股) + **风险因子归因**(M3 decompose) |
| 报告形态 | **轻量 CSV/manifest**（像 M3 risk_summary）；HTML 美化留 M7 |
| 求解库 | **cvxpy**（已依赖，QP 凸优化） |
| α 输入 | 通用截面向量（来源不限：单因子/合成/挖掘），与上游解耦 |
| 风险项形式 | M3 因子模型形式 `(Xᵀw)ᵀF(Xᵀw) + Σ(D·w)²`（与 `predict_risk` 一致，避免 n×n 大矩阵） |

---

## 2. 现状（基线）

M4 站在已入库的 M3 之上 + cvxpy 已依赖：

| 复用对象 | 提供什么 | 位置 |
|---|---|---|
| 风险模型 | `RiskModelResult`（`factor_exposures: ExposureMatrix`，`factor_covariance: (k,k)`，`specific_risk: (n,)`，`factor_returns`，`factor_names`） | `risk/model.py` |
| 风险预测 | `RiskModel.predict_risk(weights, result)` / `decompose_risk(weights, result)` | `risk/model.py` |
| 暴露矩阵 | `ExposureMatrix(codes, factor_names, matrix(n,k))` | `risk/exposures.py` |
| 凸优化 | `cvxpy>=1.4` | pixi/pyproject（已依赖） |
| 数值 | numpy / scipy / polars | 已依赖 |
| CLI | `build_parser()`，仿 `fz risk`/`fz mine` | `cli/main.py` |

**缺口**：无组合优化器、无约束库、无归因模块，全新。

---

## 3. 架构与模块边界

| 层 | 模块 | 新建/复用 | 职责 |
|---|---|---|---|
| 优化 | `portfolio/optimizer.py` | 🆕 cvxpy | mean-variance QP 求解 → 目标权重 + 求解状态 |
| 约束 | `portfolio/constraints.py` | 🆕 | 构造 cvxpy 约束：box / budget / 行业风格中性 / 换手 |
| 归因 | `attribution/brinson.py` | 🆕 | Brinson：配置效应 + 选股效应（按行业） |
| 归因 | `attribution/risk_attribution.py` | 🆕 复用 M3 | 组合收益/风险 → 风格因子 + 行业 + 特异 |
| 入口 | `pipelines/portfolio_build.py` | 🆕 | `run_portfolio`：α + M3 → 优化 → 权重 + 归因 + 报告 |
| 风险 | `RiskModel` / `RiskModelResult` / `ExposureMatrix` | ♻️ M3 | 风险项 + 归因基础 |
| CLI | `cli/main.py` `fz portfolio build` | 🆕（扩展） | |

`portfolio/` 与 `attribution/` 是两个相对独立子系统，但都服务"因子→组合→归因"链路，共用 `RiskModelResult` 与 universe 数据。`research/combination/`（因子**合成**，信号层）是 M4 的可选 α 上游，不在本里程碑改动。

---

## 4. 优化器（portfolio/optimizer.py）

**目标（QP，凸）**：
```
maximize   αᵀw − λ · risk(w)
其中       risk(w) = (Xᵀw)ᵀ F (Xᵀw) + Σᵢ (Dᵢ wᵢ)²   ← M3 因子风险模型形式
```
- `α: (n,)` 截面信号；`X: (n,k)` 暴露（M3 ExposureMatrix.matrix）；`F: (k,k)` 因子协方差；`D: (n,)` 特质风险（std）；`λ` 风险厌恶系数。
- cvxpy 表达：`risk = cp.quad_form(X.T @ w, F) + cp.sum_squares(cp.multiply(D, w))`，`objective = cp.Maximize(alpha @ w - lam * risk)`。

**接口**：
```python
@dataclass
class OptimizeResult:
    weights: np.ndarray          # (n,) 目标权重（infeasible 时为 None）
    status: str                  # cvxpy status: "optimal"/"infeasible"/"unbounded"/...
    objective_value: float | None
    solve_seconds: float

def optimize_portfolio(alpha, risk_result, *, codes, risk_aversion=1.0,
                       constraints, solver="ECOS") -> OptimizeResult
```

**求解稳定性（验收核心）**：cvxpy `prob.status` 非 `optimal` 时——`OptimizeResult.weights = None` + 记录 status，**绝不返回垃圾权重**；pipeline 层可选"软约束松弛"（把硬中性/换手改成目标的惩罚项重解，见 §5），并在报告标注降级。

---

## 5. 约束（portfolio/constraints.py）

约束构造器：输入 cvxpy 变量 `w` + 参数，返回 `list[cvxpy.Constraint]`。

| 约束 | 数学 | cvxpy |
|---|---|---|
| box（个股上下限） | `0 ≤ w ≤ w_max` | `w >= 0, w <= w_max` |
| budget（全额投资） | `Σw = 1` | `cp.sum(w) == 1` |
| 行业/风格中性 | `Xₛᵀw = Xₛᵀw_bench`（选定列 s 暴露对齐 benchmark；或 = 0） | `X_s.T @ w == target` |
| 换手 | `‖w − w_prev‖₁ ≤ turnover_budget` | `cp.norm1(w - w_prev) <= budget` |

```python
@dataclass
class ConstraintConfig:
    w_max: float = 0.05
    long_only: bool = True
    neutral_factors: list[str] | None = None   # 要中性的 exposure 列名（风格/行业）
    benchmark_weights: np.ndarray | None = None # 中性目标（None → 中性到 0）
    turnover_budget: float | None = None
    prev_weights: np.ndarray | None = None

def build_constraints(w, *, exposures: ExposureMatrix, config: ConstraintConfig) -> list
```

**软约束松弛（可选，求解稳定性兜底）**：硬中性/换手导致 `infeasible` 时，pipeline 把它们移出硬约束、加进目标惩罚（`− γ·‖Xₛᵀw − target‖²`），保证有解并在报告标注。MVP 先硬约束 + infeasible 显式报告，软松弛作为开关。

---

## 6. 归因

> 两种归因是**互补视角，口径不同、不必相等**：Brinson 用**实际收益**事后分解（行业配置 vs 选股），风险因子归因用**因子模型口径**（暴露 × 因子收益）。前者回答"行业配置/选股谁贡献"，后者回答"哪些风格/行业因子驱动"。报告并列呈现，不强行对账。

### 6.1 Brinson（attribution/brinson.py）
组合相对 benchmark 的超额收益，按**行业**分解为配置效应 + 选股效应：
- 配置效应ᵢ = (w_pᵢ − w_bᵢ) · (r_bᵢ − r_b)
- 选股效应ᵢ = w_bᵢ · (r_pᵢ − r_bᵢ)（或交互项归入选股）
- 守恒：Σ(配置 + 选股) = 组合超额收益。
```python
@dataclass
class BrinsonResult:
    allocation: dict[str, float]   # 各行业配置效应
    selection: dict[str, float]    # 各行业选股效应
    total_excess: float
def brinson_attribution(port_weights, bench_weights, sector_returns, sectors) -> BrinsonResult
```

### 6.2 风险因子归因（attribution/risk_attribution.py）
基于 M3，把组合**风险与收益**分解到风格因子 + 行业 + 特异：
- 风险：复用/扩展 `RiskModel.decompose_risk(weights, result)` → 每因子风险贡献（MCR）+ 特异。
- 收益：`组合在因子 j 的暴露 (Xᵀw)ⱼ × 因子收益 f_j` = 因子 j 的收益贡献；特异收益 = 残差。
- 守恒：Σ 因子收益贡献 + 特异 ≈ 组合收益（在因子模型口径下）。
```python
@dataclass
class RiskAttributionResult:
    factor_return_contrib: dict[str, float]   # 各因子收益贡献
    factor_risk_contrib: dict[str, float]     # 各因子风险贡献(MCR)
    specific_return: float
    specific_risk: float
def risk_factor_attribution(weights, risk_result, factor_returns) -> RiskAttributionResult
```

---

## 7. 报告（CSV/manifest）

`run_portfolio` 落 `workspace/portfolios/{run_id}/`：
```text
weights.parquet          ts_code × target_weight（+ prev_weight/active_weight）
risk_summary.csv         组合年化风险 + 因子风险贡献 + 特异占比（复用 M3 口径）
attribution.csv          Brinson(配置/选股 按行业) + 风险因子归因(因子/行业/特异 收益+风险贡献)
manifest.json            参数(λ/约束/universe)/求解 status/objective/换手/git_sha/耗时
```
报告让人 30 秒看懂"组合买了什么、风险来自哪、收益归谁"。HTML 美化（复用 reports/ 引擎）留 M7。

---

## 8. 接口契约

**新建**（§4/§5/§6 已列签名）：`optimize_portfolio`/`OptimizeResult`、`build_constraints`/`ConstraintConfig`、`brinson_attribution`/`BrinsonResult`、`risk_factor_attribution`/`RiskAttributionResult`、`run_portfolio(alpha_df, risk_result, *, benchmark_weights=None, prev_weights=None, risk_aversion=1.0, constraint_config, out_dir, run_id=None) -> dict`。

**复用**（writing-plans 用 interface agent 精确化）：`RiskModel`/`RiskModelResult`/`ExposureMatrix`/`decompose_risk`、`get_universe`（行业）、`loader`、`build_parser`、cvxpy。

---

## 9. 测试策略 + 验收（DoD）

全 mock 离线（小 universe + 构造 α/F/X/D），cvxpy 确定性：
- [ ] **优化器求解**：返回 `status=="optimal"`，权重满足全部约束——`Σw≈1`、`0≤w≤w_max`、中性 `|Xₛᵀw − target|<1e-6`、换手 `‖w−w_prev‖₁ ≤ budget+1e-6`。
- [ ] **求解稳定（验收核心）**：构造矛盾约束 → `status=="infeasible"`、`weights is None`，**不返回垃圾**（非恒真）。
- [ ] **Brinson 守恒**：`Σ(配置+选股) ≈ total_excess`（rel_tol 1e-9）。
- [ ] **风险因子归因守恒**：因子收益贡献 + 特异 ≈ 组合收益；因子风险贡献与 M3 `decompose_risk` 一致（跨函数验证，非恒真）。
- [ ] **端到端**：`run_portfolio`（mock）落 weights/risk_summary/attribution/manifest，manifest 含 status/objective/换手。
- [ ] 真实数据 smoke（手动）：`fz portfolio build` 跑通 + 产物 + 归因可解释。
- [ ] cvxpy 已依赖（无新增）；ruff/test 绿；`git add` 只含 M4 不带 M0；提交 `rookiewu417`。

CI 离线：mock 求解（cvxpy 纯本地）；真实数据/universe 为手动 smoke。

---

## 10. 建议实现顺序（为 writing-plans 铺垫）

1. **`portfolio/constraints.py`**：`ConstraintConfig` + `build_constraints`（cvxpy 约束，纯构造可单测）。
2. **`portfolio/optimizer.py`**：`optimize_portfolio` + `OptimizeResult`（mean-variance QP + 求解状态 + infeasible 处理）。
3. **`attribution/risk_attribution.py`**：风险因子归因（复用 M3 decompose + 因子收益，守恒断言）。
4. **`attribution/brinson.py`**：Brinson 配置/选股（守恒断言）。
5. **`pipelines/portfolio_build.py`**：`run_portfolio`（拉数据/M3 → 优化 → 归因 → 落 weights/csv/manifest）。
6. **CLI `fz portfolio build`** + README。
7. **真实 smoke** + plan/memory 完成记录。

---

## 11. 范围外

- 跟踪误差约束（进阶，需 benchmark 权重 + Σ）→ 后续。
- 风险预算 / risk parity 优化器（mean-variance 已覆盖主线）→ 后续。
- HTML 归因报告（复用 reports/ 引擎）→ M7 展示。
- 多期/动态再平衡、交易成本模型优化 → 后续。
- 把 M4 接回 M1/M5/M6（让 Agent 直接产组合）→ 后续。

---

*M4 完成后，FactorZen 拥有"α 信号 → 带约束凸优化目标组合 → Brinson + 风险因子归因"的组合构建链路——求解稳定、收益来源可解释（因子 vs 行业 vs 特异），与 M3 风险模型天然衔接，是简历级的「带约束凸优化 + 归因」成果。*
