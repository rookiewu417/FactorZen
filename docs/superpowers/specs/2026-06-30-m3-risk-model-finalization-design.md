# M3 · 风险模型收口 — 设计文档

> 状态：设计已评审通过（2026-06-30），待转实现计划。
> 上游：[FactorZen 升级计划](../../FactorZen-升级计划.md) 的里程碑 **M3**。
> 定位：把已写好但**散落工作区未提交、缺测试、未接入主线**的 Barra 风险模型（`src/factorzen/risk/`，1198 行）收口——补核心模块测试 + `fz risk build` CLI + 轻量风险报告 + 提交入库。

---

## 1. 目标与定位

给已存在的 Barra 多因子风险模型补上**正确性保护**与**主线接入**，让它能跑、能看、可复现，并入库。**不重写现有 risk 算法**（仅在测试暴露 bug 时顺手修）。

### 1.1 已拍板决策（评审结论）

| 决策 | 选择 |
|---|---|
| 收口范围 | 补测试 + `fz risk build` CLI + 轻量 CSV/markdown 报告 |
| 测试暴露 risk 现有 bug | **顺手修**（纳入收口，既然在补测试） |
| 报告形态 | CSV/markdown 摘要（完整 HTML 图表报告留后续） |
| 产物目录 | `workspace/risk_models/{run_id}/` |
| 现有 risk 代码 | 不重写（除非测试暴露 bug） |

---

## 2. 现状（基线）

`src/factorzen/risk/` 已是完整 Barra 风险模型（`import` OK、无 TODO/占位）：
- `style_factors.py`（359）：风格因子 `STYLE_FACTOR_REGISTRY` + `cs_standardize`。**已有测试**（`test_style_factors.py`，6 passed）。
- `industry_factors.py`（80）：`get_industry_dummies` / `get_industry_names`。**无测试**。
- `exposures.py`（163）：`ExposureMatrix` + `compute_exposures`。**无测试**。
- `covariance.py`（195）：`estimate_factor_covariance`（Newey-West）/ `estimate_specific_risk`（shrinkage）/ `eigenvector_adjustment`。**无测试**。
- `model.py`（367）：`RiskModel.build/predict_risk/decompose_risk` + `RiskModelResult`。**无测试**。

`build` 流程：拉 `daily_data`(pct_chg) + `daily_basic`(估值) + `stocks`(行业)，逐日 `compute_exposures` → 截面 OLS（`sm.OLS`，含行业哑变量）估因子收益 + 残差 + R² → `estimate_factor_covariance` + `estimate_specific_risk`。

**缺口**：4 个核心模块无测试；完全未接入主线（无 CLI/报告/pipeline）。

---

## 3. 补测试（4 个无测试模块，构造验证）

全部纯 mock（仿 `test_style_factors.py`：`np.random.default_rng` + 构造 polars DataFrame），无磁盘/网络。

| 测试文件 | 覆盖 | 关键断言（已知输入 → 性质） |
|---|---|---|
| `tests/test_risk_industry.py` | `get_industry_dummies` / `get_industry_names` | 哑变量每行和=1（每股属一个行业）；列数=行业数；行业名列表正确 |
| `tests/test_risk_exposures.py` | `compute_exposures` / `ExposureMatrix` | 暴露矩阵 shape = (n_stocks, n_factors)；风格因子截面标准化（均值≈0）；含行业哑变量列；`n_stocks`/`n_factors` property 正确 |
| `tests/test_risk_covariance.py` | `estimate_factor_covariance` / `estimate_specific_risk` / `eigenvector_adjustment` | 协方差**对称**且**半正定**（特征值≥0）；Newey-West lags 生效；特质风险**全正**；shrinkage 把极端值拉向均值；eigenvector_adjustment 返回同形状对称阵 |
| `tests/test_risk_model.py` | `RiskModel.build` / `predict_risk` / `decompose_risk` | 端到端 mock → `RiskModelResult` 各字段非空且 shape 一致；**平均 R² ∈ [0,1]**；`predict_risk(weights) > 0`；**`decompose_risk` 各分量之和 ≈ 总方差**（因子风险 + 特质风险 = 总风险） |

**测试暴露 bug 的处理**：若构造验证发现 risk 现有代码的真实 bug（如协方差非半正定、风险分解不守恒），在对应 task 内顺手修复并加回归断言。

---

## 4. CLI `fz risk build`（顶层命令组，仿 `fz mine`）

```bash
fz risk build --universe csi500 --start 20230101 --end 20241231 \
   [--cov-half-life 90 --nw-lags 2 --spec-half-life 90 --spec-shrinkage 0.3]
```

`build_parser()` 增顶层 `risk` subparser（与 `factor`/`mine`/`validate` 并列）。`_cmd_risk_build(args) -> int`（直调式，仿 `_cmd_factor_sweep`）：
1. 拉数据：`FactorDataContext` 提供 `daily`/`daily_basic`；`fetch_stock_basic`（含 industry）提供 `stocks`；`get_universe` 解析 universe。
2. `RiskModel(cov_half_life, nw_lags, spec_half_life, spec_shrinkage).build(...)`。
3. 落产物到 `workspace/risk_models/{run_id}/`：

```text
workspace/risk_models/{run_id}/
├── exposures.parquet         最新暴露矩阵（ts_code × factor）
├── factor_covariance.parquet 因子协方差（factor × factor）
├── specific_risk.parquet     特质风险（ts_code, specific_risk）
├── factor_returns.parquet    因子收益时间序列
├── risk_summary.csv          轻量报告（见 §5）
└── manifest.json             参数 / universe / start-end / git SHA / 平均 R² / 耗时
```

CI 离线：`fz risk build` 端到端 smoke 为**手动命令**（需 Tushare/本地缓存），不进默认 CI；单元测试全 mock。

---

## 5. 轻量风险报告（CSV/markdown 摘要）

`risk_summary.csv` + 终端打印，让人 30 秒看懂「风险来自哪」：
- 因子数、平均回归 R²、有效交易日数。
- 每个因子的波动（因子协方差对角线 √）。
- 特质风险分布（均值 / 中位 / 分位）。
- 风格因子暴露统计（最新一期各风格因子的截面均值/标准差）。
- **一个等权组合的风险分解示例**：`predict_risk` 总风险 + `decompose_risk` 的因子风险 vs 特质风险占比。

---

## 6. 接口契约（复用现有，不重造）

| 用途 | 复用接口 | 位置 |
|---|---|---|
| 风险模型 | `RiskModel(...).build/predict_risk/decompose_risk` → `RiskModelResult` | `risk/model.py` |
| 暴露 | `compute_exposures(daily_data, daily_basic, stocks, trade_date)` → `ExposureMatrix` | `risk/exposures.py` |
| 协方差/特质风险 | `estimate_factor_covariance` / `estimate_specific_risk` / `eigenvector_adjustment` | `risk/covariance.py` |
| 行业 | `get_industry_dummies` / `get_industry_names` | `risk/industry_factors.py` |
| 风格因子 | `STYLE_FACTOR_REGISTRY` / `cs_standardize` | `risk/style_factors.py` |
| 股票基本信息(行业) | `fetch_stock_basic(...)` → 含 `industry` 列 | `core/loader.py:535` |
| 数据上下文 | `FactorDataContext` 的 `daily`/`daily_basic` | `daily/data/context.py` |
| universe | `get_universe(date, name)` | `core/universe.py` |
| CLI 接入 | `build_parser()`，仿 `fz mine` | `cli/main.py` |

新增 `pipelines/risk_build.py`（`run_risk_build(...)`）做数据拉取 + build + 落产物编排。

---

## 7. 测试策略 + 验收（DoD）

- [ ] 4 个新测试文件（industry/exposures/covariance/model）+ 现有 `test_style_factors`（6）全绿，纯 mock 离线。
- [ ] 协方差半正定、风险分解守恒、R²∈[0,1] 等性质有断言保护。
- [ ] `fz risk build` 端到端跑通（手动 smoke），产出 6 个产物 + manifest；同参数可复现。
- [ ] `risk_summary.csv` 含因子波动 / 特质风险分布 / R² / 风格暴露 / 组合风险分解示例。
- [ ] 测试暴露的 risk bug 已修并有回归断言。
- [ ] `git add` 只含 risk 相关文件（不带其它 M0 未提交改动）；ruff/typecheck/test 绿。

---

## 8. 建议实现顺序（为 writing-plans 铺垫）

> 先补测试（从最简单的 industry 到端到端 model），测试就位后再接 CLI/报告。补测试时若暴露 bug 顺手修。

1. **`test_risk_industry.py`**（最简单，get_industry_dummies）。
2. **`test_risk_exposures.py`**（compute_exposures + ExposureMatrix）。
3. **`test_risk_covariance.py`**（协方差/特质风险/eigenvector，性质断言）。
4. **`test_risk_model.py`**（RiskModel.build/predict_risk/decompose_risk 端到端 + 风险分解守恒）。
5. **`pipelines/risk_build.py` + 轻量报告**（run_risk_build：拉数据 build 落产物 + risk_summary.csv）。
6. **CLI `fz risk build`**（接入 build_parser + _cmd_risk_build）。
7. **收口提交**：把 risk/ 源码 + 新测试 + pipeline + CLI 一并提交入库。

---

## 9. 范围外
组合优化（M4 用风险模型）· 完整 HTML 风险报告（图表渲染）· 改写 risk 现有算法（仅在测试暴露 bug 时改）· 把风险模型接入挖掘/归因。

---

*M3 收口后，Barra 风险模型即有测试保护、能从 CLI 跑出、有可读摘要，成为可展示成果，并为 M4 组合优化铺路。*
