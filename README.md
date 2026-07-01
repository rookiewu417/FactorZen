<div align="center">

<img src="docs/assets/logo-horizontal-light.svg" alt="FactorZen logo" width="520">

# FactorZen

**端到端、可复现的 A 股量化研究平台**

把量化研究从「单因子 IC 检验」扩展为「因子挖掘 → 防过拟合 → 风险建模 → AI 智能挖掘 → 组合优化与归因 → 模拟交易 → 成果展示」的完整买方级链路，<br>
每一步都落 manifest、可审计、可复现。

[![CI](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml/badge.svg)](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[快速开始](#快速开始) · [文档](docs/README.md) · [架构](docs/architecture.md) · [运行手册](docs/runbook.md) · [路线图](docs/evolution-plan-2026.md) · [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html)

</div>

---

## 设计原则

FactorZen 把研究可信度放在展示效果之前。三条原则贯穿数据、评估到报告：

1. **数据 PIT 无未来函数** —— 停牌、涨跌停、ST、次新、T+1 等交易约束在口径层面就被约束，universe 快照全部 point-in-time，错误数据不进入下游。
2. **评估带样本外/防过拟合护栏** —— holdout 段永久隔离、block bootstrap IC CI、Deflated Sharpe、PBO/CSCV，多重检验从入库起就记账；样本不足、OOS 不成立会被显式标注，而不是用漂亮图表掩盖。
3. **一切产物落 manifest 可复现** —— 每次运行生成 `manifest.json`，记录配置、命令、git SHA、lockfile hash 与阶段耗时；种子/参数/数据版本全部固定，可任意重放。

---

## 核心能力矩阵

| 能力域 | 能力 | 模块路径 | 代表命令 |
|------|------|----------|----------|
| **基线** | 单因子研究链路：IC / 分层回测 / walk-forward / Tear Sheet | `daily/` `reports/` | `fz factor run <name>` |
| **微观结构与交易约束** | 交易约束（停牌/涨跌停/ST/次新/T+1）+ universe 快照 | `core/` `daily/evaluation/` | （内嵌回测） |
| **因子挖掘** | 算子库 + 表达式 AST↔字符串编译 + 随机/遗传搜索 + 贪心去相关 | `discovery/` | `fz mine search` |
| **防过拟合** | block bootstrap IC CI + Deflated Sharpe + PBO/CSCV + holdout 永久隔离 | `validation/` | `fz validate overfit` |
| **Barra 风险模型** | Barra 风格（8）+ 行业因子暴露 + Newey-West 协方差 + 特质风险收缩 + MCR 分解 | `risk/` | `fz risk build` |
| **组合优化与归因** | cvxpy 因子形式 mean-variance QP + box/budget/行业风格中性/换手约束；Brinson + 风险因子归因 | `portfolio/` `attribution/` | `fz portfolio build` |
| **单 Agent 挖掘** | LLM 闭环（假设→生成→护栏→critic→反思），零依赖自建 loop，Negative RAG | `agents/` | `fz mine agent` |
| **多 Agent 团队** | 4 角色 Agent（Hypothesis/Coder/Critic/Librarian）+ Evaluator 评估环节 + 跨轮否决 + 跨 session 长期记忆 | `agents/roles/` `team_orchestrator.py` `experiment_index.py` | `fz mine team` |
| **模拟交易** | 组合权重回测（对齐行情/扣换手成本/净值/夏普/最大回撤） | `sim/` | `fz sim run` |
| **成果展示** | 组合绩效 HTML Dashboard（指标卡+净值曲线+月度热图+归因+风险摘要） | `reports/portfolio_report.py` | `fz report portfolio` |

---

## 安装

推荐使用 [pixi](https://pixi.sh/) 管理环境（Python ≥3.10 <3.13）。所有命令从仓库根目录执行。

```bash
pixi install
cp .env.example .env
pixi run smoke
```

`.env` 不入库。真实数据拉取需要配置 `TUSHARE_TOKEN`；LLM 挖掘功能（单 / 多 Agent，`fz mine agent` / `fz mine team`）需要显式配置 `FACTORZEN_LLM_*`，缺失会直接报错退出（不会自动跳过）；仅报告的可选 LLM 解读功能在缺失配置时才会自动跳过。

---

## 快速开始

### 端到端链路（Step 0 前置数据 → 成果展示）

```bash
# 0. 拉数据（前置：Tushare → 本地 parquet 缓存）
pixi run fz data fetch daily --start 20200101 --end 20241231
pixi run fz data fetch daily-basic --start 20200101 --end 20241231

# 1. 挖因子（随机/遗传搜索 或 LLM Agent 团队）→ workspace/mining_sessions/session_<seed>_<method>/
pixi run fz mine search --start 20200101 --end 20231231 --method genetic --trials 200 --top-k 10
#   或: pixi run fz mine team --start 20200101 --end 20231231

# 2. 导出 α 截面（取候选榜第 1 名，在指定日生成 ts_code+alpha 两列 parquet）
pixi run fz mine export-alpha \
  --session workspace/mining_sessions/session_42_genetic --rank 1 \
  --date 20231231 --universe all_a --lookback 60 --out alpha.parquet

# 3. 防过拟合验收（对已注册因子：Deflated Sharpe + bootstrap IC CI；仅打印，不落盘）
pixi run fz validate overfit <factor> --start 20200101 --end 20231231

# 4. 建风险模型（Barra 因子暴露 + Newey-West 协方差）
pixi run fz risk build --start 20200101 --end 20231231 --universe all_a

# 5. 组合优化建仓（单截面：在 --end 当日解一次 QP → 目标权重 + 归因）
pixi run fz portfolio build --start 20200101 --end 20231231 \
  --alpha-file alpha.parquet --industry-neutral

# 6. 模拟交易（组合权重 → 净值/夏普/最大回撤）
pixi run fz sim run --portfolio-dir workspace/portfolios --start 20240101 --end 20241231

# 7. 成果展示页（指标卡 + 净值曲线 + 归因 → HTML Dashboard）
pixi run fz report portfolio \
  --sim-dir workspace/sim/<run_id> --portfolio-dir workspace/portfolios/<run_id>
```

### 单因子评估

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20230101 --end 20241231
pixi run fz report path <run_id>
```

无 `--config` 时使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化。walk-forward 默认关闭，按需用 `--set walk_forward.enabled=true` 开启。

**命令行调参，无需改 YAML**

`--set key=value` 在校验前覆盖任意配置字段，可重复，且仍写入 `manifest.json` 保持可复现：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 --set preprocessing.neutralize=true \
  --set walk_forward.train_days=252
```

**参数网格扫描**

```bash
pixi run fz factor sweep --config workspace/configs/daily/daily_factor_template.yaml \
  --grid backtest.top_n=30,50,100 --grid preprocessing.normalizer=zscore,rank_normal \
  --sort-by ir
```

---

## 输出产物

每次运行的产物写入独立目录，按类型分组：

```text
workspace/
├── mining_sessions/session_<seed>_<method>/  因子挖掘候选
│   ├── candidates.csv             候选排行（表达式 + IC）
│   ├── manifest.json              配置/命令/seed/git SHA
│   └── exported/*.py              可复现因子代码
├── factor_evaluations/{run_id}/   单因子评估
│   ├── report.html                Tear Sheet 报告（含分层回测结果，无独立 backtest 文件）
│   ├── manifest.json              配置/命令/git SHA/lockfile hash/阶段耗时
│   ├── factor.parquet             因子值
│   ├── ic.parquet                 IC 结果
│   ├── universe.parquet           universe 快照
│   ├── quality.json               数据质量报告
│   └── walk_forward.json          walk-forward 摘要
├── risk_models/{run_id}/          Barra 风险模型（exposures / factor_covariance / specific_risk + risk_summary.csv）
├── portfolios/{run_id}/           组合权重 + 归因（weights.parquet + attribution.csv + risk_summary.csv）
├── sim/{run_id}/                  模拟净值 + 绩效（nav.parquet + metrics.json）
└── reports/portfolio_<run_id>.html  组合绩效 HTML Dashboard
```

---

## 项目结构

```text
src/factorzen/
├── config/               路径、常量、Tushare 配置
├── core/                 日历、universe、存储、实验元数据
├── builtin_factors/      框架自带因子（daily/weekly/monthly/intraday/qlib），随包分发
├── daily/                低频主线：数据、因子、预处理、评估与优化
├── discovery/            因子挖掘：算子库 + 表达式 AST + 随机/遗传搜索
├── validation/           防过拟合：bootstrap IC CI + Deflated Sharpe + PBO/CSCV
├── risk/                 Barra 风险模型：因子暴露 + Newey-West 协方差
├── portfolio/            组合优化：cvxpy mean-variance QP + 约束体系
├── attribution/          绩效归因：Brinson + 风险因子归因
├── agents/               LLM 挖掘：单 Agent 闭环 + 多 Agent 团队
├── sim/                  模拟交易：组合权重净值回测
├── reports/              Tear Sheet + 组合 Dashboard 报告引擎
├── llm/                  可选 OpenAI-compatible 研究解读
├── pipelines/            端到端流程编排
├── intraday/             分钟线研究代码（当前非主线）
└── cli/                  fz 命令行入口
workspace/
├── factors/              自定义因子（默认空，与框架自带分离）
└── configs/              实验 YAML 配置
tests/                    pytest 测试（1185 个）
docs/                     架构、运行手册、因子编写指南
```

---

## 技术栈

- **Python** ≥3.10 <3.13，pixi 环境管理（conda-forge，win-64/linux-64）
- **数值**：polars ≥1.0 / numpy / scipy / pandas
- **优化**：cvxpy ≥1.4（CLARABEL solver）/ optuna
- **数据**：tushare / pyarrow
- **统计**：statsmodels
- **报告**：matplotlib / jinja2
- **工程**：pydantic / pyyaml
- **质量**：1185 pytest 测试 / ruff / mypy；CLI 入口 `pixi run fz`

---

## 适用边界

**适合**

- 在 A 股日/周/月频数据上评估单因子的稳定预测能力（IC、HAC t 统计、分层收益、换手、成本、容量约束、walk-forward 样本外）。
- 用表达式搜索或 LLM Agent 自动挖掘因子，并经防过拟合护栏验收。
- 用 Barra 风险模型控制因子暴露，凸优化建仓，通过模拟交易评估组合绩效。
- 产出可审计产物：`manifest.json`、universe 快照、parquet 结果、HTML 报告。

**不覆盖 / MVP 限制**

- **不接实盘 OMS/不做实盘下单**：FactorZen 提供组合优化与模拟交易闭环，但不接 OMS/EMS，不做实盘撮合与风控执行。
- **行业中性是相对等权基准**（MVP 限制）：`--industry-neutral` 约束基于等权行业基准，不等同于市场加权中性。
- **收益归因需持仓期收益**（MVP 限制）：Brinson 归因要求提供持仓期区间收益，不支持日内高频归因。
- `intraday/` 当前非主线；Tick 级研究与生产组合执行不纳入本框架。

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [项目说明](docs/project-explanation.md) | 系统事实、数据流、配置、质量门与边界 |
| [架构](docs/architecture.md) | 框架包、工作区、数据流与产物边界 |
| [运行手册](docs/runbook.md) | 常用命令、报告入口、数据拉取、故障处理 |
| [因子编写](docs/factor-authoring.md) | 因子放哪里、实现什么接口、如何验证 |
| [路线图](docs/evolution-plan-2026.md) | 公开路线图与非目标 |
| [发布记录 v0.3.0](docs/release-notes/v0.3.0.md) | 完整买方研究平台升级变更日志 |
| [端到端教程](docs/end-to-end-tutorial.md) | 手把手走完：从拉数据到组合 Dashboard |
| [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html) | 真实 Tear Sheet 示例（GitHub Pages） |

文档索引见 [docs/README.md](docs/README.md)。

---

## 开发

```bash
pixi run lint
pixi run format
pixi run typecheck
pixi run test
pixi run coverage
```

如本机已安装 `pre-commit`，提交前可启用本地钩子：

```bash
pre-commit install
```

---

## 安全

不要提交 `.env`、API token、商业行情数据或私有研究产物。安全策略与凭据轮换流程见 [SECURITY.md](SECURITY.md)。

---

## 许可

本项目以 [MIT License](LICENSE) 开源。
