# 架构

> [FactorZen](../README.md) · [文档](README.md) · **架构** · [运行手册](runbook.md) · [路线图](evolution-plan-2026.md)

FactorZen 是端到端、可复现的 A 股量化研究平台。平台由**八层能力**构成：每层职责独立、产物落 manifest、接口稳定，层间只通过标准化 parquet / JSON 传递数据。

---

## 平台分层架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│  AI 编排层 (agents/)                                                  │
│  单 Agent 挖掘闭环 ←→ 多 Agent 团队（4 角色：Hypothesis/Coder/       │
│  Critic/Librarian + Evaluator 评估环节）+ 跨 session 长期记忆         │
│  横跨因子层与评估层，通过 fz mine agent / fz mine team 驱动           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ LLM 生成表达式 / 反思迭代
┌──────────────────────────────▼──────────────────────────────────────┐
│  因子层 (daily/  discovery/)                                         │
│  单因子研究链路：DailyFactor 基类 / PIT 上下文 / 预处理               │
│  因子挖掘：算子库 + 表达式 AST↔字符串编译 + 随机/遗传搜索             │
└──────┬──────────────────────────────────────────────────────────────┘
       │ 因子值 parquet
┌──────▼──────────────────────────────────────────────────────────────┐
│  评估层 (daily/evaluation/  validation/)                             │
│  IC / 分层回测 / walk-forward / 防过拟合护栏                          │
│  block bootstrap CI + Deflated Sharpe + PBO/CSCV + holdout 永久隔离  │
└──────┬──────────────────────────────────────────────────────────────┘
       │ α 信号 parquet（discovery export-alpha）/ 护栏通过标志
┌──────▼──────────────────────────────────────────────────────────────┐
│  风险层 (risk/)                                                       │
│  Barra 风险模型：8 风格因子 + 行业因子暴露                             │
│  Newey-West 协方差 + 特质风险收缩 + MCR（边际风险贡献）分解            │
└──────┬──────────────────────────────────────────────────────────────┘
       │ 因子协方差矩阵 / 因子暴露矩阵
┌──────▼──────────────────────────────────────────────────────────────┐
│  组合层 (portfolio/  attribution/)                                   │
│  cvxpy 因子形式 mean-variance QP（单截面建仓）                         │
│  约束：box / budget / 行业风格中性 / 换手上限                          │
│  Brinson 归因 + 风险因子归因                                          │
└──────┬──────────────────────────────────────────────────────────────┘
       │ 目标权重 parquet + 归因 csv
┌──────▼──────────────────────────────────────────────────────────────┐
│  执行层 (sim/)                                                        │
│  组合权重回测：对齐行情 / 扣换手成本 / 净值 / 夏普 / 最大回撤          │
└──────┬──────────────────────────────────────────────────────────────┘
       │ 净值序列 / 绩效指标
┌──────▼──────────────────────────────────────────────────────────────┐
│  展示层 (reports/)                                                    │
│  单因子 Tear Sheet HTML + 组合 Dashboard                              │
│  指标卡 + 净值曲线 + 月度热图 + 归因可视化 + 风险摘要                  │
└─────────────────────────────────────────────────────────────────────┘

基础：数据层 (core/  data/)
  universe 快照 / PIT 无未来函数 / benchmark / 日历 / Tushare 加载器
  微观结构与交易约束：停牌/涨跌停/ST/次新/T+1 内嵌于 universe 快照
```

---

## 端到端数据流

```mermaid
graph TD
    TS[("Tushare API")] -->|fz data fetch| DATA[("data/ parquet 缓存")]

    DATA --> CORE["core/<br/>universe / PIT / calendar / benchmark"]

    CORE --> DAILY["daily/<br/>DailyFactor + 预处理 + IC/回测"]
    CORE --> DISC["discovery/<br/>算子库 + AST + 随机/遗传搜索"]
    CORE --> RISK["risk/<br/>Barra 风险模型"]

    DISC -->|候选表达式| DAILY
    AGENTS["agents/<br/>单 Agent / 多 Agent 团队"] -->|生成表达式| DISC

    DAILY --> VAL["validation/<br/>holdout 隔离 + Deflated Sharpe + bootstrap CI（仅打印）"]

    DISC -->|export-alpha → alpha.parquet| PORT["portfolio/<br/>cvxpy QP 优化（单截面）"]
    RISK -->|因子协方差 + 暴露| PORT
    RISK --> ATTR["attribution/<br/>Brinson + 风险因子归因"]
    PORT --> ATTR

    PORT -->|目标权重| SIM["sim/<br/>组合权重回测"]

    SIM --> RPT["reports/<br/>Tear Sheet + 组合 Dashboard"]
    ATTR --> RPT

    RPT --> WS[("workspace/<br/>reports/portfolio_&lt;run_id&gt;.html")]
```

> 纯文本版数据流（不支持 mermaid 时）：
>
> ```text
> Tushare → data/ → core/(universe/PIT/benchmark)
>   ├─→ daily/(因子评估) ←── discovery/(表达式搜索 + export-alpha) ←── agents/(LLM)
>   │        │                          │ alpha.parquet
>   │   validation/(防过拟合护栏，仅打印)  │
>   │                                    ▼
>   ├─→ risk/(Barra 协方差) ──→ portfolio/(凸优化) ──→ attribution/(归因)
>   │                                  │
>   │                             sim/(组合回测)
>   │                                  │
>   └─────────────────────────── reports/(Tear Sheet + Dashboard)
>                                      │
>                          workspace/reports/portfolio_<run_id>.html
> ```

---

## 模块职责表

| 能力域 | 模块路径 | 职责 |
|------|----------|------|
| 基础 | `config/` | 集中路径、研究常量、Tushare 配置 |
| 基础 | `core/` | 日历、universe 快照、存储、加载、数据审计、配置校验、实验 manifest、计时与日志 |
| 微观结构与交易约束 | `core/universe.py` `daily/evaluation/benchmark.py` `daily/evaluation/backtest.py` | 停牌/涨跌停/ST/次新/T+1 交易约束、universe 快照（PIT）、策略 vs 指数基准超额收益对比 |
| 基线 | `daily/` | 低频主线：PIT 数据上下文、因子基类、预处理、IC、回测、归因、成本与优化 |
| 因子挖掘 | `discovery/` | 算子库（时序/截面/算术）+ 表达式 AST↔字符串双向编译 + 随机/遗传搜索 + 贪心去相关 + 截面 α 导出 |
| 防过拟合 | `validation/` | block bootstrap IC CI + Deflated Sharpe Ratio + PBO/CSCV + holdout 永久隔离、多重检验记账 |
| Barra 风险模型 | `risk/` | Barra 风格（8 因子：size/value/momentum/volatility/liquidity/quality/growth/leverage）+ 行业因子暴露 + Newey-West 协方差 + 特质风险收缩 + MCR 分解 |
| 组合优化与归因 | `portfolio/` `attribution/` | cvxpy 因子形式 mean-variance QP（CLARABEL solver）、约束体系；Brinson + 风险因子归因 |
| 单 Agent 挖掘 | `agents/` | LLM 闭环（假设→生成→护栏→critic→反思），零依赖自建 loop，Negative RAG |
| 多 Agent 团队 | `agents/roles/` `agents/team_orchestrator.py` `agents/experiment_index.py` | 4 角色 Agent（Hypothesis/Coder/Critic/Librarian）+ Evaluator 评估环节 + 跨轮否决 + 跨 session 长期记忆（`ExperimentIndex`） |
| 模拟交易 | `sim/` | 组合权重回测：对齐行情、扣换手成本、净值序列、夏普、最大回撤 |
| 成果展示 | `reports/portfolio_report.py` | 组合 HTML Dashboard：指标卡 + 净值曲线 + 月度热图 + 归因 + 风险摘要 |
| 基础 | `reports/` | 单因子 Tear Sheet 报告引擎、图表、评分、摘要与模板 |
| 基础 | `pipelines/` | 端到端流程编排：`daily_single` / `generate_report` / `factor_mine*` / `risk_build` / `portfolio_build` |
| 基础 | `cli/` | 统一 `fz` 命令行入口 |

---

## 产物边界与 workspace 结构

```text
workspace/
  mining_sessions/session_<seed>_<method>/
    manifest.json           # 配置 / seed / git_sha
    candidates.csv          # 候选排行（rank / n_trials / expression / ic_train ...）
    exported/*.py           # 可复现因子代码

  factor_evaluations/{run_id}/
    manifest.json           # 配置 / git_sha / pixi.lock hash / 阶段耗时
    factor.parquet          # 因子值
    ic.parquet              # IC 时序
    universe.parquet        # universe 快照
    quality.json            # 数据质量报告
    walk_forward.json       # walk-forward 摘要
    report.html             # Tear Sheet（含分层回测结果，无独立 backtest 文件）

  risk_models/{run_id}/
    manifest.json
    exposures.parquet
    factor_covariance.parquet
    specific_risk.parquet
    factor_returns.parquet
    risk_summary.csv        # 长表：section / metric / value

  portfolios/{run_id}/
    manifest.json
    weights.parquet         # 单截面目标权重（ts_code / target_weight / prev_weight）
    attribution.csv         # Brinson + 风险因子归因
    risk_summary.csv        # 组合风险分解

  sim/{run_id}/
    manifest.json
    nav.parquet             # 净值序列
    metrics.json            # 夏普 / 最大回撤 / 年化收益 / 换手 / 成本

  reports/
    portfolio_<run_id>.html # 组合 Dashboard（成果展示）

data/                       # Tushare parquet 缓存（不提交 git）
```

每个 `run_id` 目录下必有 `manifest.json`，记录：配置 YAML、命令行参数、`git_sha`、`pixi.lock` hash、工作树 dirty 状态、各阶段耗时、输出路径、成功/失败状态。跨运行索引写入 `workspace/factor_evaluations/experiment_index.jsonl`，由 `fz runs list` / `fz runs show` 查询。

---

## 关键设计原则

### 1. PIT 无未来函数

所有因子计算、universe 过滤均使用 point-in-time 数据上下文（`daily/data/`）。财务数据按公告日对齐，不使用报告期末数据。universe 快照在调仓前 T 日生成，确保停牌/涨跌停/ST 约束无穿越。

### 2. Holdout 永久隔离

防过拟合护栏要求：
- 训练/验证期与 holdout（OOS）期严格分离，holdout 段在整个研究流程中**只用一次**（最终验收），不参与任何参数调优。
- 每次挖掘/搜索的 `trial_count`（多重检验次数）登记到 `ExperimentIndex`，用于 Deflated Sharpe 矫正。

### 3. 命名空间分离

| 命名空间 | 用途 |
|----------|------|
| `daily/optimization/` | 单因子研究流中的截面股票选择（IC 最大化） |
| `portfolio/` | 组合构建流：α 信号 + 风险模型 → 目标权重（QP 优化） |

两者接口不混用：`daily/` 产出的 `alpha_score` 文件作为 `portfolio/` 的输入，通过 parquet 解耦。

### 4. 可复现性

- 随机搜索、遗传算法均记录 seed，可从 manifest 完整复现。
- LLM Agent 调用记录 prompt/response（含 model id）到 `agents/logs/`，可复现推理路径。
- `pixi.lock` 锁定完整依赖树，确保跨机器一致。

### 5. agents/ 横跨因子层

AI 编排层（`agents/`）不直接写因子文件，而是通过 `discovery/` 的算子库和 AST 编译器生成表达式字符串，再走防过拟合护栏评估。Agent 只读 `ExperimentIndex`（避免重复挖掘），不直接修改 `workspace/`。

---

## 明确非目标

- 不内置商业行情数据（需自行配置 Tushare token）。
- 提供组合优化 + 模拟交易闭环，但**不接实盘 OMS/EMS，不做实盘下单**。
- 不把 Tushare 网络请求放进默认 CI（CI 全用 mock）。
- 不把 `data/`、`workspace/` 的本地产物提交到仓库。
- 不把 `intraday/` 与 Tick 级研究作为当前主线。
