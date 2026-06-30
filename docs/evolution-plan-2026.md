# 2026 演进计划

> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · [运行手册](runbook.md) · **路线图**

本文记录 FactorZen 的演进方向与完成状态。最新目录结构以 [README](../README.md) 与 [architecture](architecture.md) 为准。

## 阶段总览（平台升级全部完成）

| 能力 | 主题 | 状态 |
|------|------|------|
| 基线 | 单因子研究链路（IC / 分层回测 / walk-forward / Tear Sheet） | ✅ 已完成（v0.2.0）|
| 微观结构与交易约束 | universe 快照 + 涨跌停/T+1 + 基准管理 | ✅ 已完成（v0.3.0）|
| 因子挖掘引擎 | 算子 + 表达式 AST + 随机/遗传搜索 | ✅ 已完成（v0.3.0）|
| 防过拟合护栏 | DSR + bootstrap CI + PBO + holdout 隔离 | ✅ 已完成（v0.3.0）|
| Barra 风险模型 | 风格 + 行业 + 协方差 + 风险贡献 | ✅ 已完成（v0.3.0）|
| 组合优化与归因 | QP + Brinson + 风险归因 | ✅ 已完成（v0.3.0）|
| 单 Agent 挖掘 | LLM 假设→生成→护栏→反思闭环 | ✅ 已完成（v0.3.0）|
| 多 Agent 团队 | 4 角色 Agent + Evaluator 评估 + 长期记忆 | ✅ 已完成（v0.3.0）|
| 模拟交易 + 成果展示 | 多周期净值 + 组合绩效 Dashboard | ✅ 已完成（v0.3.0）|

## 当前定位（v0.3.0）

FactorZen 已从单因子研究框架升级为**端到端、可复现的 A 股量化研究平台**，完整链路：

```text
本地数据缓存（Tushare → parquet）
  → PIT 数据上下文 + 微观结构约束
  → 因子计算 + 预处理
  → 因子挖掘（随机/遗传搜索 / LLM Agent / 多 Agent 团队）
  → 防过拟合护栏（holdout 永久隔离 / DSR / bootstrap CI / PBO）
  → Barra 风险模型（风格 + 行业暴露 + 协方差）
  → 组合优化建仓（mean-variance QP + 行业中性 + 换手约束）+ 归因
  → 模拟交易（多周期净值 + 绩效指标）
  → 成果展示（组合绩效 HTML Dashboard）
```

设计三原则（强化保留）：
1. 数据 PIT：无未来函数。
2. 评估带 OOS 护栏：holdout 永久隔离 + Deflated Sharpe + PBO。
3. 一切产物落 manifest（seed/参数/git_sha）可复现。

## 优先级原则

1. 数据正确性优先于下游结论。
2. 可复现性优先于展示效果。
3. 报告结论必须暴露样本不足、覆盖率不足与模块缺失。
4. 新能力进入主线前必须有测试与质量门保护。

## 微观结构与交易约束 — ✅ 已完成

已落地：

- `core/universe.py` universe 快照（停牌/涨跌停/ST/次新股/流通市值过滤，日级快照）。
- `daily/evaluation/benchmark.py` 策略 vs 指数基准超额收益对比（HS300/ZZ500/ZZ1000 等真实指数日线，年化超额/跟踪误差/信息比率/超额回撤）；`core/benchmark.py` 是流水线步骤耗时/峰值内存的性能计时工具，与金融基准无关。
- `daily/evaluation/backtest.py` 涨跌停判断的浮点比较容差（`1e-9`，按板块阈值：主板 9.8%/创业板及科创板 19.8%/北交所 29.8%，防止开盘涨跌幅边界浮点误差导致漏判涨停/跌停）+ T+1 + `signal_date` 显式集成 + ADV 零值 fallback。

## 因子挖掘引擎 — ✅ 已完成

已落地：

- `discovery/operators.py` 算子库（30+ 时序/截面/算术算子，group-safe `pct_change`，`ts_rank` 强制 `min_samples`）。
- `discovery/expression.py` 表达式 AST ↔ 字符串双向序列化。
- `discovery/mining_session.py` + `discovery/search/`（`random_search.py` / `genetic.py`）随机 + 遗传搜索，落 session 产物。
- `discovery/scoring.py` IC 打分 + 贪心去相关筛选；`discovery/export.py` 候选导出单截面 alpha（`fz mine export-alpha`）。

## 防过拟合护栏 — ✅ 已完成

已落地：

- `validation/bootstrap.py` block bootstrap IC 置信区间。
- `validation/deflated_sharpe.py` Deflated Sharpe Ratio（DSR）。
- `validation/pbo.py` PBO/CSCV（候选池过拟合概率）。
- `validation/holdout.py` holdout 段永久隔离机制。

## Barra 风险模型 — ✅ 已完成

已落地：

- `risk/style_factors.py` Barra 风格（8 因子：`size` / `value` / `momentum` / `volatility` / `liquidity` / `quality` / `growth` / `leverage`）；`risk/industry_factors.py` 行业暴露（按 Tushare `stock_basic.industry` 字段分类，one-hot 编码）。
- `risk/covariance.py` Newey-West 协方差 + 半衰期衰减。
- `risk/exposures.py` + `risk/model.py` 暴露矩阵组装 + 风险模型收口（含 James-Stein 特质风险收缩、边际风险贡献分解）。

## 组合优化与归因 — ✅ 已完成

已落地：

- `portfolio/optimizer.py` + `portfolio/constraints.py` cvxpy mean-variance QP（CLARABEL solver）：box / 预算 / 换手 / 行业中性约束。
- `attribution/brinson.py` Brinson 多期归因。
- `attribution/risk_attribution.py` 风险因子归因。

## 单 Agent 挖掘 — ✅ 已完成

已落地：

- `agents/orchestrator.py` + `agents/nodes.py` 零外部依赖自建 LLM 闭环：假设 → 表达式 → 护栏 → IC 验证 → critic 反思。
- `agents/memory.py` Negative RAG 历史失败注入。
- `agents/manifest.py` 候选 manifest 落盘可审计。

## 多 Agent 团队 — ✅ 已完成

已落地：

- `agents/roles/` 4 个角色 Agent（Hypothesis / Coder / Critic / Librarian）+ `agents/evaluation.py` IC 评估环节。
- `agents/team_orchestrator.py` 跨轮 Critic 否决机制。
- `agents/experiment_index.py` 跨 session 长期记忆。

## 模拟交易 + 成果展示 — ✅ 已完成

已落地：

- `sim/engine.py` 多周期权重回测（对齐行情 / 扣换手成本 / 净值序列 + 绩效指标：年化收益 / 夏普 / 最大回撤 / 卡尔玛 / 换手 / IR）。
- `reports/portfolio_report.py` 组合绩效 HTML Dashboard（指标卡 + 净值曲线 + 月度热图 + 归因 + 风险摘要）。

## 未来可选增强（无承诺时间表）

以下方向均尚未实现，可按需扩展，具体接口现状见下表说明：

| 增强项 | 说明 |
|--------|------|
| 真实指数成分权重 | 组合优化的行业中性约束目前用等权基准（MVP 限制）；接入真实沪深 300/中证 500 成分权重可替换为市值加权基准（当前未实现，无预留接口） |
| 跟踪误差约束（TEV） | 组合优化加入显式 TE 约束（当前通过换手约束近似） |
| 遗传搜索并行化 | 搜索引擎多进程 / 分布式并行 |
| 实盘 OMS 对接 | 模拟交易输出→实盘下单接口（不在当前路线图） |
| 高频/日内因子 | `intraday/` 路径保留，分钟线因子评估框架 |
| 多因子组合 OOS 估权重 | `research/combination/` 当前为样本内工具；补 OOS 估权 |

## 非目标（长期边界）

- 不内置商业行情数据。
- 不承诺生产交易或实盘执行能力（不接实盘 OMS，不做实盘下单）。
- 不把真实 Tushare 网络请求放入默认 CI（CI 保持离线可重复）。
- 不把本地 `data/`、`workspace/` 运行产物提交到仓库。

## 质量门（持续维护）

```bash
pixi run lint
pixi run typecheck
pixi run test        # 1111 用例，全部离线可重复
pixi run coverage    # 门槛 ≥74%
git status --short
```

同时确认 `.env`、本地行情数据、运行日志、代理状态目录与任何 token 都没有被跟踪。
