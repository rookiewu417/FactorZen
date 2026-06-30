# 2026 演进计划

> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · [运行手册](runbook.md) · **路线图**

本文记录 FactorZen 的演进方向与完成状态。最新目录结构以 [README](../README.md) 与 [architecture](architecture.md) 为准。

## 阶段总览（M0-M7 全部完成）

| 阶段 | 主题 | 状态 |
|------|------|------|
| 基线 | 单因子研究链路（IC / 分层回测 / walk-forward / Tear Sheet） | ✅ 已完成（v0.2.0）|
| M0 | 微观结构与回测约束 | ✅ 已完成（v0.3.0）|
| M1 | 因子挖掘引擎（算子 + AST + 遗传搜索） | ✅ 已完成（v0.3.0）|
| M2 | 防过拟合护栏（DSR + bootstrap CI + PBO） | ✅ 已完成（v0.3.0）|
| M3 | Barra 风险模型（风格 + 行业 + 协方差 + MCR） | ✅ 已完成（v0.3.0）|
| M4 | 组合优化 + 归因（QP + Brinson + 风险归因） | ✅ 已完成（v0.3.0）|
| M5 | LLM 单 Agent 因子挖掘 | ✅ 已完成（v0.3.0）|
| M6 | 多 Agent 团队（5 角色 + 长期记忆） | ✅ 已完成（v0.3.0）|
| M7 | 模拟交易 + 组合绩效 Dashboard | ✅ 已完成（v0.3.0）|

## 当前定位（v0.3.0）

FactorZen 已从单因子研究框架升级为**端到端、可复现的 A 股量化研究平台**，完整链路：

```text
本地数据缓存（Tushare → parquet）
  → PIT 数据上下文 + 微观结构约束（M0）
  → 因子计算 + 预处理
  → 因子挖掘（随机/遗传搜索 / LLM Agent / 多 Agent 团队）（M1/M5/M6）
  → 防过拟合护栏（holdout 永久隔离 / DSR / bootstrap CI / PBO）（M2）
  → Barra 风险模型（风格 + 行业暴露 + 协方差）（M3）
  → 组合优化建仓（mean-variance QP + 行业中性 + 换手约束）+ 归因（M4）
  → 模拟交易（多周期净值 + 绩效指标）（M7）
  → 成果展示（组合绩效 HTML Dashboard）（M7）
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

## M0 · 微观结构 — ✅ 已完成

已落地：

- `core/universe.py` universe 快照（停牌/涨跌停/ST/次新股/流通市值过滤，日级快照）。
- `core/benchmark.py` 基准管理（HS300、ZZ500、ZZ1000 + 行业等权替代基准）。
- `daily/evaluation/backtest.py` GEM 双路径容差 + T+1 + `signal_date` 显式集成 + ADV 零值 fallback。

## M1 · 因子挖掘 — ✅ 已完成

已落地：

- `discovery/operators.py` 算子库（30+ 时序/截面/算术算子，group-safe `pct_change`，`ts_rank` 强制 `min_samples`）。
- `discovery/expression.py` 表达式 AST ↔ 字符串双向序列化。
- `discovery/search.py` 随机 + 遗传搜索。
- `discovery/dedup.py` 贪心去相关筛选。

## M2 · 防过拟合 — ✅ 已完成

已落地：

- `validation/bootstrap.py` block bootstrap IC 置信区间。
- `validation/deflated_sharpe.py` Deflated Sharpe Ratio（DSR）。
- `validation/pbo.py` PBO/CSCV。
- holdout 段永久隔离机制。

## M3 · Barra 风险模型 — ✅ 已完成

已落地：

- `risk/factor_model.py` Barra 风格（8 因子）+ 行业（中信一级）暴露。
- `risk/covariance.py` Newey-West 协方差 + 半衰期衰减。
- `risk/specific_risk.py` James-Stein 收缩特质风险。
- `risk/mcr.py` 边际风险贡献分解。

## M4 · 组合优化 + 归因 — ✅ 已完成

已落地：

- `portfolio/optimizer.py` cvxpy mean-variance QP（CLARABEL solver）：box / 预算 / 换手 / 行业中性约束。
- `attribution/brinson.py` Brinson 多期归因。
- `attribution/risk_attribution.py` 风险因子归因。

## M5 · LLM 单 Agent — ✅ 已完成

已落地：

- `agents/agent.py` 零外部依赖自建 LLM 闭环：假设 → 表达式 → 护栏 → IC 验证 → critic 反思。
- Negative RAG 历史失败注入。
- 候选 manifest 落盘可审计。

## M6 · 多 Agent 团队 — ✅ 已完成

已落地：

- `agents/roles/` 5 角色（Hypothesis / Coder / Critic / Librarian / Evaluator）。
- `agents/team_orchestrator.py` 跨轮 Critic 否决机制。
- `agents/experiment_index.py` 跨 session 长期记忆。

## M7 · 模拟交易 + 展示 — ✅ 已完成

已落地：

- `sim/runner.py` 多周期权重回测（对齐行情 / 扣换手成本 / 净值序列）。
- `sim/metrics.py` 绩效指标（年化收益 / 夏普 / 最大回撤 / 卡尔玛 / 换手 / IR）。
- `reports/portfolio_report.py` 组合绩效 HTML Dashboard（指标卡 + 净值曲线 + 月度热图 + 归因 + 风险摘要）。

## 未来可选增强（无承诺时间表）

以下方向在 M0-M7 内已预留接口，但尚未实现，可按需扩展：

| 增强项 | 说明 |
|--------|------|
| `fetch_index_weights` 真实指数基准 | M0 基准管理已预留接口；接入真实沪深 300/中证 500 成分权重 |
| 跟踪误差约束（TEV） | M4 组合优化加入显式 TE 约束（当前通过换手约束近似） |
| 遗传搜索并行化 | M1 搜索引擎多进程 / 分布式并行 |
| 实盘 OMS 对接 | M7 模拟交易输出→实盘下单接口（不在当前路线图） |
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
pixi run test        # 1109 用例，全部离线可重复
pixi run coverage    # 门槛 ≥70%
git status --short
```

同时确认 `.env`、本地行情数据、运行日志、代理状态目录与任何 token 都没有被跟踪。
