# 文档索引

> [FactorZen](../README.md) · **文档** · [架构](architecture.md) · [运行手册](runbook.md) · [路线图](evolution-plan-2026.md)

本目录每份文档各司其职，边界清晰、互不复述。下表说明读者、职责与保留理由。

## 文档地图

| 文档 | 读者 | 职责 |
|------|------|------|
| [README](../README.md) | 新用户、开源访客 | 项目定位、安装、快速开始、核心结构 |
| [project-explanation](project-explanation.md) | 维护者、自动化代理 | 系统事实、数据流、配置、质量门与边界 |
| [architecture](architecture.md) | 维护者 | 平台分层架构（M0-M7）、数据流、产物边界与关键设计原则（架构契约） |
| [factor-authoring](factor-authoring.md) | 因子作者 | 新因子放哪里、实现什么接口、如何验证 |
| [runbook](runbook.md) | 日常使用者、值守者 | 常用命令、报告入口、数据拉取、故障处理 |
| [evolution-plan-2026](evolution-plan-2026.md) | 维护者 | 公开路线图与非目标 |
| [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html) | 新用户、复核者 | 真实 tear sheet 示例 |
| [release-notes/](release-notes/) | 发布使用者 | 已发布版本的历史说明（发布后不回写） |

---

## M1-M7 设计文档（Specs）

M1-M7 升级的设计文档存放于 `docs/superpowers/specs/`，每份对应一个里程碑的技术设计与接口约定。

| 里程碑 | 设计文档 | 一句话 |
|--------|----------|--------|
| M1 因子挖掘 | [2026-06-29-m1-factor-mining-design.md](superpowers/specs/2026-06-29-m1-factor-mining-design.md) | 算子库 + 表达式 AST + 随机/遗传搜索设计 |
| M2 防过拟合 | [2026-06-30-m2-overfit-guards-design.md](superpowers/specs/2026-06-30-m2-overfit-guards-design.md) | holdout 隔离 + Deflated Sharpe + PBO/CSCV 设计 |
| M3 风险模型 | [2026-06-30-m3-risk-model-finalization-design.md](superpowers/specs/2026-06-30-m3-risk-model-finalization-design.md) | Barra 风格 + 行业因子 + Newey-West 协方差设计 |
| M4 组合优化 | [2026-06-30-m4-portfolio-optimization-design.md](superpowers/specs/2026-06-30-m4-portfolio-optimization-design.md) | cvxpy QP 建仓 + Brinson 归因设计 |
| M5 单 Agent | [2026-06-30-m5-llm-agent-mining-design.md](superpowers/specs/2026-06-30-m5-llm-agent-mining-design.md) | LLM 假设→生成→护栏→反思闭环设计 |
| M6 多 Agent | [2026-06-30-m6-multi-agent-mining-design.md](superpowers/specs/2026-06-30-m6-multi-agent-mining-design.md) | 5 角色团队 + 跨 session 长期记忆设计 |

---

## M0-M7 实现计划（Plans）

实现计划存放于 `docs/superpowers/plans/`，记录 TDD 任务、进度 checkbox 与阶段里程碑。

| 计划文件 | 目标摘要 |
|----------|----------|
| [2026-06-29-m1-factor-mining-engine.md](superpowers/plans/2026-06-29-m1-factor-mining-engine.md) | M1 挖掘引擎 MVP：`fz mine search` 端到端产出 top-K + manifest |
| [2026-06-30-m2-overfit-guards.md](superpowers/plans/2026-06-30-m2-overfit-guards.md) | M2 防过拟合护栏：bootstrap CI + Deflated Sharpe + holdout 隔离 |
| [2026-06-30-m3-risk-model-finalization.md](superpowers/plans/2026-06-30-m3-risk-model-finalization.md) | M3 风险模型收口：补测试 + `fz risk build` CLI + 轻量风险报告 |
| [2026-06-30-m4-portfolio-optimization.md](superpowers/plans/2026-06-30-m4-portfolio-optimization.md) | M4 组合优化 + Brinson 归因：α + 风险模型 → 目标权重 |
| [2026-06-30-m5-llm-agent-mining.md](superpowers/plans/2026-06-30-m5-llm-agent-mining.md) | M5 LLM 单 Agent 挖掘闭环：`fz mine agent` 全程可审计 |
| [2026-06-30-m6-multi-agent-mining.md](superpowers/plans/2026-06-30-m6-multi-agent-mining.md) | M6 多 Agent 团队 + 长期记忆：`fz mine team` |
| [2026-06-30-m0-m7-finalize-and-showcase.md](superpowers/plans/2026-06-30-m0-m7-finalize-and-showcase.md) | M0 收口 + M7 模拟交易 + 成果展示页：完整链路封装 |

---

## 早期计划（Plans Archive）

`docs/plans/` 存放平台升级前的旧计划，供历史参考：

| 计划文件 | 说明 |
|----------|------|
| [2026-06-04-docs-polish-plan.md](plans/2026-06-04-docs-polish-plan.md) | 文档打磨计划（M0 前） |
| [2026-06-05-cli-set-override-and-sweep.md](plans/2026-06-05-cli-set-override-and-sweep.md) | CLI `--set` 覆盖与 `sweep` 参数扫描 |

---

## 局部 README

仓库内还有两份作用域明确的局部 README，**不并入根 README**：

- [`src/factorzen/builtin_factors/qlib/README.md`](../src/factorzen/builtin_factors/qlib/README.md) —— 只解释 qlib 因子与数据源。
- [`src/factorzen/research/combination/README.md`](../src/factorzen/research/combination/README.md) —— 只解释实验性多因子合成。
