# 文档

> [FactorZen](../README.md) · **文档**

按阅读路径分四层。每份文档职责单一、互不复述——需要交叉的内容用链接而不是复制。

## 入门 · getting-started/

第一次接触平台，按顺序读。

| 文档 | 内容 |
|---|---|
| [安装与环境](getting-started/installation.md) | pixi 环境、`.env` 凭据、可选依赖、装完怎么验证 |
| [快速上手](getting-started/quickstart.md) | 5 分钟跑通核心闭环：挖掘 → 增量准入 → 组合 |
| [端到端教程](getting-started/end-to-end-tutorial.md) | 完整链路逐步走：拉数据 → 挖掘 → 准入 → 风险 → 组合 → 模拟 → 报告 |

## 原理 · concepts/

理解平台为什么这样设计。改代码前建议先读。

| 文档 | 内容 |
|---|---|
| [架构](concepts/architecture.md) | 分层结构、端到端数据流、模块职责与边界 |
| [设计铁律](concepts/design-principles.md) | PIT 无未来函数、护栏咬合、可复现——三条原则的具体落地与已知例外 |
| [因子库与增量准入](concepts/factor-library.md) | **平台核心机制**：lift 裁决、四态状态机、probation → 向前确认 → 转正 |
| [防过拟合护栏](concepts/guardrails.md) | bootstrap IC CI、Deflated Sharpe、PBO/CSCV、holdout、空假设校准，以及它们如何咬合进筛选 |
| [多市场适配](concepts/multi-market.md) | Ports & Adapters 结构、四市场的真实能力边界 |

## 指南 · guides/

按任务查。

| 文档 | 内容 |
|---|---|
| [因子编写](guides/factor-authoring.md) | 手写因子放哪里、实现什么接口、如何入库与验证 |
| [因子挖掘](guides/mining.md) | 表达式搜索、LLM 单 Agent 与团队挖掘、日内叶子与 scout |
| [多因子组合](guides/combination.md) | 从因子库消费因子，四方法样本外对比 |
| [风险与组合优化](guides/risk-and-portfolio.md) | Barra 风险模型、凸优化建仓、归因 |
| [模拟与向前执行](guides/execution.md) | 模拟交易、向前执行引擎、分歧归因 |
| [无人值守运营](guides/operations.md) | 8 阶段日链路、告警、失败恢复 |
| [部署](guides/deployment.md) | Web 服务与定时任务部署 |
| [性能与资源](guides/performance.md) | 耗时基准、内存占用、并行与子进程隔离 |

## 参考 · reference/

查具体参数与字段。

| 文档 | 内容 |
|---|---|
| [CLI 参考](reference/cli.md) | 16 个顶层命令 / 47 个叶子命令，含参数表与示例 |
| [配置](reference/configuration.md) | 配置模型、YAML 模板、`--set` 覆盖机制 |
| [产物布局](reference/artifacts.md) | `workspace/` 与 `data/` 目录结构、`manifest.json` 字段 |
| [环境变量](reference/environment.md) | `TUSHARE_TOKEN`、`FACTORZEN_LLM_*` 全表与缺失行为 |
| [数据源与口径](reference/data-sources.md) | 各市场数据源、**单位口径**、缓存键完整性 |

## 其他

| 位置 | 内容 |
|---|---|
| [release-notes/](release-notes/) | 已发布版本说明（发布后不回写） |
| [../CHANGELOG.md](../CHANGELOG.md) | 变更日志 |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | 开发流程、验证要求、提交规范 |
| [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html) | 真实 Tear Sheet（GitHub Pages） |

作用域限定的局部说明随代码放置，不并入本目录：
[`tools/`](../tools/README.md) · [`builtin_factors/qlib/`](../src/factorzen/builtin_factors/qlib/README.md) · [`research/combination/`](../src/factorzen/research/combination/README.md)
