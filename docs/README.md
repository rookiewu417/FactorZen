# 文档索引

> [FactorZen](../README.md) · **文档** · [架构](architecture.md) · [运行手册](runbook.md) · [路线图](evolution-plan-2026.md)

本目录每份文档各司其职，边界清晰、互不复述。下表说明读者、职责与保留理由。

## 文档地图

| 文档 | 读者 | 职责 |
|------|------|------|
| [README](../README.md) | 新用户、开源访客 | 项目定位、安装、快速开始、核心结构 |
| [project-explanation](project-explanation.md) | 维护者、自动化代理 | 系统事实、数据流、配置、质量门与边界 |
| [architecture](architecture.md) | 维护者 | 框架包、工作区、数据流与产物边界（架构契约）|
| [factor-authoring](factor-authoring.md) | 因子作者 | 新因子放哪里、实现什么接口、如何验证 |
| [runbook](runbook.md) | 日常使用者、值守者 | 常用命令、报告入口、数据拉取、故障处理 |
| [evolution-plan-2026](evolution-plan-2026.md) | 维护者 | 公开路线图与非目标 |
| [release-notes/](release-notes/) | 发布使用者 | 已发布版本的历史说明（发布后不回写）|

## 局部 README

仓库内还有两份作用域明确的局部 README，**不并入根 README**：

- [`src/factorzen/builtin_factors/qlib/README.md`](../src/factorzen/builtin_factors/qlib/README.md) —— 只解释 qlib 因子与数据源。
- [`src/factorzen/research/combination/README.md`](../src/factorzen/research/combination/README.md) —— 只解释实验性多因子合成。
