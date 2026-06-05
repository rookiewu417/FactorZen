# 架构

> [FactorZen](../README.md) · [文档](README.md) · **架构** · [运行手册](runbook.md) · [路线图](evolution-plan-2026.md)

FactorZen 分为**框架包**与**工作区**两层：框架包提供稳定接口，工作区承载日常研究资产。

```text
src/factorzen/                  框架代码（稳定接口）
src/factorzen/builtin_factors/  框架自带因子（随包分发）
workspace/factors/              你的自定义因子（默认空）
workspace/configs/              实验 YAML 配置
workspace/factor_evaluations/   单次运行的自包含输出
workspace/runs/                 调度日志和中间产物
data/                           本地市场数据缓存
tests/                          pytest 测试
tools/                          维护与性能辅助脚本
docs/                           项目文档
```

## 分层职责

| 模块 | 职责 |
|------|------|
| `config/` | 集中路径、研究常量与 Tushare 配置 |
| `core/` | 日历、universe、存储、加载、数据审计、配置校验、实验 manifest、计时与日志 |
| `daily/` | 低频主线：PIT 数据上下文、因子基类、预处理、IC、回测、归因、成本与优化 |
| `intraday/` | 分钟线研究代码，当前不作为主线演进目标 |
| `pipelines/` | `daily_single` 与 `generate_report` 端到端流程 |
| `reports/` | Tear Sheet 报告引擎、图表、评分、摘要与模板 |
| `llm/` | 可选 OpenAI-compatible 研究解读，默认关闭 |
| `research/combination/` | 实验性多因子合成，不应解读为无偏 OOS 组合表现 |
| `cli/` | 统一 `fz` 命令行入口 |
| `workspace/` | 用户新增因子、实验配置与本地运行产物 |

## 数据流

低频主线从本地缓存出发，逐级走到 HTML 报告：

```text
本地 parquet 缓存 (data/)
        │
        ▼
PIT 数据上下文 (daily/data)
        │
        ▼
因子计算 (daily/factors) ──► 预处理 (daily/preprocessing)
        │
        ▼
前向收益 + IC / 回测 / 换手 / walk-forward / 归因 / 基准
        │
        ▼
报告引擎 (reports/tear_sheet)
        │
        ▼
workspace/factor_evaluations/{run_id}/report.html
```

`pipelines/daily_single.py` 与 `pipelines/generate_report.py` 负责编排上述步骤，由 `fz factor run` / `fz report build` 调用。兼容入口仍保留，但新增流程优先使用 `fz`。

## 因子边界

框架自带因子（示例与测试用）在 `src/factorzen/builtin_factors/`，随包分发；你的研究因子按频率放在 `workspace/factors/` 下（默认空）：

```text
src/factorzen/builtin_factors/{daily,weekly,monthly,intraday,qlib}/   框架自带
workspace/factors/{daily,weekly,monthly,intraday}/                   你的因子
```

注册表同时扫描两组，同名时 `workspace`（用户）覆盖 `builtin_factors`（框架）。`src/factorzen/daily/factors/` 与 `src/factorzen/intraday/factors/` 只放框架基类和注册中心。**你的日常研究因子始终写在 `workspace/factors/`，不要写进 `src/`（`builtin_factors/` 由框架维护）。**

## 产物与可复现

每次评估写入 `workspace/factor_evaluations/{run_id}/`。标准产物包括 `report.html`、`manifest.json`、`universe.parquet`、IC / 回测 parquet，以及运行期间登记到 manifest 的附加输出。

`manifest.json` 记录配置、命令、git SHA、工作树 dirty 状态、`pixi.lock` hash、阶段耗时、输出路径与成功/失败状态。跨运行索引写入 `workspace/factor_evaluations/experiment_index.jsonl`，由 `fz runs list` 与 `fz runs show` 查询。

## 明确非目标

- 不内置商业行情数据。
- 不提供实盘 OMS/EMS、撮合、盘口交易或生产风控执行闭环。
- 不把真实 Tushare 网络请求放进默认 CI。
- 不把 `data/`、`workspace/runs/`、`workspace/factor_evaluations/` 的本地产物提交到仓库。
- 不把 `intraday/` 与 Tick 级研究作为当前主线。
