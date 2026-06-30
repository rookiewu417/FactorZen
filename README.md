<div align="center">

<img src="docs/assets/logo-horizontal-light.svg" alt="FactorZen logo" width="520">

# FactorZen

**面向 A 股低频因子的可复现研究框架**

把一个信号从本地数据、因子计算、IC 检验、分层回测、walk-forward 样本外验证，<br>
到实验 manifest 与 HTML Tear Sheet 报告，串成一条可审计、可复现的证据链。

[![CI](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml/badge.svg)](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[快速开始](#快速开始) · [文档](docs/README.md) · [架构](docs/architecture.md) · [路线图](docs/evolution-plan-2026.md) · [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html)

</div>

---

## 设计原则

FactorZen 把研究可信度放在展示效果之前。三条原则贯穿数据、评估到报告：

1. **数据正确性优先于结论** —— 收益对齐、停牌、涨跌停、容量约束在口径层面就被约束，错误数据不进入下游。
2. **可复现优先于好看** —— 每次运行落一份 `manifest.json`，记录配置、命令、git SHA、lockfile hash 与阶段耗时。
3. **报告必须暴露问题** —— 样本不足、覆盖率不足、OOS 不成立会被显式标注，而不是用漂亮图表掩盖。

## 核心能力

| 维度 | 能力 |
|------|------|
| 数据 | 本地 parquet 缓存、PIT 数据上下文、离线可重复的数据质量审计 |
| 因子 | 日/周/月频因子基类与注册中心、YAML 配置驱动、qlib Alpha158/Alpha360 接入 |
| 评估 | Rank / Pearson IC、HAC t 统计、分层回测、换手、成本、容量约束、walk-forward 样本外 |
| 挖掘 | 表达式因子挖掘：算子库 + AST↔字符串编译、随机/遗传搜索、贪心去相关，`fz mine` |
| 防过拟合 | 多重检验记账、block bootstrap IC 置信区间、Deflated Sharpe、PBO/CSCV、holdout 永久隔离，`fz validate overfit` |
| 风险 | Barra 多因子风险模型：风格 + 行业暴露、Newey-West 协方差、特质风险收缩、风险预测/分解，`fz risk build` |
| 报告 | HTML Tear Sheet：评分卡、分析面板、限制说明、复现摘要与模块状态 |
| 可复现 | 实验 manifest、跨运行索引、阶段计时、工作树 dirty 提示 |

## 适用边界

**适合**

- 验证日频、周频、月频单因子是否具备稳定预测能力。
- 检查 IC、HAC t 统计、分层收益、换手、成本、容量约束与 walk-forward 样本外表现。
- 产出可审计产物：`manifest.json`、universe 快照、parquet 结果、数据质量摘要与 Tear Sheet HTML。
- 在 `workspace/factors/` 编写自定义因子，用统一 `fz` CLI 跑完整评估流程。

**不覆盖**

FactorZen 不是实盘交易系统：不提供 OMS/EMS、撮合、风控执行闭环，也不内置商业行情数据。`intraday/` 保留为分钟线研究代码，当前主线仍是低频因子评估；Tick 级研究与生产组合执行不纳入本框架。

## 安装

推荐使用 [pixi](https://pixi.sh/) 管理环境。所有命令从仓库根目录执行，并通过 `pixi run` 进入项目环境。

```bash
pixi install
cp .env.example .env
pixi run smoke
```

`.env` 不入库。真实数据拉取需要配置 `TUSHARE_TOKEN`；无 YAML 默认运行会尝试 LLM 研究解读，缺少 `FACTORZEN_LLM_*` 配置时自动跳过。

## 快速开始

**查看与创建因子**

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
```

**无需 YAML 运行新因子**

```bash
pixi run fz factor run my_alpha --start 20230101 --end 20241231
pixi run fz report path <run_id>
```

无 `--config` 时会使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、内置 4 策略套件、both IC、neutralized IC、event study 与 LLM 解读（缺 `FACTORZEN_LLM_*` 配置时自动跳过）。walk-forward 默认关闭，按需用 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启。

**命令行调参，无需改 YAML**

`--set key=value` 在校验前覆盖任意配置字段（含 `preprocessing` / `backtest` / `walk_forward`），可重复，且仍写入 `manifest.json` 保持可复现：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 --set preprocessing.neutralize=true --set walk_forward.train_days=252
```

无 YAML 默认配置下，`--set backtest.top_n=N` 会同步更新默认主策略为 `topn_N`。

`factor sweep` 建在 `--set` 之上：一条命令跑参数网格的笛卡尔积，按指标排序输出对比表并落 CSV：

```bash
pixi run fz factor sweep --config workspace/configs/daily/daily_factor_template.yaml \
  --grid backtest.top_n=30,50,100 --grid preprocessing.normalizer=zscore,rank_normal --sort-by ir
```

**校验并运行 YAML 配置**

```bash
pixi run fz config validate workspace/configs/daily/daily_factor_template.yaml
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
```

**拉取本地研究数据**

```bash
pixi run fz data fetch daily --start 20230101 --end 20241231
pixi run fz data fetch daily-basic --start 20230101 --end 20241231
```

**查看历史运行**

```bash
pixi run fz runs list
pixi run fz runs show <run_id>
```

## 输出

标准评估产物写入单次运行的自包含目录：

```text
workspace/factor_evaluations/{run_id}/
├── report.html         Tear Sheet 报告
├── manifest.json       配置、命令、git SHA、lockfile hash、阶段耗时
├── universe.parquet    universe 快照
├── *_ic.parquet        IC 结果
└── *_backtest.parquet  分层回测结果
```

本地数据、日志与实验产物默认不提交到 Git：

```text
data/                          本地行情与缓存
workspace/runs/                运行日志和中间产物
workspace/factor_evaluations/  每次评估输出
```

## 项目结构

```text
src/factorzen/
├── config/               路径、常量、Tushare 配置
├── core/                 日历、universe、存储、加载、数据审计、实验元数据
├── builtin_factors/      框架自带因子（daily/weekly/monthly/intraday/qlib），随包分发
├── daily/                低频主线：数据、因子、预处理、评估与优化
├── intraday/             分钟线研究代码，当前非主线
├── llm/                  可选 OpenAI-compatible 研究解读
├── pipelines/            daily_single、generate_report 端到端流程
├── reports/              Tear Sheet 报告引擎和模板
├── research/combination/ 实验性多因子合成
└── cli/                  fz 命令行入口
workspace/
├── factors/              你的自定义因子（默认空，与框架自带分离）
└── configs/              实验 YAML 配置
tests/                    pytest 测试
docs/                     架构、运行手册、因子编写与演进计划
```

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

## 文档

| 文档 | 内容 |
|------|------|
| [项目说明](docs/project-explanation.md) | 系统事实、数据流、配置、质量门与边界 |
| [架构](docs/architecture.md) | 框架包、工作区、数据流与产物边界 |
| [因子编写](docs/factor-authoring.md) | 因子放哪里、实现什么接口、如何验证 |
| [运行手册](docs/runbook.md) | 常用命令、报告入口、数据拉取、故障处理 |
| [演进计划](docs/evolution-plan-2026.md) | 公开路线图与非目标 |
| [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html) | 真实 tear sheet 示例 |

文档索引见 [docs/README.md](docs/README.md)。

## 安全

不要提交 `.env`、API token、商业行情数据或私有研究产物。安全策略与凭据轮换流程见 [SECURITY.md](SECURITY.md)。

## 许可

本项目以 [MIT License](LICENSE) 开源。
