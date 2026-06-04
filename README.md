# FactorZen

FactorZen 是一个面向 A 股单因子的研究框架，目标是把一个信号从数据、因子计算、预处理、IC 检验、分层回测、walk-forward 样本外验证到 HTML Tear Sheet 报告串成一条可复现的链路。

[![CI](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml/badge.svg)](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
## 适合做什么

- 验证日频单因子是否有稳定预测能力。
- 检查分层收益、Rank/Pearson IC、HAC t 统计、换手、成本和容量约束。
- 生成可审计的实验产物：`manifest.json`、universe 快照、parquet 结果和 Tear Sheet HTML。
- 编写自定义日频/周频/月频因子，并用统一 CLI 跑评估流程。

## 不覆盖什么

FactorZen 不是实盘交易系统，不提供 OMS/EMS、撮合、风控执行闭环，也不内置商业行情数据。`intraday/` 目前保留为分钟线研究代码，主线仍聚焦低频因子评估。

## 核心原则

1. **无未来函数**：T 日信号生成，T+1 开盘执行；前向收益和成交约束按可获得数据对齐。
2. **可复现**：运行配置、git SHA、lockfile hash、输出路径和阶段耗时写入 manifest。
3. **可信结论**：样本不足、覆盖率偏低、缺失模块会在报告中显式标注。
4. **质量门**：lint、mypy、pytest、coverage 由本地命令和 CI 共同守护。

## 安装

推荐使用 [pixi](https://pixi.sh/) 管理环境：

```bash
pixi install
cp .env.example .env
pixi run smoke
```

`.env` 不入库。真实数据拉取需要在 `.env` 中配置 `TUSHARE_TOKEN`；LLM 解读是可选能力，默认关闭。

## 快速开始

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz config validate workspace/configs/daily/daily_factor_template.yaml
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 --universe csi500
pixi run fz report path <run_id>
```

使用 YAML 配置运行：

```bash
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
```

数据拉取示例：

```bash
pixi run fz data fetch daily --start 20230101 --end 20241231
pixi run fz data fetch daily-basic --start 20230101 --end 20241231
```

## 输出在哪里

每次评估会写入：

```text
workspace/factor_evaluations/{run_id}/
  report.html
  manifest.json
  universe.parquet
  *_ic.parquet
  *_backtest.parquet
```

运行日志、实验产物和本地行情数据默认不提交到 Git：

```text
data/                         # 本地行情与缓存
workspace/runs/               # 运行日志和中间产物
workspace/factor_evaluations/ # 每次评估输出
```

## 项目结构

```text
src/factorzen/
  config/               # 路径、常量、Tushare 配置
  core/                 # 日历、universe、存储、加载、数据审计、实验元数据
  daily/                # 日频数据、因子、预处理、评估和优化
  intraday/             # 分钟线研究代码，当前非主线
  llm/                  # 可选的 OpenAI-compatible 研究解读
  pipelines/            # daily_single、generate_report 端到端流程
  reports/              # Tear Sheet 报告引擎和模板
  research/combination/ # 实验性多因子合成
  cli/                  # fz 命令行入口
workspace/
  factors/              # 用户自定义因子
  configs/              # 实验 YAML 配置
tests/                  # pytest 测试
docs/                   # 架构、因子编写、运行手册和路线图
```

## 开发

```bash
pixi run lint
pixi run format
pixi run typecheck
pixi run test
pixi run coverage
```

提交前建议启用：

```bash
pre-commit install
```

## 文档

- [项目说明](docs/project-explanation.md)
- [架构](docs/architecture.md)
- [因子编写](docs/factor-authoring.md)
- [运行手册](docs/runbook.md)
- [演进计划](docs/evolution-plan-2026.md)

## 安全

不要提交 `.env`、API token、商业行情数据或私有研究产物。安全问题请参考 [SECURITY.md](SECURITY.md)。

## 许可

本项目以 [MIT License](LICENSE) 开源。
