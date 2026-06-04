# FactorZen

> **FactorZen** 是一个面向 A 股单因子的可信研究框架,强调严谨、克制与可复现。核心主线覆盖:因子计算 → 预处理 → IC / 分层回测评估 → walk-forward 样本外验证 → 数据质量报告 → 实验 manifest → Tear Sheet HTML 报告生成。当前聚焦**日频评估与报告**;`research/combination/` 提供实验性多因子合成工具(用于研究对比,不作为生产组合优化模块)。

[![CI](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml/badge.svg)](https://github.com/rookiewu417/FactorZen/actions/workflows/ci.yml)

## 设计原则

1. **无未来函数** —— 信号在 T 日生成,T+1 开盘执行;前向收益用复权价,严格隔离样本外。
2. **可复现** —— 每次运行落盘 universe 快照、manifest(配置 / git SHA / 参数)与产物。
3. **可信结论** —— 样本不足、覆盖率偏低、缺失模块都会在报告中显式标注,不把缺失当作零信号。
4. **质量门守护** —— `lint / typecheck / test / coverage` 进入 CI,关键路径回归有测试。

## 项目结构

```text
src/factorzen/          # 框架代码
  config/               # 配置加载与常量
  core/                 # 日历、universe、存储、数据质量、实验 manifest、日志
  daily/                # 日频主线
    data/               # PIT 数据上下文
    preprocessing/      # 去极值、标准化、中性化
    evaluation/         # IC、分层回测、换手、walk-forward、归因、基准、成本模型
    factors/            # 因子注册与基类
    optimization/       # 组合优化器(研究用)
  intraday/             # 分钟线(已冻结,不在当前迭代范围)
  llm/                  # LLM 研究解读(可选)
  pipelines/            # daily_single、generate_report 端到端流程
  reports/              # Tear Sheet 报告引擎 + Jinja 模板
  research/combination/ # 实验性多因子合成
  cli/                  # 统一 CLI 入口(fz)
workspace/factors/              # 用户日常新增因子
workspace/configs/              # 实验配置(YAML)
workspace/factor_evaluations/   # 每次运行的 report.html / manifest.json / parquet 产物
data/                   # 本地数据缓存(parquet)
tests/                  # pytest 测试(658+ 用例)
docs/                   # 架构、因子编写、运行手册、演进计划
```

## 环境

本仓库使用 **pixi**(conda-forge + PyPI)管理环境,支持 Python 3.10–3.12。

```bash
pixi install                      # 安装依赖
cp .env.example .env              # 配置 TUSHARE_TOKEN 等(.env 不入库)
pixi run smoke                    # 自检:import polars/tushare 正常
```

## 快速开始(统一 CLI)

```bash
pixi run fz factor list                                            # 列出已注册因子
pixi run fz factor new my_alpha --frequency daily                  # 生成因子模板
pixi run fz factor run my_alpha --start 20250101 --end 20260513 \
    --universe csi500                                              # 运行单因子评估
pixi run fz report path <run_id>                                   # 打印报告路径
pixi run fz config validate workspace/configs/my_run.yaml          # 校验运行配置
```

每次 `factor run` 会在 `workspace/factor_evaluations/{run_id}/` 下生成:
`report.html`(Tear Sheet)、`manifest.json`(可复现元数据)、`universe.parquet`、IC / 回测 parquet。

## 报告(Tear Sheet)包含

综合结论与评级评分卡 · 收益表现(分层 / 多空 / 月度) · 预测能力(Rank/Pearson IC、多持有期、样本外分割) · 结构检验(单调性、自相关、因子相关性) · 交易可行性(换手、成本、成交约束) · 风险归因(市值 / 行业 / 市场状态) · walk-forward OOS · 数据质量 · 附录(复现摘要 + 模块状态)。

## 开发与质量门

```bash
pixi run lint        # ruff 检查
pixi run format      # ruff 格式化
pixi run typecheck   # mypy(src/factorzen)
pixi run test        # pytest
pixi run coverage    # 覆盖率
```

提交前建议启用 `pre-commit install`(ruff + ruff-format + mypy)。CI 在 push / PR 到 `master` 时运行上述质量门。

## 范围与边界

- **当前聚焦:** 日频因子评估与报告。
- **已冻结:** 分钟线(intraday)主线与报告 UI、Tick 级研究、实盘 OMS/EMS、生产组合执行闭环。

## 安全

`.env` 与凭据不入库。若发现密钥泄露,请立即轮换;详见 [SECURITY.md](SECURITY.md)。

## 文档

- [架构](docs/architecture.md) · [因子编写](docs/factor-authoring.md) · [运行手册](docs/runbook.md) · [演进计划](docs/evolution-plan-2026.md)
- 升级计划:[docs/superpowers/plans/](docs/superpowers/plans/)

## 许可

本项目以 [MIT License](LICENSE) 开源。
