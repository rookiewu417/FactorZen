# 项目说明

> [FactorZen](../README.md) · [文档](README.md) · **项目说明** · [架构](architecture.md) · [运行手册](runbook.md)

本文面向维护者、研究使用者与自动化代理，说明 FactorZen 的系统事实、运行边界与质量约束。

> 新人上手先读 [README](../README.md)，日常命令查 [runbook](runbook.md)，新增因子查 [factor-authoring](factor-authoring.md)，架构边界查 [architecture](architecture.md)。

当前版本：`0.2.0` · 许可：MIT

## 1. 项目定位

FactorZen 是面向 A 股低频单因子的可信研究框架。目标不是实盘交易，而是把因子从数据、计算、预处理、IC / 分层回测、walk-forward 样本外验证、数据质量、实验 manifest 到 HTML 报告串成可复现链路。

当前主线：

```text
本地数据缓存
  → PIT 数据上下文
  → 因子计算
  → 预处理
  → IC / 分层回测 / walk-forward
  → 数据质量与实验 manifest
  → Tear Sheet HTML 报告
```

明确不覆盖：实盘 OMS/EMS、盘口撮合、逐笔成交、生产组合执行闭环、Tick 数据接入、商业行情数据内置。`intraday/` 保留为分钟线研究代码，但不是当前路线图主线。

## 2. 文档分工

| 文档 | 职责 |
|------|------|
| `README.md` | 对外入口：定位、安装、快速开始、输出与文档链接 |
| `docs/project-explanation.md` | 当前系统事实与维护上下文 |
| `docs/architecture.md` | 目录、职责、数据流与产物边界 |
| `docs/factor-authoring.md` | 因子作者手册 |
| `docs/runbook.md` | 日常命令、报告、数据与故障处理 |
| `docs/evolution-plan-2026.md` | 公开路线图 |
| `docs/release-notes/` | 已发布版本历史，发布后不回写 |

原则：不轻易删除现有文档，更重要的是避免同一细节在多处发散。

## 3. 目录结构

```text
src/factorzen/
  config/         路径、常量、Tushare 配置
  core/           日历、universe、存储、加载、数据质量、配置校验、实验 manifest、计时、日志
  daily/          低频主线：data(PIT)、preprocessing、factors、evaluation、optimization
  intraday/       分钟线研究代码，当前非主线
  llm/            可选 LLM 研究解读
  pipelines/      daily_single、generate_report 端到端流程
  reports/        Tear Sheet 报告引擎和模板
  research/       实验性多因子合成
  cli/            统一 CLI 入口(fz)
workspace/factors/            用户新增因子
workspace/configs/            实验 YAML 配置
workspace/factor_evaluations/ report.html / manifest.json / parquet 产物 + experiment_index.jsonl
workspace/runs/               调度日志和中间产物
data/                         本地数据缓存，不入库
```

## 4. CLI 入口

主入口统一为 `pixi run fz ...`。

| 命令 | 用途 |
|------|------|
| `fz factor list` | 列出已注册因子 |
| `fz factor new <name> --frequency daily` | 在 `workspace/factors/` 生成因子模板 |
| `fz factor run <name>` | 运行单因子评估 |
| `fz report build <name>` | 生成报告 |
| `fz report path <run_id>` | 打印报告路径 |
| `fz data fetch daily` | 拉取日行情缓存 |
| `fz data fetch daily-basic` | 拉取 daily_basic 缓存 |
| `fz config validate <path>` | 校验 YAML 并打印生效配置 |
| `fz runs list` | 查看运行索引 |
| `fz runs show <run_id>` | 查看单次运行 manifest |

兼容入口 `pixi run daily`、`pixi run report`、`fz factor test`、`fz report open` 仍保留，但新增文档与脚本优先使用上表命令。

## 5. 配置体系

| 模块 | 职责 |
|------|------|
| `config/settings.py` | 集中路径与调度默认值，业务代码从这里取路径 |
| `config/constants.py` | 研究常量：交易日数、MAD 参数、IC 最小样本、默认分位数、涨跌停阈值、成本与基准映射 |
| `config/tushare_config.py` | 读取 `.env`，暴露 `TUSHARE_TOKEN` 与 `ensure_token()`；token 首次真正调用时才校验，离线测试不因 import 失败 |
| `core/config_loader.py` | Pydantic v2 校验 YAML 运行配置 |

配置样例在 `workspace/configs/daily/daily_factor_template.yaml`。常用字段包括 `factor`、`universe`、`start`、`end`、`benchmark`、`seed`、`preprocessing`、`backtest`、`walk_forward`、`ic_method`、`event_study` 与 `neutralized_ic`。其中 `walk_forward.enabled` 默认 `false`（样本外 walk-forward 按需开启）。

## 6. 数据流

低频主线：

```text
本地 parquet 缓存(data/)
  → PIT 数据上下文(daily/data)
  → 因子计算(daily/factors) + 预处理(daily/preprocessing)
  → 前向收益 + IC 分析 + 分层回测 + 换手 + walk-forward
  → 报告引擎(reports/tear_sheet)
  → workspace/factor_evaluations/{run_id}/
```

原始数据按 Hive 风格 `year=YYYY/month=MM` 分区落在 `data/raw/`，缓存落在 `data/cache/`。CI 保持离线可重复，不依赖真实 Tushare 网络请求。

## 7. 回测口径

- t 日因子生成目标权重，t+1 开盘执行调仓。
- 旧持仓承担 overnight return，新持仓承担 open-to-close return。
- 前向收益使用复权价。
- `adv_20d` 只取执行日之前最多 20 期平均成交额，用于平方根冲击成本。
- 停牌不成交，涨停不买入，跌停不卖出。
- 支持 `max_participation_rate`、`max_abs_weight`、`max_gross_exposure` 与 `rebalance_threshold`。

策略接口：继承 `Strategy`，实现 `generate_weights(context) -> DataFrame[ts_code, target_weight]`。内置策略包括 `QuantileLongShortStrategy`、`TopNLongOnlyStrategy`、`FactorWeightedStrategy` 与 `OptimizerStrategy`。

## 8. 评估与报告

评估模块覆盖：

- Rank IC、Pearson IC、中性化 IC、多持有期一致性、HAC t 统计。
- 分层回测、分位收益、多空 NAV、月度收益与分位价差。
- 单调性、Rank 自相关、因子相关性、市值/行业/市场状态分层、事件研究、walk-forward（默认关闭，按需开启）。
- 成本模型、容量约束与基准比较。

报告由 `reports/tear_sheet.generate_tear_sheet` 生成，包含评分卡、分析面板、限制说明、复现摘要与模块状态。报告引擎按职责拆为 `_formatting`、`_scoring`、`_charts`、`_strategy`、`_summaries` 与模板。

## 9. 可复现与可观测

`core/experiment.run_experiment` 为每次运行写入 `manifest.json`，记录：

- `run_id`、开始/结束时间、`duration_seconds`。
- 原始命令与生效配置。
- `git_sha`、`git_dirty`、`pixi_lock_sha256`。
- 输出路径与运行期元数据，例如 `stage_timings`。
- `status=success` 或 `status=failure`，失败时记录错误信息。

跨运行索引追加到 `workspace/factor_evaluations/experiment_index.jsonl`。工作树 dirty 时会记录并提示，因为该 run 不能只凭 git SHA 完全复现。

## 10. 质量门

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
```

CI 在 push / PR 到 `main` 或 `master` 时运行同一套检查。提交前如已安装 `pre-commit`，可执行 `pre-commit install` 启用本地钩子。

## 11. 扩展原则

- 新因子写入 `workspace/factors/{daily,weekly,monthly,intraday}/`，不要写进 `src`。
- 框架共享行为、注册中心、评估逻辑与报告逻辑才进入 `src/factorzen/`。
- 样本内多因子合成仍是实验工具，不应标成样本外组合能力。
- 新能力进入主线前必须补测试，并通过质量门。
