# 项目说明

> [FactorZen](../README.md) · [文档](README.md) · **项目说明** · [架构](architecture.md) · [运行手册](runbook.md)

本文面向维护者、研究使用者与自动化代理，说明 FactorZen 的系统事实、运行边界与质量约束。

> 新人上手先读 [README](../README.md)，日常命令查 [runbook](runbook.md)，新增因子查 [factor-authoring](factor-authoring.md)，架构边界查 [architecture](architecture.md)。

当前版本：`0.3.0` · 许可：MIT

## 1. 项目定位

FactorZen 是**端到端、可复现的 A 股量化研究平台**。从 v0.3.0 起，项目从「单因子 IC 检验框架」扩展为完整买方研究链路：

```text
本地数据缓存（Tushare → parquet）
  → PIT 数据上下文 + 微观结构约束（M0）
  → 因子计算 + 预处理
  → IC / 分层回测 / walk-forward / Tear Sheet（基线）
  → 因子挖掘（算子库 + 遗传搜索 / LLM Agent / 多 Agent 团队）（M1/M5/M6）
  → 防过拟合护栏（holdout 永久隔离 / DSR / bootstrap CI / PBO）（M2）
  → Barra 风险模型（风格 + 行业暴露 + 协方差 + MCR）（M3）
  → 组合优化建仓（mean-variance QP + 行业中性 + 换手约束）+ 归因（M4）
  → 模拟交易（多周期净值 + 绩效指标）（M7）
  → 成果展示（组合绩效 HTML Dashboard）（M7）
```

贯穿的设计三原则：
1. **数据 PIT**：无未来函数，所有信号在信号日之前信息闭合。
2. **OOS 护栏**：holdout 段永久隔离，Deflated Sharpe 修正 selection bias，PBO 估过拟合概率。
3. **产物落 manifest**：每次运行记录 seed/参数/git_sha，可审计可复现。

明确不覆盖：实盘 OMS/EMS、盘口撮合、逐笔成交、生产组合执行闭环、商业行情数据内置。`intraday/` 保留为分钟线研究代码，不是当前路线图主线。

## 2. 文档分工

| 文档 | 职责 |
|------|------|
| `README.md` | 对外入口：定位、安装、快速开始、输出与文档链接 |
| `docs/project-explanation.md` | 当前系统事实与维护上下文（本文） |
| `docs/architecture.md` | 目录、职责、数据流与产物边界 |
| `docs/factor-authoring.md` | 因子作者手册 |
| `docs/runbook.md` | 日常命令、报告、数据与故障处理 |
| `docs/evolution-plan-2026.md` | 公开路线图（M0-M7 全部 ✅） |
| `docs/release-notes/` | 已发布版本历史，发布后不回写 |

原则：不轻易删除现有文档，避免同一细节在多处发散。

## 3. 目录结构

```text
src/factorzen/
  config/         路径、常量、Tushare 配置
  core/           日历、universe 快照、存储、加载、数据质量、配置校验、实验 manifest、计时、日志
                  + benchmark.py（M0 基准管理）
  daily/          低频主线：data(PIT)、preprocessing、factors、evaluation（含 M0 回测约束）、optimization
  intraday/       分钟线研究代码，当前非主线
  llm/            可选 LLM 研究解读
  pipelines/      daily_single、generate_report 端到端流程
  reports/        Tear Sheet 报告引擎 + portfolio_report（M7 Dashboard）
  research/       实验性多因子合成（样本内工具）
  discovery/      M1 因子挖掘：operators / expression(AST) / search / dedup
  validation/     M2 防过拟合：bootstrap / deflated_sharpe / pbo
  risk/           M3 风险模型：factor_model / covariance / specific_risk / mcr
  portfolio/      M4 组合优化：optimizer（QP + 约束）
  attribution/    M4 归因：brinson / risk_attribution
  agents/         M5/M6 LLM Agent：agent / roles / team_orchestrator / experiment_index
  sim/            M7 模拟交易：runner / metrics
  cli/            统一 CLI 入口(fz)

workspace/
  factors/            用户新增因子
  configs/            实验 YAML 配置
  factor_evaluations/ report.html / manifest.json / parquet 产物 + experiment_index.jsonl
  risk_models/        Barra 风险模型产物（M3）
  portfolios/         组合优化建仓产物（M4）
  sim/                模拟交易净值与绩效（M7）
  reports/            组合 Dashboard HTML（M7）
  runs/               调度日志和中间产物

data/                 本地数据缓存，不入库
```

## 4. CLI 入口

主入口统一为 `pixi run fz ...`。

**单因子研究（基线）**

| 命令 | 用途 |
|------|------|
| `fz factor list` | 列出已注册因子 |
| `fz factor new <name> --frequency daily` | 在 `workspace/factors/` 生成因子模板 |
| `fz factor run <name> [--start --end --universe --config]` | 运行单因子完整评估 |
| `fz factor sweep <name> --grid K=V1,V2` | 参数网格扫描 |
| `fz report build <name>` | 生成 Tear Sheet |
| `fz report path <run_id>` | 打印报告路径 |
| `fz data fetch daily --start --end` | 拉取日行情缓存 |
| `fz data fetch daily-basic --start --end` | 拉取 daily_basic 缓存 |
| `fz config validate <path>` | 校验 YAML 并打印生效配置 |
| `fz runs list` | 查看运行索引 |
| `fz runs show <run_id>` | 查看单次运行 manifest |

**因子挖掘（M1/M5/M6）**

| 命令 | 用途 |
|------|------|
| `fz mine search --start --end [--method random/genetic --trials --top-k --seed]` | 表达式随机/遗传搜索 |
| `fz mine leaderboard <session_dir>` | 打印搜索排行榜 |
| `fz mine agent --start --end [--iterations]` | LLM 单 Agent 挖掘 |
| `fz mine team --start --end` | 多 Agent 团队挖掘 |

**防过拟合 / 风险 / 组合 / 模拟（M2-M7）**

| 命令 | 用途 |
|------|------|
| `fz validate overfit <factor> --start --end` | Deflated Sharpe + bootstrap IC CI + PBO |
| `fz risk build --start --end [--cov-half-life --nw-lags --spec-shrinkage]` | 构建 Barra 风险模型 |
| `fz portfolio build --start --end --alpha-file <parquet> [--lam --w-max --turnover --industry-neutral]` | 凸优化建仓 + 归因 |
| `fz sim run --portfolio-dir <dir> --start --end` | 多周期净值回测 |
| `fz sim show --sim-dir <dir>` | 打印绩效摘要 |
| `fz report portfolio --sim-dir --portfolio-dir [--out]` | 组合绩效 HTML Dashboard |

兼容入口 `pixi run daily`、`pixi run report` 仍保留，但新增文档优先使用上表命令。

## 5. 配置体系

| 模块 | 职责 |
|------|------|
| `config/settings.py` | 集中路径与调度默认值 |
| `config/constants.py` | 研究常量：交易日数、MAD 参数、IC 最小样本、默认分位数、涨跌停阈值、成本与基准映射 |
| `config/tushare_config.py` | 读取 `.env`，暴露 `TUSHARE_TOKEN` 与 `ensure_token()`；离线测试不因 import 失败 |
| `core/config_loader.py` | Pydantic v2 校验 YAML 运行配置 |

配置样例在 `workspace/configs/daily/daily_factor_template.yaml`。常用字段包括 `factor`、`universe`、`start`、`end`、`benchmark`、`seed`、`preprocessing`、`backtest`、`walk_forward`、`ic_method`、`event_study` 与 `neutralized_ic`。

## 6. 数据流

**低频单因子主线（基线）**

```text
本地 parquet 缓存(data/)
  → PIT 数据上下文(daily/data) + universe 快照(core/universe)
  → 因子计算(daily/factors) + 预处理(daily/preprocessing)
  → 前向收益 + IC 分析 + 分层回测 + 换手 + walk-forward
  → 报告引擎(reports/tear_sheet)
  → workspace/factor_evaluations/{run_id}/
```

**M0 微观结构约束**

- Universe 快照：停牌/涨跌停/ST/次新股过滤，t 日信号对应 t+1 可交易标的。
- GEM 双路径容差：open → vwap 回落 ≤1% 视为可成交，减少涨停漏成交。
- `signal_date` 显式字段：因子信号日与执行日解耦，避免前视偏差。

**M1-M7 扩展链路**

```text
discovery/(M1)  → 候选因子表达式集合
validation/(M2) → holdout 段 IC CI / DSR / PBO 护栏
risk/(M3)       → Barra 因子暴露矩阵 + 协方差 + 特质风险
portfolio/(M4)  → 目标权重（QP 优化）→ workspace/portfolios/{run_id}/
attribution/(M4)→ Brinson 归因 + 风险因子归因
agents/(M5/M6)  → LLM Agent 候选因子 → 进入 discovery/ 链路
sim/(M7)        → 净值序列 + 绩效指标 → workspace/sim/{run_id}/
reports/(M7)    → 组合绩效 HTML Dashboard
```

原始数据按 Hive 风格 `year=YYYY/month=MM` 分区落在 `data/raw/`，缓存落在 `data/cache/`。CI 保持离线可重复，不依赖真实 Tushare 网络请求。

## 7. 回测口径

**单因子回测**

- t 日因子生成目标权重，t+1 开盘执行调仓。
- 旧持仓承担 overnight return，新持仓承担 open-to-close return。
- 前向收益使用复权价。
- `adv_20d` 只取执行日之前最多 20 期平均成交额，用于平方根冲击成本。
- 停牌不成交，涨停不买入，跌停不卖出。

**组合模拟回测（M7）**

- 读取 `workspace/portfolios/{run_id}` 的目标权重，按日对齐行情，扣换手成本。
- 输出净值序列、年化收益、夏普、最大回撤、卡尔玛、换手率、信息比率。
- 不模拟盘口深度、部分成交或实盘滑点。

策略接口：继承 `Strategy`，实现 `generate_weights(context) -> DataFrame[ts_code, target_weight]`。内置策略：`QuantileLongShortStrategy`、`TopNLongOnlyStrategy`、`FactorWeightedStrategy`、`OptimizerStrategy`。

## 8. 评估与报告

**单因子评估（基线 + M0）**

- Rank IC、Pearson IC、中性化 IC、多持有期一致性、HAC t 统计。
- 分层回测、分位收益、多空 NAV、月度收益与分位价差。
- 单调性、Rank 自相关、因子相关性、市值/行业/市场状态分层、事件研究、walk-forward。
- 成本模型、容量约束与基准比较（等权基准，指数基准接口预留）。

**防过拟合护栏（M2）**

- block bootstrap IC 置信区间（HAC 相容）。
- Deflated Sharpe Ratio（DSR）：修正多次独立尝试的 selection bias。
- PBO/CSCV：估计回测过拟合概率。

**组合归因与报告（M4/M7）**

- Brinson 多期归因（配置 / 选股 / 相互作用效果分解）。
- 风险因子归因（持仓风格暴露 × 因子收益）。
- 组合绩效 HTML Dashboard：指标卡 + 净值曲线 + 月度热图 + 归因条形图 + 风险摘要。

报告引擎：`reports/tear_sheet.generate_tear_sheet`（单因子）、`reports/portfolio_report`（组合 Dashboard）。Tear Sheet 按职责拆为 `_formatting`/`_scoring`/`_charts`/`_strategy`/`_summaries`。

## 9. 可复现与可观测

`core/experiment.run_experiment` 为每次运行写入 `manifest.json`，记录：

- `run_id`、开始/结束时间、`duration_seconds`。
- 原始命令与生效配置。
- `git_sha`、`git_dirty`、`pixi_lock_sha256`。
- 输出路径与运行期元数据（`stage_timings`）。
- `status=success` 或 `status=failure`，失败时记录错误信息。

跨运行索引追加到 `workspace/factor_evaluations/experiment_index.jsonl`（M6 Agent 同步写入 `agents/experiment_index.py` 的长期记忆索引）。工作树 dirty 时记录并提示。

## 10. 质量门

```bash
pixi run lint
pixi run typecheck
pixi run test        # 1109 用例（v0.3.0），全部离线可重复
pixi run coverage    # 门槛 ≥70%
```

CI 在 push / PR 到 `main` 或 `master` 时运行同一套检查。提交前如已安装 `pre-commit`，可执行 `pre-commit install` 启用本地钩子。

## 11. 扩展原则

- 新因子写入 `workspace/factors/{daily,weekly,monthly,intraday}/`，不要写进 `src`。
- 框架共享行为、注册中心、评估逻辑与报告逻辑才进入 `src/factorzen/`。
- 样本内多因子合成（`research/combination/`）仍是实验工具，不应标成 OOS 组合能力。
- 新能力进入主线前必须补测试，并通过质量门。
- LLM Agent（M5/M6）依赖外部 API，不内置模型；无 API key 时 Agent 命令不可用，其余功能不受影响。

## 12. MVP 限制诚实说明

| 功能 | 限制 |
|------|------|
| 行业中性约束（M4） | 相对等权基准，非真实指数成分权重（`fetch_index_weights` 接口已预留） |
| 跟踪误差约束 | 未内置 TEV，通过换手约束近似控制主动风险 |
| 模拟交易（M7） | 不模拟盘口深度/部分成交/实盘滑点；不接实盘 OMS |
| LLM Agent（M5/M6） | 依赖外部 LLM API key；无 key 则不可用 |
| 遗传搜索（M1） | 单进程；大规模并行需外部 tmux 多进程 |
