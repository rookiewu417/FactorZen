<div align="center">

<img src="docs/assets/logo-horizontal-light.svg" alt="FactorZen logo" width="520">

# FactorZen

**以因子库准入为核心的多市场量化研究平台**

因子挖掘 → 防过拟合护栏 → **增量准入进库** → 风险与组合 → 模拟与向前执行 → 无人值守运营 → 成果展示。<br>
每一步落 `manifest.json`，可审计、可复现。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[快速开始](#快速开始) · [文档](docs/README.md) · [核心机制](docs/concepts/factor-library.md) · [CLI 参考](docs/reference/cli.md) · [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html)

</div>

---

## 这是什么

大多数因子平台回答的是「**这个因子好不好**」。FactorZen 回答的是一个更难、也更有用的问题：

> **这个因子，对我已有的因子库还有没有增量？**

单因子指标漂亮但与在库因子高度重合，是研究里最常见的自欺。FactorZen 把**增量检验（lift）作为入库的最终裁决**——候选因子必须在既有因子库的基础上跑出统计显著的增量，才能进库；单因子门槛降级为排序信号，硬门只剩数据质量。因子库因此是一份持续收敛、互相不冗余的资产，而不是一张越堆越长的候选表。

围绕这个核心，平台提供从数据接入到无人值守运营的完整链路，覆盖 A 股日频、crypto USDT-M 永续（含分钟级）、期货与美股。

---

## 三条设计铁律

冲突时，以下三条是裁决依据。

1. **PIT 无未来函数** —— t 日信号只用 ≤t 收盘可得的信息：universe 逐日快照，财务按公告日对齐，执行定价用 pre_close，滚动因子扩窗预热。停牌/涨跌停/ST/次新/T+1 在口径层就被约束。
   > 已知例外：美股 universe 用的是静态成分快照，存在幸存者偏差，见[适用边界](#适用边界)。
2. **护栏咬合** —— bootstrap IC 置信区间、Deflated Sharpe、PBO/CSCV、holdout 隔离**默认参与筛选**，不是「只算不判」。多重检验从挖掘起就记账。
3. **可复现** —— 每次运行落 `manifest.json`（配置、命令、`git_sha`、seed、窗口、universe、依赖 lock hash）。因子库记录连评估窗口、CV 参数、阈值与基线 hash 一并存档，事后能重跑出同样结果。

---

## 核心能力

| 能力域 | 内容 | 入口命令 |
|---|---|---|
| **数据接入** | A 股（Tushare）· crypto（Binance Vision 数据湖）· 期货（主力连续后复权）· 美股（Yahoo，MVP universe） | `fz data fetch` |
| **日内微观结构** | 分钟 bar → 日频特征面板（17 特征电池），可直接作为挖掘叶子 | `fz data intraday-features build` |
| **因子挖掘** | 算子库 + 表达式 AST 双向编译 + 随机/遗传搜索 | `fz mine search` |
| **LLM 挖掘** | 单 Agent 闭环 · 4 角色团队（Hypothesis/Coder/Critic/Librarian）+ Evaluator + 跨轮否决 + 跨 session 记忆 | `fz mine agent` · `fz mine team` |
| **因子库准入** | 唯一登记簿 + **lift 增量裁决** + 四态状态机 + 向前确认（probation → forward → promote） | `fz factor-library lift-test` |
| **防过拟合** | block bootstrap IC CI · Deflated Sharpe · PBO/CSCV · holdout 隔离 · 空假设校准 | `fz validate overfit` |
| **风险模型** | Barra 风格（8 因子）+ 行业暴露 + Newey-West 协方差 + 特质风险收缩 + MCR 分解（A 股） | `fz risk build` |
| **组合优化与归因** | cvxpy 因子形式 mean-variance QP + 约束体系；Brinson-Fachler + 风险因子归因 | `fz portfolio build` |
| **多因子组合研究** | 四方法样本外对比：等权 / IC 加权 / max_ir / LightGBM | `fz combine from-library` |
| **模拟与向前执行** | 组合权重回测 · 向前执行引擎（纸面撮合）· A 类分歧归因 | `fz sim run` · `fz live step` |
| **无人值守运营** | 8 阶段幂等日链路（守卫→取数→审计→日内特征→信号→执行→报告→发布）+ 失败告警 | `fz ops daily` |
| **成果展示** | 单因子 Tear Sheet · 组合 Dashboard · 只读 REST API + Web 页 | `fz report portfolio` · `pixi run serve` |

单因子研究链路（IC / 分层回测 / walk-forward / Tear Sheet）作为基础能力贯穿其中：`fz factor run`。

---

## 安装

推荐 [pixi](https://pixi.sh/) 管理环境（Python ≥3.10 <3.13）。所有命令从仓库根目录执行。

```bash
pixi install
cp .env.example .env   # 填入 TUSHARE_TOKEN
pixi run smoke
```

- 真实数据拉取需 `TUSHARE_TOKEN`；crypto 走本地数据湖，无需 token。
- LLM 挖掘（`fz mine agent` / `fz mine team`）需配置 `FACTORZEN_LLM_*`，**缺失会直接报错退出**，不会静默跳过。单因子评估与报告不依赖 LLM。
- Web Dashboard 依赖 `fastapi`/`uvicorn`，二者属 **dev extras 而非运行时依赖**；只装运行时依赖起不来 server。

详见[安装与环境](docs/getting-started/installation.md)。

---

## 快速开始

平台的**核心闭环**是「挖掘 → 增量准入 → 组合」。最短路径：

```bash
# 1. 拉数据（Tushare → 本地 parquet 缓存）
pixi run fz data fetch daily --start 20200101 --end 20241231
pixi run fz data fetch daily-basic --start 20200101 --end 20241231

# 2. 挖因子（表达式搜索；或用 fz mine team 走 LLM 团队）
pixi run fz mine search --start 20200101 --end 20231231 \
  --method genetic --trials 200 --top-k 10

# 3. 增量准入 —— 平台的核心一步
#    候选因子必须相对现有因子库跑出显著增量才进库
pixi run fz factor-library lift-test --market ashare      # 默认 dry-run，只看裁决
pixi run fz factor-library lift-test --market ashare --apply   # 确认后才写库

# 4. 查看因子库现状
pixi run fz factor-library list --market ashare

# 5. 用库里的因子做多因子组合（四方法样本外对比）
pixi run fz combine from-library --market ashare \
  --start 20200101 --end 20231231
```

> ⚠️ `lift-test` 与 `forward-review` **默认是 dry-run**，必须显式加 `--apply` 才会写入因子库。这是有意设计：准入是不可逆的库变更。

完整链路（含风险模型、组合优化、模拟交易、报告）见[端到端教程](docs/getting-started/end-to-end-tutorial.md)。

### 单因子评估

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20230101 --end 20241231
pixi run fz report path <run_id>
```

无 `--config` 时使用内置研究级默认配置（`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、walk-forward 默认关闭）。

> ⚠️ 内置默认预设与 `workspace/configs/` 下的 YAML 模板对 `neutralize` 取值不同（预设 `true`，模板 `false`）。见[配置参考](docs/reference/configuration.md)。

`--set key=value` 可在校验前覆盖任意配置字段，可重复，且写入 `manifest.json` 保持可复现：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 --set walk_forward.train_days=252
```

---

## 项目结构

```text
src/factorzen/                  约 49,500 行
├── discovery/      因子挖掘 + 因子库 + lift 准入（最大子包）
├── daily/          A 股日频主干：PIT 数据、预处理、IC、回测、walk-forward
├── core/           日历、universe 快照、Tushare 加载与缓存、叶子 schema 单一真源
├── agents/         LLM 挖掘：单 Agent 闭环 + 4 角色团队 + 实验索引
├── markets/        Ports & Adapters：ashare / crypto / futures / us
├── pipelines/      端到端编排：单因子链路、组合、research run
├── cli/            fz 命令行入口（16 个顶层命令）
├── intraday/       分钟 bar → 日内微观结构特征面板
├── risk/           Barra 风险模型（A 股）
├── research/       多因子组合研究（四方法 OOS 对比）
├── execution/      向前执行引擎（纸面撮合 + 分歧归因）
├── reports/        Tear Sheet + 组合 Dashboard 渲染
├── ops/            无人值守 8 阶段日链路
├── llm/            LLM 客户端（双 profile）
├── validation/     防过拟合统计原语
├── portfolio/      组合优化（因子形式 QP）
├── attribution/    Brinson-Fachler + 风险因子归因
├── server/         只读 REST API + Web Dashboard（dev extras）
└── builtin_factors/ 内置因子（daily/weekly/monthly/intraday/qlib）

workspace/          研究产出（因子库、挖掘 session、评估、组合、报告）
data/               行情数据与缓存（不入库）
tests/              2,561 个 pytest 测试
```

产物布局与 `manifest.json` 字段见[产物参考](docs/reference/artifacts.md)。

---

## 技术栈

- **Python** ≥3.10 <3.13，pixi 环境管理（conda-forge，win-64/linux-64）
- **数值**：polars ≥1.0 / numpy / scipy / pandas
- **统计**：statsmodels
- **ML**：lightgbm / scikit-learn / optuna
- **优化**：cvxpy ≥1.4（CLARABEL solver）
- **数据**：tushare（A 股）/ ccxt（crypto）/ pyarrow
- **LLM**：openai SDK（OpenAI-compatible 网关）
- **报告**：matplotlib / jinja2
- **质量**：2,561 个 pytest 测试 / ruff / mypy（全包扫描）

---

## 适用边界

**适合**

- 在多市场行情上挖掘因子，并用**相对已有因子库的增量**而非孤立指标来决定是否采纳。
- 用防过拟合护栏（bootstrap CI / DSR / PBO / holdout）对候选因子做严格验收。
- 用 Barra 风险模型控制暴露、凸优化建仓、模拟交易评估组合绩效。
- 产出可审计产物：`manifest.json`、universe 快照、parquet 结果、HTML 报告。

**已知限制**（均为当前实现的真实边界，非表述保守）

| 限制 | 说明 |
|---|---|
| **市场覆盖不均** | ashare / crypto 全链路可跑；**futures / us 只通到挖掘与因子库**，没有数据拉取子命令与组合优化接线。 |
| **美股 PIT 打折** | universe 用约 2024 年的静态成分快照（约 490 支），**存在幸存者偏差**，非 PIT 历史成分。用它回看历史窗口需自行承担偏差。 |
| **风险模型仅 A 股** | Barra 模型未接入多市场 Port；crypto 有独立的风险实现，futures / us 无风险模型。 |
| **行业中性是等权基准** | `--industry-neutral` 约束相对**等权**行业基准，不等同市值加权中性。 |
| **归因为两项法** | Brinson-Fachler 两项法，交互项并入选股；不提供 BHB 三项法，不支持日内高频归因。 |
| **组合优化偏薄** | 组合优化与归因是平台当前最轻的能力，相对挖掘与因子库侧的成熟度有明显落差。 |
| **research run 为单因子** | `fz research run` 目前是单因子 + in-sample 编排。 |
| **实盘尚未接入** | 向前执行引擎跑的全部是纸面撮合（`PaperBroker`）；券商接口字段已按 miniQMT 形状预留，但**实盘下单未实现**。这是分阶段推进的路线目标，不是永久非目标。 |
| **向前确认需手动** | `fz factor-library forward-track` 尚未接进无人值守日链路，probation 因子的每日确认目前需手动执行。 |
| **Web 展示为 dev extras** | `server/` 只读、无鉴权、无分页；依赖不在运行时依赖集内。 |

---

## 文档

| 入口 | 内容 |
|---|---|
| [文档索引](docs/README.md) | 全部文档导航 |
| [快速上手](docs/getting-started/quickstart.md) | 5 分钟跑通核心闭环 |
| [端到端教程](docs/getting-started/end-to-end-tutorial.md) | 从拉数据到组合 Dashboard |
| [因子库与准入](docs/concepts/factor-library.md) | lift 裁决、状态机、向前确认 |
| [架构](docs/concepts/architecture.md) | 分层结构、数据流、模块边界 |
| [CLI 参考](docs/reference/cli.md) | 16 个顶层命令 / 47 个叶子命令全量 |

---

## 开发

```bash
pixi run lint        # ruff check
pixi run typecheck   # mypy（全包）
pixi run test        # pytest -n auto
pixi run coverage    # 全量测试 + 覆盖率门槛
```

> ⚠️ **不要运行 `pixi run format`**。全仓 ruff format 会一次改动数百个文件、污染 diff；格式问题请按 lint 报错逐处修。

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 安全

不要提交 `.env`、API token、商业行情数据或私有研究产物。安全策略与凭据轮换见 [SECURITY.md](SECURITY.md)。

---

## 许可

[MIT License](LICENSE)。
