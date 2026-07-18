# 端到端教程：从信号到 Dashboard

> [FactorZen](../README.md) · [文档](README.md) · [运行手册](runbook.md) · **端到端教程**

本教程手把手带你跑通 FactorZen 完整研究链路：拉数据 → 挖因子并导出 alpha → 防过拟合验收 → 建风险模型 → 组合优化建仓 → 模拟交易 → 成果展示页。每一步都有预期产物和输出解读。

> 全流程共 **Step 0–6 七个步骤**：Step 0 为一次性的前置数据拉取，核心研究六步是 Step 1–6。

---

## 前置条件

### 1. 安装环境

```bash
# 克隆仓库
git clone https://github.com/your-org/FactorZen.git
cd FactorZen

# 安装依赖（pixi 管理 conda-forge 环境）
pixi install

# 配置 Tushare token
cp .env.example .env
# 编辑 .env，填入：TUSHARE_TOKEN=<your_token>
```

### 2. 环境自检

```bash
pixi run smoke
```

全部 PASSED 即可继续。如有 FAILED，参见 [运行手册 · 常见故障](runbook.md#常见故障)。

---

## Step 0：拉数据

**做什么**：从 Tushare 拉取 A 股日行情和日基础数据，缓存到本地 parquet，供后续所有步骤使用。

```bash
# 拉日行情（OHLCV + 涨跌幅 + 成交额）
pixi run fz data fetch daily --start 20200101 --end 20241231

# 拉日基础数据（市值 / 换手率 / PE / PB 等）
pixi run fz data fetch daily-basic --start 20200101 --end 20241231
```

**产物位置**

```
data/raw/
├── daily/year=YYYY/month=MM/data.parquet         # 按年/月分区的行情 parquet
└── daily_basic/year=YYYY/month=MM/data.parquet   # 按年/月分区的基础数据 parquet
```

**解读输出**

- 命令会逐日打印进度，最终汇报拉取日期数和失败日期（节假日正常跳过）。
- 如遇 Tushare 频率限制，命令内部自动限速重试；无需手动重跑。
- 拉取完成后可用以下命令验证分区完整性：

```bash
pixi run smoke-data --start 20200101 --end 20241231
```

> **数据说明**：所有数据经 PIT（point-in-time）对齐处理，不含未来函数。

---

## Step 1：挖因子并导出 alpha

**做什么**：用遗传算法在内置算子库（时序/截面/算术算子）上自动搜索表达式因子，保留 IC 最高的 top-k 个；或用 LLM Agent 生成并评估假设驱动的因子。挖完再用 `mine export-alpha` 把选中的候选导出成下游建仓需要的 alpha 信号。

### 方案 A：遗传表达式搜索（推荐入门）

```bash
pixi run fz mine search \
  --start 20200101 --end 20231231 \
  --method genetic \
  --trials 200 \
  --top-k 10 \
  --seed 42
```

运行时间取决于机器性能与 trials 数。途中可实时看到搜索进度。

### 方案 B：LLM 单 Agent 挖掘（需配置 LLM 环境变量）

```bash
pixi run fz mine agent \
  --start 20200101 --end 20231231 \
  --iterations 10
```

### 方案 C：多 Agent 团队挖掘（需 LLM 配置）

```bash
pixi run fz mine team --start 20200101 --end 20231231
```

**查看排行榜**

```bash
pixi run fz mine leaderboard workspace/mining_sessions/session_42_genetic
```

**产物位置**（以方案 A 为例，session 目录名为 `session_{seed}_{method}`）

```
workspace/mining_sessions/session_42_genetic/
├── candidates.csv    # 候选排行榜（列 rank,n_trials,expression,ic_train,...）
└── manifest.json     # 参数 / seed / 复现说明
```

> 入库候选：`fz factor-library list` 查 name 后 `fz factor run <name> --set preprocessing.neutralize=false`；未入库候选表达式在 `candidates.csv`。

> LLM 方案 B/C 的产物分别落在 `workspace/mine_agent/<run_id>/`、`workspace/mine_team/<run_id>/`，文件结构同上。

**解读输出**

- `candidates.csv` 按挖掘内 IC 降序；关注 `ic_train` 较高且表达式简洁的候选。
- 注意 `candidates.csv` 的 IC 是挖掘内估计（plain zscore，无中性化），与 `fz factor run` 默认带中性化的口径不同。

### 导出 alpha 信号

把排行榜里选中的候选（如 rank 1）在建仓信号日当天的截面 α 导出成 `(ts_code, alpha)` 两列 parquet，供 Step 4 组合建仓使用。`--date` 应与 Step 4 的 `--end` 对齐：

```bash
pixi run fz mine export-alpha \
  --session workspace/mining_sessions/session_42_genetic \
  --rank 1 \
  --date 20231231 \
  --universe all_a \
  --out alpha.parquet
```

**产物**：`alpha.parquet`（两列 `ts_code, alpha` 的单截面长表）。

---

## Step 2：防过拟合验收

**做什么**：对候选因子执行 Deflated Sharpe（DSR）+ block bootstrap IC 置信区间评估，结果只打印到终端、不落盘。单因子样本数 N=1，**不计算 PBO**（PBO/CSCV 适用于一池候选因子的多重检验，不适用于单因子）。

> 验收对象必须是**已注册因子名**。入库候选由 library provider 注入 registry（`fz factor-library list` 查 name）；未入库则表达式仍在 session `candidates.csv`。

```bash
# 替换 <factor_name> 为已注册因子名
pixi run fz validate overfit <factor_name> --start 20200101 --end 20241231
```

**输出**：终端打印一行 `IC / IR / DSR p 值 / bootstrap IC 95% CI`，不产生任何文件。

**解读输出**

| 指标 | 参考 | 说明 |
|------|------|------|
| DSR p 值 | 越小越好 | Deflated Sharpe 的显著性 p 值 |
| bootstrap IC 95% CI | 下界 > 0 | IC 均值显著异于零 |

DSR 显著且 bootstrap IC CI 下界 > 0 的因子更稳健；否则回到 Step 1 重新搜索或调整算子。

---

## Step 3：建风险模型

**做什么**：构建 Barra 风格因子风险模型——计算 8 个风格因子暴露（`size / value / momentum / volatility / liquidity / quality / growth / leverage`）和行业因子暴露，用 Newey-West 估计因子协方差矩阵，并对特质风险做收缩。

> 这一步是**独立的风险模型诊断步骤**，用于落盘并检视风险模型产物。Step 4 的 `fz portfolio build` 会**在内部用同段数据现算风险模型**，并不消费这里的产物——所以即使跳过 Step 3，建仓仍可正常进行。

```bash
pixi run fz risk build \
  --start 20200101 --end 20241231 \
  --cov-half-life 90 \
  --nw-lags 2 \
  --spec-half-life 90 \
  --spec-shrinkage 0.3
```

**产物位置**

```
workspace/risk_models/<run_id>/
├── exposures.parquet          # 风格+行业因子暴露矩阵（股票×因子）
├── factor_covariance.parquet  # 因子协方差矩阵（Newey-West 估计）
├── specific_risk.parquet      # 个股特质风险向量（收缩后）
├── factor_returns.parquet     # 因子收益序列
├── risk_summary.csv           # 风险摘要
└── manifest.json
```

**解读输出**

- 命令打印因子数与回归 R²，用于判断风险模型解释度。
- `specific_risk` 越小说明系统性因子解释占比越高。

---

## Step 4：组合优化建仓

**做什么**：以 Step 1 导出的 `alpha.parquet` 为输入，在 `--end` 当日做**单截面**建仓——用 cvxpy（CLARABEL solver）求解一次 mean-variance 二次规划，生成一组目标权重，并输出归因与风险摘要。风险模型由命令内部从同段数据现算。

```bash
pixi run fz portfolio build \
  --start 20200101 --end 20231231 \
  --alpha-file alpha.parquet \
  --lam 1.0 \
  --w-max 0.05 \
  --turnover 0.3 \
  --industry-neutral
```

> `--end` 是建仓信号日（`signal_date = end`），应与 Step 1 `export-alpha` 的 `--date` 一致。只解一次 QP，输出的是单截面权重。

**产物位置**

```
workspace/portfolios/<run_id>/
├── weights.parquet     # 单截面权重（列 ts_code / target_weight / prev_weight）
├── attribution.csv     # Brinson 归因
├── risk_summary.csv    # 风险因子归因摘要
└── manifest.json       # 含 signal_date（供 Step 5 串接持仓）
```

**解读输出**

- 命令打印一行 `status=... holdings=...`：`optimal` 为正常，`infeasible` 说明约束冲突（见下方 MVP 限制）。
- `attribution.csv`：选股 / 行业配置两项效应（单期 Brinson-Fachler 两项法，交互项并入选股；相对股票池等权基准）。
- `risk_summary.csv`：各 Barra 风格因子和行业因子对组合风险的贡献（MCR 分解）。

> **MVP 限制**：
> - `--industry-neutral` 行业中性约束使用行业**等权**为基准，不是市值加权基准。
> - 收益归因（Brinson）基于权重×收益近似，精确归因需持仓期收益（暂不支持多持仓期精细核算）。

---

## Step 5：模拟交易

**做什么**：把 `--portfolio-dir` 根目录下各 `<run_id>/weights.parquet`（按 manifest 的 `signal_date` 串成持仓序列）执行多周期净值回测，对齐真实行情，扣除换手成本（内部 `CostModel`）后输出净值曲线、年化收益、夏普比率、最大回撤等。

> `--portfolio-dir` 传的是组合产物**根目录** `workspace/portfolios`（其下每个 `<run_id>/` 含 `weights.parquet` + `manifest.json`），不是单个 run 目录。

```bash
pixi run fz sim run \
  --portfolio-dir workspace/portfolios \
  --start 20200101 \
  --end 20241231
```

```bash
# 快速查看绩效摘要
pixi run fz sim show --sim-dir workspace/sim/<sim_id>
```

**产物位置**

```
workspace/sim/<sim_id>/
├── nav.parquet     # 净值序列
├── metrics.json    # 年化收益 / 夏普 / 最大回撤 / 年化换手 / 总成本
└── manifest.json
```

**解读输出**

| 指标 | 参考范围 | 说明 |
|------|----------|------|
| 年化收益（超额） | > 5% | 相对 CSI 500 基准的超额年化收益 |
| 夏普比率 | > 1.0 | 净值/波动率比 |
| 最大回撤 | < 20% | 净值历史最大峰谷跌幅 |
| 年化换手 | < 换手上限×年化次数 | 应与 `--turnover` 约束一致 |

> **MVP 限制**：模拟交易不接真实 OMS/经纪商，不做实盘下单。换手成本由内部 `CostModel` 线性计算（默认单边佣金万 2.5 + 滑点万 5，卖出加千 1 印花税），无 CLI 成本入口，未建模非线性市场冲击。

---

## Step 6：成果展示页

**做什么**：整合模拟结果和归因报告，生成单页 HTML Dashboard——指标卡 + 净值曲线 + 月度收益热图 + 因子归因 + 风险摘要。可直接分享或在浏览器查看。

```bash
pixi run fz report portfolio \
  --sim-dir workspace/sim/<sim_id> \
  --portfolio-dir workspace/portfolios/<run_id>
```

**产物位置**

```
workspace/reports/portfolio_<sim_id>.html
```

> 输出文件名里的 `<sim_id>` 取自 `--sim-dir` 目录名；也可用 `--out` 指定路径。

用浏览器打开即可查看。Dashboard 包含：

- **指标卡**：年化收益、夏普比率、最大回撤、年化换手（与基准对比）。
- **净值曲线**：策略净值 vs. 基准净值，含回撤区间标注。
- **月度收益热图**：按年月矩阵展示超额收益冷热分布。
- **归因摘要**：风格因子 / 行业 / 选股效应的收益贡献饼图。
- **风险摘要**：各因子 MCR 贡献柱状图。

```bash
# 也可单独重建因子 Tear Sheet
pixi run fz report build <factor_name> \
  --start 20230101 --end 20241231 --universe csi500
```

---

## 完整命令链（一键复制）

```bash
# Step 0：拉数据
pixi run fz data fetch daily --start 20200101 --end 20241231
pixi run fz data fetch daily-basic --start 20200101 --end 20241231

# Step 1：挖因子（遗传搜索）+ 导出 alpha
pixi run fz mine search \
  --start 20200101 --end 20231231 \
  --method genetic --trials 200 --top-k 10 --seed 42
pixi run fz mine export-alpha \
  --session workspace/mining_sessions/session_42_genetic \
  --rank 1 --date 20231231 --universe all_a --out alpha.parquet

# Step 2：防过拟合验收（替换 <factor> 为已注册因子名）
pixi run fz validate overfit <factor> --start 20200101 --end 20241231

# Step 3：建风险模型（独立诊断步骤，可选）
pixi run fz risk build --start 20200101 --end 20231231

# Step 4：组合优化建仓（--end 与 export-alpha 的 --date 对齐）
pixi run fz portfolio build \
  --start 20200101 --end 20231231 \
  --alpha-file alpha.parquet \
  --industry-neutral

# Step 5：模拟交易（--portfolio-dir 传根目录）
pixi run fz sim run \
  --portfolio-dir workspace/portfolios \
  --start 20200101 --end 20241231

# Step 6：成果展示页
pixi run fz report portfolio \
  --sim-dir workspace/sim/<sim_id> \
  --portfolio-dir workspace/portfolios/<run_id>
```

---

## 全景：从信号到 Dashboard

| 步骤 | 命令 | 产物 |
|------|------|------|
| Step 0 拉数据 | `fz data fetch daily / daily-basic` | `data/raw/{daily,daily_basic}/year=YYYY/month=MM/data.parquet`（parquet 缓存，PIT 对齐） |
| Step 1 挖因子 | `fz mine search / agent / team` + `fz mine export-alpha` | `workspace/mining_sessions/session_{seed}_{method}/`（candidates.csv + manifest）→ `alpha.parquet`；入库后 `fz factor run <name>` |
| Step 2 防过拟合 | `fz validate overfit` | 终端打印 IC / IR / DSR p / bootstrap CI（不落盘） |
| Step 3 风险模型 | `fz risk build`（独立诊断，可选） | `workspace/risk_models/<run>/`（exposures / covariance / specific_risk） |
| Step 4 组合建仓 | `fz portfolio build`（内部现算风险模型） | `workspace/portfolios/<run>/`（weights.parquet + attribution.csv + risk_summary.csv） |
| Step 5 模拟交易 | `fz sim run` | `workspace/sim/<id>/`（nav.parquet + metrics.json） |
| Step 6 成果展示 | `fz report portfolio` | `workspace/reports/portfolio_<id>.html` |

> 每个 run 都产出 `manifest.json`（seed + 参数 + git_sha），可审计、可复现。

---

## MVP 限制汇总（诚实声明）

| 限制项 | 当前状态 | 计划 |
|--------|----------|------|
| 行业中性基准 | 行业等权基准（不是市值加权） | 后续版本升级为市值加权基准 |
| 收益归因精度 | 按权重近似（不支持持仓期内多次调仓的精细核算） | 需补持仓期收益接口 |
| 换手成本模型 | 线性近似（固定费率：佣金+滑点+印花税），未建模市场冲击和流动性折损 | 可扩展为非线性冲击模型 |
| 实盘接入 | 不支持，无 OMS/经纪商接口，不做实盘下单 | 不在路线图内 |
| LLM Agent | 需外部 LLM 配置（`FACTORZEN_LLM_*`），非必须 | 核心链路无 LLM 依赖 |
| 数据源 | 仅 Tushare（A 股），无港股/美股/期货 | 可通过数据插件扩展 |

---

## 下一步

- 调参：`fz factor sweep` 批量扫描参数空间 → 见 [运行手册](runbook.md#factor--单因子研究)
- 深度定制：编写自定义因子 → `fz factor new <name>` 后编辑 `compute()` 方法
- 查看历史：`fz runs list` / `fz runs show <run_id>`
- 完整命令参考：[运行手册](runbook.md)
- 架构设计：[架构文档](architecture.md)
