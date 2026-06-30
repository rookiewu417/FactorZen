# 端到端教程：从信号到 Dashboard 六步走

> [FactorZen](../README.md) · [文档](README.md) · [运行手册](runbook.md) · **端到端教程**

本教程手把手带你跑通 FactorZen 完整研究链路：拉数据 → 挖因子 → 防过拟合验收 → 建风险模型 → 组合优化建仓 → 模拟交易 → 成果展示页。每一步都有预期产物和输出解读。

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
data/tushare/
├── daily/               # 按日期分区的行情 parquet
└── daily_basic/         # 按日期分区的基础数据 parquet
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

## Step 1：挖因子

**做什么**：用遗传算法在内置算子库（时序/截面/算术算子）上自动搜索表达式因子，保留 IC 最高的 top-k 个；或用 LLM Agent 生成并评估假设驱动的因子。

### 方案 A：遗传表达式搜索（推荐入门）

```bash
pixi run fz mine search \
  --start 20200101 --end 20231231 \
  --method genetic \
  --trials 200 \
  --top-k 10 \
  --seed 42
```

运行约 5-20 分钟（取决于机器性能）。途中可实时看到每轮最优 IC。

### 方案 B：LLM 单 Agent 挖掘（需配置 LLM 环境变量）

```bash
pixi run fz mine agent \
  --start 20200101 --end 20231231 \
  --iterations 10
```

### 方案 C：多 Agent 团队挖掘（M6，需 LLM 配置）

```bash
pixi run fz mine team --start 20200101 --end 20231231
```

**查看排行榜**

```bash
pixi run fz mine leaderboard workspace/discovery/<session_id>
```

**产物位置**

```
workspace/discovery/<session_id>/
├── top10_expressions.json    # top-k 表达式字符串 + IC/IR 统计
├── alpha.parquet             # top 因子的每日截面 alpha 值
├── correlation_matrix.csv    # top-k 因子去相关矩阵
└── manifest.json             # 参数 / seed / git_sha（可复现）
```

**解读输出**

- 排行榜按 IC_mean 降序；关注 `IC_mean > 0.03`、`IC_IR > 0.5` 的候选因子。
- 去相关矩阵：相关性 > 0.7 的因子对视为冗余，只保留 IC 较高的一个。
- `session_id` 格式为 `YYYYMMDD_HHMMSS`，后续步骤需要用到。

---

## Step 2：防过拟合验收

**做什么**：对候选因子执行 Deflated Sharpe（DSR）、block bootstrap IC 置信区间、PBO/CSCV 过拟合概率评估。holdout 段（通常是最后 20% 的时间）永久隔离，从不参与训练，只在此处使用一次。

```bash
# 对 top 因子逐个验收（替换 <factor_name> 为排行榜中的因子名或表达式字符串）
pixi run fz validate overfit <factor_name> --start 20200101 --end 20241231
```

**产物位置**

```
workspace/validation/<run_id>/
├── dsr_report.json           # Deflated Sharpe 统计（DSR、t-stat、p-value）
├── bootstrap_ic_ci.png       # bootstrap IC 均值分布图（含 95% CI）
├── pbo_matrix.csv            # CSCV 组合内外胜率矩阵
└── manifest.json
```

**解读输出**

| 指标 | 通过标准 | 说明 |
|------|----------|------|
| DSR | > 0 | Deflated Sharpe > 0，经多重比较修正后仍显著 |
| bootstrap IC 95% CI | 下界 > 0 | IC 均值在 holdout 段显著异于零 |
| PBO | < 0.5 | 过拟合概率低于 50% |

三项均通过的因子才进入后续链路；否则回到 Step 1 重新搜索或调整算子。

---

## Step 3：建风险模型

**做什么**：构建 Barra 风格因子风险模型——计算 8 个风格因子暴露（规模/价值/动量/波动率/成长/杠杆/流动性/非线性规模）和行业因子暴露，用 Newey-West 估计因子协方差矩阵，并对特质风险收缩压缩。

```bash
pixi run fz risk build \
  --start 20200101 --end 20241231 \
  --cov-half-life 63 \
  --nw-lags 5 \
  --spec-shrinkage 0.1
```

运行约 2-5 分钟（全量数据）。

**产物位置**

```
workspace/risk_models/<run_id>/
├── factor_exposures.parquet  # 每日风格+行业因子暴露矩阵（股票×因子）
├── factor_cov.parquet        # 因子协方差矩阵（Newey-West 估计）
├── specific_risk.parquet     # 个股特质风险向量（收缩后）
└── manifest.json
```

**解读输出**

- 命令输出每个风格因子的 IC 分布摘要，用于判断因子暴露质量。
- `factor_cov` 应为正定矩阵（命令内部自动校验，若不正定会警告并强制谱剪裁）。
- `specific_risk` 越小说明风险模型解释度越高。

---

## Step 4：组合优化建仓

**做什么**：以 Step 1/2 产出的 alpha 信号和 Step 3 的风险模型为输入，用 cvxpy（CLARABEL solver）求解 mean-variance 二次规划，生成每日目标权重，并输出 Brinson 归因和风险因子归因。

```bash
pixi run fz portfolio build \
  --start 20200101 --end 20241231 \
  --alpha-file workspace/discovery/<session_id>/alpha.parquet \
  --lam 1.0 \
  --w-max 0.05 \
  --turnover 0.3 \
  --industry-neutral
```

运行约 5-15 分钟（每个交易日一次 QP 求解）。

**产物位置**

```
workspace/portfolios/<run_id>/
├── weights.parquet           # 每日股票权重矩阵
├── attribution_report.html   # Brinson 归因 + 风险因子归因 HTML
└── manifest.json
```

**解读输出**

- 优化日志输出每日求解状态：`optimal` 为正常，`infeasible` 说明约束冲突（见下方 MVP 限制）。
- 归因报告分为两部分：
  - **Brinson 归因**：选股效应 / 行业配置效应 / 交叉效应（相对等权基准）。
  - **风险因子归因**：各 Barra 风格因子和行业因子对组合波动率的贡献比例（MCR 分解）。

> **MVP 限制**：
> - `--industry-neutral` 行业中性约束使用行业**等权**为基准，不是市值加权基准。
> - 收益归因（Brinson）基于每日权重×日收益近似，精确归因需持仓期收益（暂不支持多持仓期精细核算）。

---

## Step 5：模拟交易

**做什么**：对 Step 4 产出的每日目标权重执行多周期净值回测，对齐真实行情（停牌/涨跌停过滤），扣除换手成本后输出净值曲线、年化收益、夏普比率、最大回撤等。

```bash
pixi run fz sim run \
  --portfolio-dir workspace/portfolios/<run_id> \
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
├── nav.parquet               # 每日净值序列
├── performance.json          # 年化收益 / 夏普 / 最大回撤 / 年化换手
└── manifest.json
```

**解读输出**

| 指标 | 参考范围 | 说明 |
|------|----------|------|
| 年化收益（超额） | > 5% | 相对 CSI 500 基准的超额年化收益 |
| 夏普比率 | > 1.0 | 净值/波动率比 |
| 最大回撤 | < 20% | 净值历史最大峰谷跌幅 |
| 年化换手 | < 换手上限×年化次数 | 应与 `--turnover` 约束一致 |

> **MVP 限制**：模拟交易不接真实 OMS/经纪商，不做实盘下单。换手成本为线性近似（默认 15bps 单边），未建模市场冲击。

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
workspace/sim/<sim_id>/portfolio_report.html
```

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

# Step 1：挖因子（遗传搜索）
pixi run fz mine search \
  --start 20200101 --end 20231231 \
  --method genetic --trials 200 --top-k 10 --seed 42

# Step 2：防过拟合验收（替换 <factor> 为排行榜 top 因子名）
pixi run fz validate overfit <factor> --start 20200101 --end 20241231

# Step 3：建风险模型
pixi run fz risk build --start 20200101 --end 20241231

# Step 4：组合优化建仓
pixi run fz portfolio build \
  --start 20200101 --end 20241231 \
  --alpha-file workspace/discovery/<session_id>/alpha.parquet \
  --industry-neutral

# Step 5：模拟交易
pixi run fz sim run \
  --portfolio-dir workspace/portfolios/<run_id> \
  --start 20200101 --end 20241231

# Step 6：成果展示页
pixi run fz report portfolio \
  --sim-dir workspace/sim/<sim_id> \
  --portfolio-dir workspace/portfolios/<run_id>
```

---

## 全景：从信号到 Dashboard

```
                     FactorZen 端到端链路
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Tushare API                                                       │
│       │                                                             │
│       ▼                                                             │
│  [Step 0] data/tushare/                                             │
│   日行情 + 日基础数据 (parquet 缓存, PIT 对齐)                       │
│       │                                                             │
│       ▼                                                             │
│  [Step 1] fz mine search / agent / team         M1/M5/M6           │
│   算子库 AST 搜索 / LLM 闭环挖掘                                     │
│   → workspace/discovery/<session>/alpha.parquet                     │
│       │                                                             │
│       ▼                                                             │
│  [Step 2] fz validate overfit                   M2                 │
│   DSR + bootstrap IC CI + PBO/CSCV                                  │
│   holdout 段永久隔离                                                  │
│   → workspace/validation/<run>/dsr_report.json                      │
│       │                                                             │
│       ├──────────────────────────┐                                  │
│       ▼                          ▼                                  │
│  [Step 3] fz risk build     [Step 4] fz portfolio build  M3/M4     │
│   Barra 因子暴露              mean-variance QP (cvxpy)              │
│   Newey-West 协方差           行业中性 / 换手约束                     │
│   特质风险收缩                 Brinson + MCR 归因                    │
│   → workspace/risk_models/   → workspace/portfolios/               │
│                   │                          │                      │
│                   └──────────┬───────────────┘                      │
│                              ▼                                      │
│                    [Step 5] fz sim run          M7                  │
│                    多周期净值回测                                     │
│                    扣换手成本 / 停牌过滤                              │
│                    → workspace/sim/<id>/nav.parquet                 │
│                              │                                      │
│                              ▼                                      │
│                    [Step 6] fz report portfolio  M7                 │
│                    指标卡 + 净值曲线 + 热图                           │
│                    归因 + 风险摘要                                    │
│                    → workspace/sim/<id>/portfolio_report.html       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

  每个 run 产出 manifest.json：seed + 参数 + git_sha → 可审计、可复现
```

---

## MVP 限制汇总（诚实声明）

| 限制项 | 当前状态 | 计划 |
|--------|----------|------|
| 行业中性基准 | 行业等权基准（不是市值加权） | M4 后续版本升级为市值加权基准 |
| 收益归因精度 | 按日权重近似（不支持持仓期内多次调仓的精细核算） | 需补持仓期收益接口 |
| 换手成本模型 | 线性近似（固定 bps），未建模市场冲击和流动性折损 | 可扩展为非线性冲击模型 |
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
