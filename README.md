# FactorZen

> **FactorZen** 是一个面向 A 股单因子的可信研究框架，强调严谨、克制和可复现。当前核心主线覆盖因子计算、预处理、IC/回测评估、walk-forward OOS、数据质量报告、实验 manifest 与 Tear Sheet 报告生成。`research/combination/` 提供实验性多因子合成工具，用于研究对比，不作为当前生产组合优化模块。

## 目录结构与频率词汇表

| 目录 | 频率 | 数据来源 | 成熟度 |
|------|------|---------|--------|
| `daily/` | 日/周/月（日线下采样） | Tushare 日线行情 + 估值 + 财报 | ✅ 完整 |
| `intraday/` | 分钟（1min/5min） | Tushare 分钟线 | ✅ 评估管线完整（待实数据）|
| `research/` | 实验性研究工具 | 复用 daily/intraday 数据 | ⚠️ 非生产 |
| `common/` | 通用 | — | ✅ 完整 |
| `config/` | — | — | ✅ 完整 |
| `reporting/` | HTML Tear Sheet | — | ✅ 完整 |

> **命名说明**：`daily` ≡ 低频（业界常说的日频/月频因子）；`intraday` ≡ 日内（分钟级）；`research` ≡ 实验性研究工具。

## 快速开始

### 1. 环境安装

```bash
# 安装 pixi（Windows）
winget install prefix-dev.pixi

# 克隆项目后安装依赖（含 editable install）
cd FactorZen
pixi install
```

### 2. 配置 Tushare Token

```bash
cp .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN=your_token
```

在 [tushare.pro/user/token](https://tushare.pro/user/token) 获取 token。

### 3. 拉取行情数据

```bash
# 拉取日线行情（约 5 分钟）
pixi run python -c "from common.loader import fetch_daily; fetch_daily('20250101','20260513')"

# 拉取每日估值（PE/PB/市值，月频因子依赖）
pixi run python -c "from common.loader import fetch_daily_basic; fetch_daily_basic('20250101','20260513')"
```

### 4. 运行单因子评估

```bash
# 单因子完整评估 → output/daily/results/（保存 parquet 中间结果）
pixi run python scripts/run_daily_single.py --factor momentum_20d --start 20250101 --end 20260513

# 使用 YAML 配置运行；preprocessing/backtest/cost_model/walk_forward 字段会真实生效
pixi run python scripts/run_daily_single.py --config config/runs/single_factor_momentum.yaml

# 生成 HTML Tear Sheet → output/daily/reports/（同时落盘 parquet）
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513

# 复用已有 parquet 秒出报告（需先跑过上面任意一条）
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513 --reuse

# 多因子 IC 对比
pixi run python scripts/run_daily_compare.py --factors momentum_20d,reversal_5d --start 20250101 --end 20260513
```

主要输出：

- `output/daily/factors/{factor}_{start}_{end}.parquet`：预处理后的因子矩阵。
- `output/daily/results/{factor}_{start}_{end}_quality.json`：数据质量报告，包含覆盖率、重复键、收益对齐等检查结果。
- `output/daily/results/{factor}_{start}_{end}_meta.json`：报告生成元数据，包含 IC/回测/换手、配置摘要、`walk_forward_summary`。
- `output/daily/results/{factor}_{start}_{end}_walk_forward.json`：`run_daily_single.py` 的 walk-forward/OOS 摘要；IS 表示历史观察期，OOS 表示未来验证期，固定因子主流程不会在 IS 期重新拟合参数；样本不足时记录 `{"status": "insufficient_data", "n_folds": 0}`，不让主流程失败。
- `output/daily/reports/{factor}_{start}_{end}.html`：Tear Sheet，包含 OOS 摘要区块；没有 folds 时显示样本不足。
- `output/experiments/{run_id}/manifest.json`：实验 manifest，记录完整配置、命令、git SHA、dirty 状态、`pixi.lock` hash、成功/失败状态、错误信息和已生成输出路径。

## 可用因子列表

### daily — 日频（10 个）

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `momentum_20d` | 动量 | 20 日价格动量；保留兼容，研究上建议优先使用 `momentum_12_1` |
| `momentum_12_1` | 动量 | Jegadeesh-Titman 12-1 动量，剔除最近 1 个月反转效应 |
| `reversal_5d` | 反转 | 5 日短期反转 |
| `volatility_20d` | 波动 | 20 日已实现波动率 |
| `turnover_5d` | 换手 | 5 日平均换手率 |
| `amihud_illiquidity` | 流动性 | Amihud (2002) 非流动性指标 |
| `beta_60d` | 风险 | 60 日 CAPM Beta |
| `idiosyncratic_vol_20d` | 波动 | 20 日特质波动率（去除市场 Beta 后残差 std） |
| `max_return_5d` | 彩票效应 | 5 日最大单日涨幅（Bali et al. 2011 MAX 因子） |
| `skewness_20d` | 偏度 | 20 日收益偏度（正偏股票未来收益偏低） |

### daily — 周频（3 个）

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `momentum_weekly` | 动量 | 周频快照动量 |
| `turnover_weekly` | 换手 | 周频快照换手率 |
| `volatility_weekly` | 波动 | 周频快照波动率 |

### daily — 月频（6 个）

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `pe_ttm` | 估值 | 月频滚动市盈率（依赖 daily_basic） |
| `pb` | 估值 | 月频市净率（依赖 daily_basic） |
| `ep_ratio` | 估值 | 月频 E/P（= 1/PE_TTM） |
| `bm_ratio` | 估值 | 月频 B/M（= 1/PB） |
| `roe_ttm` | 质量 | 月频 ROE TTM，PIT 对齐（依赖 finance） |
| `asset_growth` | 质量 | 年度总资产增速（依赖 finance） |

### intraday — 分钟频（2 个）

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `momentum_1min` | 动量 | 1 分钟 5-bar 收益动量 |
| `vwap_deviation` | 价格偏离 | 当前价相对日内 VWAP 偏离度 |

## 开发命令

```bash
pixi run test      # 运行测试
pixi run lint      # ruff check
pixi run typecheck # mypy：common、daily/evaluation、daily/preprocessing、daily/factors、research/combination、reporting、automation
pixi run coverage  # pytest coverage，当前门槛 70%
pixi run format    # ruff format
pixi run lab       # 启动 JupyterLab
```

## 项目架构

```
FactorZen/
├── common/          # 数据底座（loader/storage/calendar/universe）
├── config/          # 路径常量、Tushare 配置
├── daily/           # 日/周/月频因子框架
│   ├── data/        # FactorDataContext（懒加载、PIT 对齐）
│   ├── factors/     # 因子实现（qlib + personal）
│   │   ├── qlib/     # Qlib Alpha158 / Alpha360 因子
│   │   └── personal/ # 个人因子库（daily/weekly/monthly/style/custom）
│   ├── preprocessing/   # 去极值 → 填充 → 标准化 → 中性化
│   └── evaluation/  # IC / 回测 / 换手 / 相关性 / 高级指标
├── intraday/        # 分钟频因子框架
├── research/        # 实验性研究工具（非生产优化）
│   └── combination/ # 实验性多因子合成（研究对比）
├── reporting/       # HTML Tear Sheet 生成
├── scripts/         # CLI 入口
├── benchmarks/      # 性能基准脚本
└── tests/           # pytest 测试套件
```

## 数据目录约定

```
data/
├── raw/
│   ├── daily/year=YYYY/month=MM/data.parquet       # 日线行情
│   ├── daily_basic/year=YYYY/month=MM/data.parquet # 每日估值
│   ├── finance/year=YYYY/quarter=Q/data.parquet    # 财务数据
│   └── minute/year=YYYY/month=MM/data.parquet      # 分钟线
└── cache/           # 股票池、交易日历等小型缓存

output/
├── daily/
│   ├── factors/     # 预处理后因子矩阵 parquet；qlib 因子按 qlib158/qlib360 分目录
│   ├── results/     # IC/BT/TO 评价结果 parquet + 元数据 JSON；qlib 因子按 qlib158/qlib360 分目录
│   └── reports/     # HTML Tear Sheet；qlib 因子按 qlib158/qlib360 分目录
└── intraday/
    ├── results/     # intraday IC 结果
    └── reports/     # intraday HTML 报告
```

## 已知边界

- **Tick 级研究**：当前不保留正式代码包。Tushare 不提供 Tick 数据，未来如对接 CTP 或 Wind，应单独设计数据 adapter、订单簿/逐笔存储与评估口径。
- **`research/combination/`**：实验性多因子合成（等权、IC 加权、Max-IR），用于研究阶段对比。当前权重估计仍是 in-sample 口径，不能把组合结果称为 OOS，也不能直接解释为生产可交易组合优化，见 [`research/combination/README.md`](research/combination/README.md)。
- **生产交易边界**：当前框架聚焦研究可信度，不包含 tick 数据、实盘 OMS、盘口成交、真实 Tushare 网络 smoke 或生产级组合执行闭环。
