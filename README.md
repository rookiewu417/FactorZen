# 单因子研究框架

> **当前阶段**：核心主线是**单因子研究**——因子计算、预处理、IC/回测评估、Tear Sheet 报告生成。`daily/combination/` 已提供实验性多因子合成工具，用于研究对比，不作为当前生产组合优化模块。

## 目录结构与频率词汇表

| 目录 | 频率 | 数据来源 | 成熟度 |
|------|------|---------|--------|
| `daily/` | 日/周/月（日线下采样） | Tushare 日线行情 + 估值 + 财报 | ✅ 完整 |
| `intraday/` | 分钟（1min/5min） | Tushare 分钟线 | ✅ 评估管线完整（待实数据）|
| `tick/` | Tick 级 | 预留位（Tushare 不提供） | 🚫 未实现 |
| `common/` | 通用 | — | ✅ 完整 |
| `config/` | — | — | ✅ 完整 |
| `reporting/` | HTML Tear Sheet | — | ✅ 完整 |

> **命名说明**：`daily` ≡ 低频（业界常说的日频/月频因子）；`intraday` ≡ 日内（分钟级）；`tick` ≡ 高频（HFT，预留）。

## 快速开始

### 1. 环境安装

```bash
# 安装 pixi（Windows）
winget install prefix-dev.pixi

# 克隆项目后安装依赖（含 editable install）
cd 因子研究
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
python -c "from common.loader import fetch_daily; fetch_daily('20250101','20260513')"

# 拉取每日估值（PE/PB/市值，月频因子依赖）
python -c "from common.loader import fetch_daily_basic; fetch_daily_basic('20250101','20260513')"
```

### 4. 运行单因子评估

```bash
# 单因子完整评估 → output/daily/results/（保存 parquet 中间结果）
pixi run python scripts/run_daily_single.py --factor momentum_20d --start 20250101 --end 20260513

# 生成 HTML Tear Sheet → output/daily/reports/（同时落盘 parquet）
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513

# 复用已有 parquet 秒出报告（需先跑过上面任意一条）
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513 --reuse

# 多因子 IC 对比
pixi run python scripts/run_daily_compare.py --factors momentum_20d,reversal_5d --start 20250101 --end 20260513
```

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
pixi run format    # ruff format
pixi run lab       # 启动 JupyterLab
```

## 项目架构

```
因子研究/
├── common/          # 数据底座（loader/storage/calendar/universe）
├── config/          # 路径常量、Tushare 配置
├── daily/           # 日/周/月频因子框架
│   ├── data/        # FactorDataContext（懒加载、PIT 对齐）
│   ├── factors/     # 因子实现（daily/weekly/monthly/custom）
│   ├── preprocessing/   # 去极值 → 填充 → 标准化 → 中性化
│   ├── evaluation/  # IC / 回测 / 换手 / 相关性 / 高级指标
│   └── combination/ # 实验性多因子合成（研究对比，非生产优化）
├── intraday/        # 分钟频因子框架
├── tick/            # 🚫 Tick 级预留
├── reporting/       # HTML Tear Sheet 生成
├── scripts/         # CLI 入口
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
│   ├── factors/     # 预处理后因子矩阵 parquet
│   ├── results/     # IC/BT/TO 评价结果 parquet + 元数据 JSON
│   └── reports/     # HTML Tear Sheet
└── intraday/
    ├── results/     # intraday IC 结果
    └── reports/     # intraday HTML 报告
```

## 已知预留位

- **`tick/`**：Tushare 不提供 Tick 数据。未来对接 CTP 或 Wind 时填充，见 [`tick/README.md`](tick/README.md)。
- **`daily/combination/`**：实验性多因子合成（等权、IC 加权、Max-IR），用于研究阶段对比。当前权重估计仍是 in-sample 口径，不能直接解释为生产可交易组合优化，见 [`daily/combination/README.md`](daily/combination/README.md)。
