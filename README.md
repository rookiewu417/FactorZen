# 单因子研究框架

> **当前阶段**：专注于**单因子研究**——因子计算、预处理、IC/回测评估、Tear Sheet 报告生成。多因子合成、组合优化等功能在计划范围之外。

## 目录结构与频率词汇表

| 目录 | 频率 | 数据来源 | 成熟度 |
|------|------|---------|--------|
| `daily/` | 日/周/月（日线下采样） | Tushare 日线行情 + 估值 + 财报 | ✅ 完整 |
| `intraday/` | 分钟（1min/5min） | Tushare 分钟线 | 🚧 评估管线开发中 |
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
# 单因子完整评估 → output/lft/results/
pixi run python scripts/run_lft_single.py --factor momentum_20d --start 20250101 --end 20260513

# 生成 HTML Tear Sheet → output/lft/reports/
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513

# 多因子 IC 对比
pixi run python scripts/run_lft_compare.py --factors momentum_20d,reversal_5d --start 20250101 --end 20260513
```

## 可用因子列表

### daily — 日频

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `momentum_20d` | 动量 | 20 日价格动量 |
| `reversal_5d` | 反转 | 5 日短期反转 |
| `volatility_20d` | 波动 | 20 日已实现波动率 |
| `turnover_20d` | 换手 | 20 日平均换手率 |

### daily — 周频

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `weekly_momentum_4w` | 动量 | 4 周动量 |
| `weekly_turnover_4w` | 换手 | 4 周换手率 |
| `weekly_volatility_4w` | 波动 | 4 周波动率 |

### daily — 月频

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `monthly_value` | 估值 | PE/PB 综合估值（依赖 daily_basic 数据） |
| `monthly_profitability` | 质量 | ROE（依赖 finance 数据） |

### intraday — 分钟频（评估管线开发中）

| 因子名 | 类别 | 描述 |
|--------|------|------|
| `momentum_1min` | 动量 | 1 分钟收益动量（demo） |

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
├── daily/           # 日/周/月频因子框架（原 lft/）
│   ├── data/        # FactorDataContext（懒加载、PIT 对齐）
│   ├── factors/     # 因子实现（daily/weekly/monthly/custom）
│   ├── preprocessing/   # 去极值 → 填充 → 标准化 → 中性化
│   ├── evaluation/  # IC / 回测 / 换手 / 相关性 / 高级指标
│   └── combination/ # 🚫 预留：多因子合成（本期不实现）
├── intraday/        # 分钟频因子框架（原 mft/）
├── tick/            # 🚫 Tick 级预留（原 hft/）
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
```

## 已知预留位

- **`tick/`**：Tushare 不提供 Tick 数据。未来对接 CTP 或 Wind 时填充，见 [`tick/README.md`](tick/README.md)。
- **`daily/combination/`**：多因子合成（IC 加权/等权/PCA）是单因子研究的下一阶段产物，当前不实现，见 [`daily/combination/README.md`](daily/combination/README.md)（待阶段 2 重命名后路径更新）。
