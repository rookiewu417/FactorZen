# FactorZen 项目完整文档

本文档面向后续维护者、研究使用者和自动化代理，解释 FactorZen 的目标、目录结构、运行方式、核心数据流、主要模块、扩展方法、质量门和已知边界。

最后更新：2026-05-25  
当前版本：`0.2.0`

---

## 1. 项目定位

FactorZen 是一个面向 A 股单因子研究的可信研究框架。它的目标不是做实盘交易系统，而是把一个因子从数据获取、因子计算、预处理、IC 分析、策略回测、walk-forward 样本外验证、质量审计、实验记录到 HTML 报告生成串成可复现链路。

当前主线能力：

- 日/周/月频因子研究：`daily/`
- 分钟级日内因子研究：`intraday/`
- 原始数据拉取、Parquet 存储、交易日历、股票池、实验记录：`common/`
- 路径、常量、Tushare 连接配置：`config/`
- HTML Tear Sheet 报告：`reporting/`
- CLI 入口脚本：`scripts/`
- 日终调度流水线：`automation/`
- 实验性研究工具：`research/`

明确不覆盖：

- 实盘 OMS / EMS
- 盘口撮合、订单簿重建、逐笔成交执行
- 真实生产组合执行闭环
- Tick 数据源接入
- 对 Tushare 网络状态的 CI 级真实 smoke test

## 2. 快速开始

### 2.1 安装环境

```bash
pixi install
```

项目使用 pixi 管理环境，默认环境包含开发依赖，并通过 editable install 安装本地包。

### 2.2 配置 Tushare

复制 `.env.example` 为 `.env`，写入：

```bash
TUSHARE_TOKEN=your_tushare_token
```

`config/tushare_config.py` 会从环境变量或 `.env` 读取 token。token 在首次真正调用 Tushare 时校验，离线测试不会因 import 失败。

### 2.3 拉取基础数据

```bash
pixi run python -c "from common.loader import fetch_daily; fetch_daily('20250101','20260513')"
pixi run python -c "from common.loader import fetch_daily_basic; fetch_daily_basic('20250101','20260513')"
```

### 2.4 运行单因子研究

```bash
pixi run python scripts/run_daily_single.py --factor momentum_20d --start 20250101 --end 20260513
```

使用 YAML 配置：

```bash
pixi run python scripts/run_daily_single.py --config config/runs/single_factor_momentum.yaml
```

生成报告：

```bash
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513
```

### 2.5 常用开发命令

```bash
pixi run lint
pixi run typecheck
pixi run pytest tests --tb=short -q
pixi run coverage
pixi run format
pixi run lab
```

说明：

- `typecheck` 使用 `.cache/mypy` 且禁用 mypy sqlite cache，避免 Windows/WSL UNC 路径下 database locked。
- `coverage` 通过 `scripts/run_coverage.py` 使用唯一临时 coverage 数据文件，避免仓库根目录 `.coverage` 锁冲突。

## 3. 技术栈

### 3.1 Python 与构建

- Python：`>=3.10,<3.13`
- 构建后端：`hatchling`
- 包管理：`pixi`
- 项目包名：`factorzen`
- 版本：`0.2.0`
- 平台：`win-64`、`linux-64`

### 3.2 运行依赖

`pyproject.toml` 和 `pixi.toml` 中声明的核心依赖：

- `polars`：主 DataFrame / LazyFrame 引擎
- `pyarrow`：Parquet 支撑
- `tushare`：A 股数据源
- `pandas`：Tushare 返回数据桥接
- `numpy`、`scipy`：数值计算
- `statsmodels`：OLS、中性化、HAC / Newey-West t 统计
- `matplotlib`：报告图表
- `jinja2`：HTML 模板
- `cvxpy`：组合优化
- `optuna`：walk-forward 超参搜索
- `apscheduler`：自动化调度
- `pydantic`、`PyYAML`：YAML 运行配置验证
- `joblib`：截面中性化并行

### 3.3 开发依赖

- `pytest`
- `coverage`
- `ruff`
- `mypy`
- `jupyterlab`

## 4. 目录总览

```text
FactorZen/
├── common/          # 通用底座：数据拉取、存储、股票池、日历、实验记录
├── config/          # 路径、常量、Tushare、YAML run config
├── daily/           # 日/周/月频因子研究主线
├── intraday/        # 分钟级因子研究
├── research/        # 实验性研究工具
├── reporting/       # HTML Tear Sheet 报告
├── automation/      # 调度 DAG、任务状态、作业封装
├── benchmarks/      # 手动性能基准
├── scripts/         # CLI 入口
├── tests/           # pytest 测试套件
├── docs/            # 项目说明、发布说明、历史归档
├── data/            # 本地数据，git ignored
└── output/          # 本地输出，git ignored
```

打包范围在 `pyproject.toml` 中声明：

```toml
packages = ["common", "config", "daily", "intraday", "research", "reporting", "automation"]
```

## 5. 配置体系

### 5.1 `config/settings.py`

集中定义路径和调度默认值：

- `ROOT`
- `CONFIG_DIR`
- `DATA_DIR`
- `DATA_RAW`
- `DATA_CACHE`
- `OUTPUT_DIR`
- `OUTPUT_DAILY_FACTORS`
- `OUTPUT_DAILY_RESULTS`
- `OUTPUT_DAILY_REPORTS`
- `OUTPUT_INTRADAY_FACTORS`
- `OUTPUT_INTRADAY_RESULTS`
- `OUTPUT_INTRADAY_REPORTS`
- `AUTOMATION_OUTPUT`
- `SCHEDULER_TIMEZONE`
- `SCHEDULER_CRON_HOUR`
- `SCHEDULER_CRON_MINUTE`
- `SCHEDULER_MAX_RETRIES`
- `SCHEDULER_RETRY_BASE_SECONDS`

所有模块应从 `config.settings` 引用路径，不应在业务代码中拼硬编码项目根目录。

### 5.2 `config/constants.py`

集中定义研究常量：

- 年、月、周交易日数量
- MAD 去极值参数
- IC 最小样本数
- 默认分位数组数
- 涨跌停阈值
- 研究评分阈值
- 回测有效性阈值
- 默认交易成本
- 基准指数映射

### 5.3 `config/tushare_config.py`

职责：

- 读取 `.env`
- 暴露 `TUSHARE_TOKEN`
- 提供 `ensure_token()`
- 配置 Tushare 积分、限流、重试和缓存过期参数

默认值：

- `TUSHARE_POINTS = 2000`
- `MAX_RPS = 5`
- `MAX_RETRIES = 3`
- `RETRY_DELAY = 1.0`
- `CACHE_EXPIRE_DAYS = 7`

### 5.4 YAML 运行配置

`common/config_loader.py` 使用 Pydantic v2 验证 YAML 配置。

核心 schema：

```yaml
factor: momentum_20d
universe: csi500
start: "20230101"
end: "20241231"
benchmark: "000300.SH"
seed: 42
preprocessing:
  outlier: mad
  normalizer: zscore
  neutralize: false
  neutralize_by: industry+size
backtest:
  top_n: 50
  quantiles: 5
  max_abs_weight: 0.1
  cost_model: linear
  rebalance_threshold: null
walk_forward:
  train_days: 504
  test_days: 63
  step_days: 63
  embargo_days: 5
  n_trials: 50
ic_method: rank
event_study: false
neutralized_ic: false
```

当前配置文件：

- `config/runs/single_factor_momentum.yaml`
- `config/runs/walk_forward_example.yaml`

注意：`preprocessing.neutralize=true` 时，主链路会按 `neutralize_by` 传入行业和/或市值数据：

- `industry`：使用股票池中的 `industry`
- `size`：加载 `daily_basic.total_mv`
- `industry+size`：同时使用行业和市值

## 6. 数据目录与存储格式

### 6.1 原始数据目录

```text
data/raw/
├── daily/year=YYYY/month=MM/data.parquet
├── daily_basic/year=YYYY/month=MM/data.parquet
├── finance/year=YYYY/month=MM/data.parquet
├── minute/year=YYYY/month=MM/data.parquet
└── adj_factor/year=YYYY/month=MM/data.parquet
```

### 6.2 缓存目录

```text
data/cache/
├── stock_basic_L_D_P.parquet
├── trade_cal.parquet
└── index_member_<index>_<yyyymm>.parquet
```

### 6.3 输出目录

```text
output/
├── daily/
│   ├── factors/
│   ├── results/
│   ├── charts/
│   └── reports/
├── intraday/
│   ├── factors/
│   ├── results/
│   └── reports/
├── experiments/
└── automation/
```

`data/` 和 `output/` 都属于本地运行产物，不应纳入 git。

## 7. 通用底座 `common/`

### 7.1 `common/storage.py`

提供统一 Parquet 读写：

- `save_parquet(df, data_type, date_col="trade_date", mode="append")`
- `load_parquet(data_type, start=None, end=None, date_col="trade_date")`
- `scan_parquet(data_type)`
- `partition_exists(data_type, year, month)`

写入采用 Hive 风格年月分区。读取返回 `pl.LazyFrame`，支持日期过滤。

### 7.2 `common/loader.py`

项目中唯一直接调用 Tushare 的模块。上层模块不应直接调用 `tushare`。

职责：

- 初始化 Tushare 单例
- 限流
- 网络错误重试
- pandas 到 polars 转换
- 按年/月/季度分段拉取
- 缓存命中跳过

主要函数：

- `fetch_daily(start, end, ts_codes=None)`
- `fetch_daily_basic(start, end, ts_codes=None)`
- `fetch_minute(ts_code, freq, start, end, call_delay=0.0)`
- `fetch_finance(api_name, start, end, ts_codes=None, fields=None)`
- `fetch_stock_basic(list_status="L,D,P")`
- `fetch_adj_factor(start, end)`
- `fetch_index_daily(index_code, start, end)`
- `fetch_trade_cal(start, end)`

### 7.3 `common/calendar.py`

负责交易日历：

- 获取交易日列表
- 前后交易日计算
- 周频/月频快照日期
- 本地缓存交易日历

日/周/月频因子统一依赖这里定义的快照规则。

### 7.4 `common/universe.py`

负责股票池构建。

预设股票池：

- `all_a`
- `csi300`
- `csi500`
- `csi800`
- `daily_default`
- `intraday_default`

过滤器：

- ST / PT 过滤
- 次新股过滤
- 停牌过滤
- 涨跌停过滤
- 流动性过滤

指数成分通过 Tushare `index_weight` 按月缓存。若指数成分加载失败，会降级为全 A 股并记录 warning。

### 7.5 `common/registry.py`

通用注册中心，用于因子发现：

- 传入基类
- 扫描指定包
- 注册继承该基类的类
- 按 `name` 获取因子

`daily.factors.registry` 和 `intraday.factors.registry` 都基于它实现。

### 7.6 `common/config_loader.py`

职责：

- 读取 YAML
- Pydantic 校验
- 构建预处理管线
- 构建回测配置
- 构建成本模型

成本模型选择：

- `linear`
- `square_root_impact`

### 7.7 `common/data_quality.py`

日频质量报告，用于检查：

- 原始行情是否为空
- 因子覆盖率
- 重复键
- 清洗后覆盖率
- 前向收益对齐
- 股票池覆盖率

主链路会输出 `*_quality.json`。

### 7.8 `common/experiment.py`

实验 manifest 记录：

- `run_id`
- git SHA
- dirty 状态
- `pixi.lock` hash
- 命令行参数
- 完整配置
- 输出路径
- 开始/结束时间
- 成功/失败状态
- 错误信息

输出位置：

```text
output/experiments/{run_id}/manifest.json
```

## 8. 因子抽象与注册

### 8.1 通用因子接口

`common/factor.py` 定义基础接口。因子需要提供：

- `name`
- `category`
- `description`
- `frequency`
- `required_data`
- `lookback_days`
- `compute(ctx)`
- `validate(df)`

### 8.2 日频因子

`daily/factors/base.py` 定义 `DailyFactor`。

注册中心：

```python
from daily.factors.registry import get_factor, list_factors
```

扫描包：

- `daily.factors.daily`
- `daily.factors.weekly`
- `daily.factors.monthly`

### 8.3 日内因子

`intraday/factors/base.py` 定义日内因子接口。

注册中心：

```python
from intraday.factors.registry import get_factor, list_factors
```

扫描包：

- `intraday.factors.demo`
- `intraday.factors.technical`

## 9. 日频数据上下文

`daily/data/context.py` 的 `FactorDataContext` 是日频因子计算的主要数据入口。

创建参数：

- `start`
- `end`
- `required_data`
- `lookback_days`
- `universe`
- `snapshot_mode`

能力：

- 自动扩展起始日期以满足 lookback
- 惰性加载 `daily`
- 惰性加载 `daily_basic`
- 如果存在 `adj_factor`，自动生成复权价列
- 无复权因子时回退为原始价格
- 支持日/周/月快照

主要属性：

- `expanded_start`
- `daily`
- `daily_basic`
- `snapshot_dates`
- `weekly`
- `monthly`
- `weekly_basic`
- `monthly_basic`

## 10. 已实现因子

### 10.1 日频因子

| 名称 | 类别 | 数据 |
| --- | --- | --- |
| `momentum_20d` | 动量 | daily |
| `momentum_12_1` | 动量 | daily |
| `reversal_5d` | 反转 | daily |
| `volatility_20d` | 波动 | daily |
| `turnover_5d` | 换手 | daily |
| `amihud_illiquidity` | 流动性 | daily |
| `beta_60d` | 风险 | daily |
| `idiosyncratic_vol_20d` | 波动 | daily |
| `max_return_5d` | 彩票效应 | daily |
| `skewness_20d` | 偏度 | daily |

### 10.2 周频因子

| 名称 | 类别 |
| --- | --- |
| `momentum_weekly` | 动量 |
| `turnover_weekly` | 换手 |
| `volatility_weekly` | 波动 |

### 10.3 月频因子

| 名称 | 类别 | 依赖 |
| --- | --- | --- |
| `pe_ttm` | 估值 | daily_basic |
| `pb` | 估值 | daily_basic |
| `ep_ratio` | 估值 | daily_basic |
| `bm_ratio` | 估值 | daily_basic |
| `roe_ttm` | 质量 | finance / PIT |
| `asset_growth` | 质量 | finance / PIT |

### 10.4 风格因子

`daily/factors/style/` 提供 Barra 风格暴露相关实现：

- beta
- liquidity
- momentum
- size
- value
- volatility

### 10.5 日内因子

| 名称 | 类别 |
| --- | --- |
| `momentum_1min` | 1 分钟动量 |
| `vwap_deviation` | VWAP 偏离 |

## 11. PIT 财务对齐

`daily/data/pit.py` 负责 point-in-time 财务数据对齐。

核心目标：

- 研究日期只能看到当时已经公告的数据
- 财报 `end_date` 不能直接泄漏未来信息
- 用公告日期或可见日期控制财务因子的有效性

月频质量因子如 `roe_ttm`、`asset_growth` 依赖 PIT 逻辑。

## 12. 日频预处理管线

`daily/preprocessing/pipeline.py` 定义 `PreprocessingPipeline`。

默认步骤：

```python
["outlier", "missing", "normalize"]
```

### 12.1 去极值

支持：

- `mad`
- `winsorize`
- `sigma`

实现文件：

- `daily/preprocessing/outlier.py`

### 12.2 缺失值处理

当前主方法：

- 按截面中位数填充

实现文件：

- `daily/preprocessing/missing.py`

### 12.3 标准化

支持：

- `zscore`
- `rank_uniform`
- `rank_normal`
- `quantile_normal`

实现文件：

- `daily/preprocessing/normalizer.py`

### 12.4 中性化

实现文件：

- `daily/preprocessing/neutralizer.py`

支持：

- 行业中性化
- 市值中性化
- 行业 + 市值中性化
- style 暴露中性化

主链路行为：

- YAML 中 `preprocessing.neutralize=true` 时，`scripts/run_daily_single.py` 会调用 `_preprocess_factor`。
- `neutralize_by=industry` 会传入股票池行业字段。
- `neutralize_by=size` 会加载并传入 `daily_basic.total_mv`。
- `neutralize_by=industry+size` 会同时传入行业和市值。

如果所需 side data 缺失，`neutralize_ols` 会记录 warning 并按现有策略回退。

## 13. 前向收益与 IC 分析

### 13.1 前向收益

`daily/evaluation/ic_analysis.py` 提供：

- `compute_fwd_returns`

当前主链路计算 horizon：

- 1d
- 5d
- 10d
- 20d

### 13.2 Rank IC

主函数：

- `compute_rank_ic`

输出：

- IC 时间序列
- 均值
- 标准差
- IR
- t 值
- 正 IC 比率
- HAC / Newey-West 口径统计

### 13.3 Pearson IC

YAML / CLI 支持：

- `rank`
- `pearson`
- `both`

`both` 会同时输出 Rank IC 和 Pearson IC。

## 14. 策略回测

实现文件：

- `daily/evaluation/backtest.py`

### 14.1 回测口径

核心约定：

- t 日因子生成目标权重
- t+1 开盘执行调仓
- 旧持仓承担 overnight return
- 新持仓承担 open-to-close return

`ret_definition`：

```text
open_to_close_with_overnight_carry
```

### 14.2 策略接口

策略继承 `Strategy` 并实现：

```python
def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
    ...
```

返回列：

- `ts_code`
- `target_weight`

内置策略：

- `QuantileLongShortStrategy`
- `TopNLongOnlyStrategy`
- `FactorWeightedStrategy`
- `OptimizerStrategy`

### 14.3 `BacktestContext`

字段：

- `signal_date`
- `execution_date`
- `factor_slice`
- `price_slice`
- `current_positions`
- `factor_col`
- `price_history`
- `adv_20d`

`adv_20d` 为执行日前最多 20 个交易期的平均成交额，用于平方根冲击成本模型。

### 14.4 成交约束

当前支持：

- 停牌不能成交
- 涨停不能买入
- 跌停不能卖出
- 按 `max_participation_rate` 控制成交容量
- `max_abs_weight`
- `max_gross_exposure`
- `rebalance_threshold`

### 14.5 成本模型

旧兼容模型：

- `CostModel`

新成本接口：

- `daily/evaluation/cost_models.py`
- `CostModelBase`
- `LinearCostModel`
- `SquareRootImpactCostModel`

`SquareRootImpactCostModel`：

- 线性成本：佣金、印花税、滑点
- 冲击成本：`alpha * |delta_weight|^1.5`
- 如果提供 ADV，则按流动性缩放冲击成本
- 主回测会传入 `adv_20d`

## 15. Walk-Forward 样本外验证

主要文件：

- `daily/evaluation/walk_forward.py`
- `daily/evaluation/walk_forward_summary.py`
- `scripts/run_walk_forward.py`

配置：

- `train_days`：IS 历史观察期长度。字段名保留为 `train` 是为了兼容配置，不表示固定因子会在这里重新拟合。
- `test_days`：OOS 未来验证期长度。
- `step_days`
- `embargo_days`
- `n_trials`

目标：

- 在历史观察期得到 IS 表现参照
- 隔开 embargo 后，在未来验证期做 OOS 评估
- 记录 IS/OOS 稳定性

注意：`run_daily_single.py` 的固定单因子主流程不会在 IS 期训练因子公式，也不会在每折里重新调参。这里的 IS/OOS 切分主要用于检查“过去看起来有效”的结论能否延续到后续时间段。只有显式运行超参搜索时，IS/OOS 才会参与参数选择，此时还需要额外 holdout 才能作为最终无偏结论。

样本不足时主流程不失败，而是输出：

```json
{"status": "insufficient_data", "n_folds": 0}
```

## 16. 高级评估指标

实现文件：

- `daily/evaluation/advanced.py`

包含：

- IC decay
- 单调性
- 行业分层 IC
- 市值分层 IC
- 因子拥挤度
- 市场状态分层 IC
- Rank autocorrelation
- Neutralized IC
- Event study
- Factor correlation

这些指标用于增强研究解释，不应替代主链路质量审计和 OOS 验证。

## 17. Benchmark 与归因

### 17.1 Benchmark

实现文件：

- `daily/evaluation/benchmark.py`

能力：

- 拉取或加载基准指数
- 计算组合相对基准的超额收益
- 输出 benchmark summary

### 17.2 归因

实现文件：

- `daily/evaluation/attribution.py`

能力：

- 分组收益贡献
- 多空贡献拆分
- 简化归因统计

## 18. 组合优化

目录：

- `daily/optimization/`

核心文件：

- `base.py`
- `covariance.py`
- `mean_variance.py`
- `max_sharpe.py`
- `risk_parity.py`

优化器依赖 `cvxpy`。当前用于研究型权重生成，不构成生产级组合管理系统。

约束包括：

- 最大权重
- 最小权重
- gross exposure
- net exposure
- turnover limit

## 19. 多因子合成

目录：

- `research/combination/`

定位：

- 实验性多因子合成
- 用于研究对比
- 不作为生产组合优化模块

方法：

- 等权
- IC 加权
- Max-IR

重要边界：

- `ic_weighted` 和 `max_ir` 依赖样本内估计
- 不能把合成结果直接称为 OOS
- 不能直接解释为可交易生产组合

## 20. 日内模块

目录：

- `intraday/`

### 20.1 数据上下文

`intraday/data/context.py` 负责加载分钟数据。

输入一般包括：

- `ts_code`
- `trade_time`
- `open`
- `high`
- `low`
- `close`
- `vol`
- `amount`

### 20.2 日内预处理

`intraday/preprocessing/pipeline.py` 提供分钟级清洗与标准化逻辑。

### 20.3 日内收益与 IC

实现文件：

- `intraday/evaluation/returns.py`
- `intraday/evaluation/ic_analysis.py`

### 20.4 日内回测

实现文件：

- `intraday/evaluation/backtest.py`

### 20.5 日内 CLI

```bash
pixi run python scripts/run_intraday_single.py --factor momentum_1min --ts-code 000001.SZ --start 20260401 --end 20260430
```

具体参数以脚本 argparse 为准。

## 21. Tick 级研究边界

当前状态：

- 不保留正式代码包
- 不在当前 roadmap 投入新功能
- 未来如需支持，应单独立项

- Tick 数据源
- 订单簿 / 逐笔成交存储
- TickDataContext
- Tick 因子接口
- Tick 评估口径

## 22. 报告系统

目录：

- `reporting/`

主要文件：

- `reporting/tear_sheet.py`
- `reporting/templates/tear_sheet.html`
- `reporting/templates/intraday_ic.html`

日频 Tear Sheet 包含：

- 因子元信息
- IC 统计
- IC 时间序列
- 分层回测结果
- 换手率
- benchmark 对比
- event study
- walk-forward / OOS 摘要
- Pearson IC
- Neutralized IC

报告输出：

```text
output/daily/reports/{factor}_{start}_{end}.html
```

## 23. CLI 入口

### 23.1 单因子完整评估

```bash
pixi run python scripts/run_daily_single.py --factor momentum_20d --start 20250101 --end 20260513
```

支持参数：

- `--factor`
- `--start`
- `--end`
- `--universe`
- `--frequency`
- `--benchmark`
- `--config`
- `--seed`
- `--ic-method`
- `--neutralized-ic`
- `--event-study`

### 23.2 报告生成

```bash
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513
pixi run report -- --factor momentum_20d --start 20250101 --end 20260513 --reuse
```

### 23.3 多因子比较

```bash
pixi run python scripts/run_daily_compare.py --factors momentum_20d,reversal_5d --start 20250101 --end 20260513
```

### 23.4 多因子合成

```bash
pixi run python scripts/run_combination.py --factors momentum_20d,reversal_5d --method equal_weight --start 20250101 --end 20260513
```

### 23.5 Walk-forward

```bash
pixi run python scripts/run_walk_forward.py --factor momentum_20d --start 20210101 --end 20241231
```

### 23.6 超参搜索

```bash
pixi run python scripts/run_hyperparameter_search.py --factor momentum_20d --start 20210101 --end 20241231
```

### 23.7 数据拉取

```bash
pixi run python scripts/fetch_data.py
pixi run python scripts/fetch_all_data.py
pixi run python scripts/fetch_minute_data.py
```

## 24. 自动化调度

目录：

- `automation/`

主要文件：

- `automation/dag.py`
- `automation/jobs.py`
- `automation/state.py`

能力：

- APScheduler 后台调度
- cron trigger
- 作业状态记录
- 重试配置
- 自动化输出写入 `output/automation`

入口：

```bash
pixi run python scripts/run_scheduler.py
```

默认调度时间来自 `config/settings.py`：

- 时区：`Asia/Shanghai`
- 时间：16:30
- 最大重试：3
- retry base：60 秒

## 25. 实验可复现性

每次主研究流程通过 `run_experiment` 包裹，会写 manifest。

示例结构：

```json
{
  "run_id": "20260525_180000",
  "git_sha": "...",
  "git_dirty": true,
  "pixi_lock_sha256": "...",
  "command": ["scripts/run_daily_single.py", "..."],
  "config": {},
  "outputs": {},
  "start_ts": "...",
  "end_ts": "...",
  "status": "success",
  "error": null
}
```

失败时：

- `status = "failure"`
- `error` 记录异常信息
- 已生成输出仍尽量记录到 `outputs`

## 26. 质量门

### 26.1 Ruff

配置：

- `target-version = "py310"`
- `line-length = 100`
- 启用 `E`、`F`、`I`、`UP`、`B`、`SIM`、`RUF`

命令：

```bash
pixi run lint
pixi run format
```

### 26.2 Mypy

当前为渐进式 typecheck。

覆盖范围：

- `common`
- `daily/evaluation`
- `daily/preprocessing`
- `daily/factors`
- `research/combination`
- `reporting`
- `automation`

命令：

```bash
pixi run typecheck
```

注意：

- 使用 `.cache/mypy`
- 禁用 sqlite cache，避免 Windows/WSL UNC 下锁文件问题

### 26.3 Coverage

覆盖范围：

- `common`
- `daily/evaluation`
- `daily/preprocessing`
- `daily/factors`
- `reporting`
- `automation`

排除：

- `scripts/*`
- `docs/archive/*`
- `tests/*`

门槛：

```text
70%
```

命令：

```bash
pixi run coverage
```

当前 `coverage` task 使用 `scripts/run_coverage.py`，每次生成唯一临时 coverage 数据文件。

### 26.4 测试

命令：

```bash
pixi run pytest tests --tb=short -q
```

最近一次验证：

- `395 passed`
- coverage total：`73%`

## 27. 新增日频因子指南

1. 在合适目录新增因子文件：

```text
daily/factors/daily/
daily/factors/weekly/
daily/factors/monthly/
```

2. 继承 `DailyFactor`。

3. 定义：

```python
name = "new_factor"
category = "..."
description = "..."
required_data = ["daily"]
lookback_days = 20
```

4. 实现：

```python
def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
    ...
```

5. 返回至少包含：

```text
trade_date
ts_code
factor_value
```

6. 添加测试：

- 因子输出列完整
- 无未来函数
- lookback 生效
- 空数据/缺列行为清晰

7. 验证注册：

```bash
pixi run python -c "from daily.factors.registry import list_factors; print(list_factors())"
```

## 28. 新增日内因子指南

1. 在 `intraday/factors/demo/` 或 `intraday/factors/technical/` 新增文件。
2. 继承日内因子基类。
3. 声明 `name`、`required_data`、`lookback_bars`。
4. 使用 `IntradayDataContext` 取分钟数据。
5. 返回 `trade_time`、`ts_code`、`factor_value`。
6. 添加测试覆盖 registry、compute、预处理和 IC。

## 29. 新增 CLI 指南

新增 CLI 时遵守：

- 放在 `scripts/`
- 使用 argparse
- 不写 `sys.path` hack
- 依赖 editable install
- 日志使用 `common.logger`
- 输出路径从 `config.settings` 取
- 需要可复现记录时用 `common.experiment.run_experiment`
- 添加脚本级测试或流程测试

## 30. 常见问题与排查

### 30.1 `TUSHARE_TOKEN` 缺失

现象：

```text
请设置 TUSHARE_TOKEN 环境变量
```

处理：

- 检查 `.env`
- 检查环境变量
- 确认不是在离线测试中误触发真实拉取

### 30.2 mypy database is locked

处理：

```bash
pixi run typecheck
```

不要直接运行旧式 `mypy` 命令。项目任务已禁用 sqlite cache。

### 30.3 coverage database is locked

处理：

```bash
pixi run coverage
```

不要直接使用仓库根目录 `.coverage`。项目任务会使用临时文件。

### 30.4 指数成分股加载失败

`get_universe("...", "csi500")` 可能因 Tushare 网络或权限失败而降级为全 A 股。研究报告里应检查股票池数量是否符合预期。

### 30.5 月频因子为空

常见原因：

- `daily_basic` 未拉取
- `finance` 未拉取
- PIT 可见日期不足
- 研究区间太短

### 30.6 walk-forward 样本不足

如果窗口参数要求的历史观察期 / 未来验证期超过数据长度，会输出 `insufficient_data`。这不是程序错误。

## 31. 当前边界

1. Tick 级研究当前不保留正式代码包。
2. 日内模块评估管线已具备，但真实分钟数据覆盖依赖本地 Tushare 拉取结果。
3. `research/combination/` 是研究工具，不是生产组合优化模块。
4. `common/loader.py` 的真实网络拉取不在 CI 中验证。
5. coverage/typecheck 仍是渐进式范围，没有覆盖全部模块。
6. 股票池指数成分拉取失败会降级，应在严肃研究前检查 universe 明细。
7. 平方根冲击模型已经接入 trailing ADV，但仍是简化模型，不代表真实市场冲击。

## 32. 推荐维护顺序

短期：

1. 稳定低频数据审计与 loader mock 测试。
2. 增加真实数据 smoke 的手动命令，不放入默认 CI。
3. 为 `common/loader.py` 增加更多 mock Tushare 测试。
4. 为 `research/combination/` 的样本内边界增加更醒目的报告标识。

中期：

1. 完善 `daily_basic` / finance 数据完整性报告。
2. 增加 universe 快照落盘，便于复现实验股票池。
3. 把日频和日内报告元数据格式统一。
4. 为优化器增加更多失败模式测试。

长期：

1. 设计 Tick 数据模块。
2. 设计独立数据供应商 adapter。
3. 引入更真实的成交和冲击模型。
4. 建立研究结果数据库或实验索引。

## 33. 一句话主链路

FactorZen 的主链路是：

```text
Tushare raw data
→ common/storage Parquet cache
→ universe snapshot
→ FactorDataContext
→ factor.compute
→ preprocessing / neutralization
→ forward returns
→ IC + backtest + turnover
→ quality report + walk-forward summary
→ experiment manifest
→ HTML Tear Sheet
```
