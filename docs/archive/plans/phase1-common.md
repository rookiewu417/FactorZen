# Phase 1: common/ 共享基础设施

> **状态**: ✅ 已完成 (2026-05-13)
> **创建时间**: 2026-05-13
> **预估工期**: 2-3 个 session

---

## A. 一期目标

搭建 `common/` 共享基础设施 + pixi 依赖就位，为 LFT/MFT 因子研究提供数据底座。

**核心交付物**：
- 完整的 pixi 依赖环境（install 即用）
- 8 个 common/ 模块：路径配置、Tushare 配置、日志、交易日历、Parquet 存储、数据拉取桥接、股票池过滤
- 全链路 `pl.DataFrame` / `pl.LazyFrame` 数据流
- Hive 分区 Parquet 存储体系

---

## B. 任务清单

### T1: 更新 pixi.toml（添加全部依赖）

- **文件**: `E:\code\量化研究\因子研究\pixi.toml`
- **描述**: 在现有空骨架中填入完整依赖声明
- **输入**: 现有 `pixi.toml`（仅 workspace 元信息）
- **输出**: 含完整 `[dependencies]` + `[dev-dependencies]` + `[tasks]` 的 pixi.toml
- **关键决策**:
  - Python 版本锁定 `>=3.10,<3.13`
  - Polars >= 1.0.0（惰性 API / Hive 分区谓词下推）
  - Tushare >= 1.4.0
  - pyarrow 作为 Parquet 后端
- **tasks 定义**:
  - `pixi run lint` → `ruff check .`
  - `pixi run test` → `pytest tests/ -v`
  - `pixi run lab` → `jupyter lab`
  - `pixi run smoke` → `python -c "import polars; import tushare; print('ok')"`
- **验收标准**: `pixi install` 无报错；`pixi run smoke` 输出 `ok`

```toml
[workspace]
authors = ["吴一凡 <1007372080@qq.com>"]
channels = ["conda-forge"]
name = "因子研究"
platforms = ["win-64"]
version = "0.1.0"

[tasks]
lint = "ruff check ."
test = "pytest tests/ -v"
lab = "jupyter lab"
smoke = "python -c \"import polars; import tushare; print('ok')\""

[dependencies]
python = ">=3.10,<3.13"
polars = ">=1.0.0"
pyarrow = ">=14.0.0"
tushare = ">=1.4.0"
pandas = ">=2.0.0"
numpy = ">=1.24.0"
scipy = ">=1.11.0"
statsmodels = ">=0.14.0"
matplotlib = ">=3.7.0"
seaborn = ">=0.12.0"
plotly = ">=5.17.0"
jinja2 = ">=3.1.0"
pyyaml = ">=6.0"

[dev-dependencies]
pytest = ">=7.0.0"
ruff = ">=0.3.0"
jupyterlab = ">=4.0.0"
```

- [x] T1: pixi.toml 依赖就位

---

### T2: config/settings.py（全局路径配置）

- **文件**: `E:\code\量化研究\因子研究\config\settings.py`
- **描述**: 项目全局路径常量，所有模块通过 `from config.settings import *` 引用
- **输入**: 无（硬编码）
- **输出**: 一个 `class Paths` 或模块级常量
- **关键决策**:
  - 使用 `pathlib.Path`，不裸写字符串路径
  - 项目根目录自动探测（`Path(__file__).parent.parent`）
  - 不含 `data/raw/` 等运行时才会写入的路径的自动创建逻辑（由 storage.py 负责）
- **暴露符号**:

```python
# config/settings.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 因子研究/
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_RAW_DAILY = DATA_RAW / "daily"
DATA_RAW_FINANCE = DATA_RAW / "finance"
DATA_RAW_MINUTE = DATA_RAW / "minute"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_CACHE = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "output"
OUTPUT_LFT = OUTPUT_DIR / "lft"
OUTPUT_MFT = OUTPUT_DIR / "mft"
SCRIPTS_DIR = ROOT / "scripts"
NOTEBOOKS_DIR = ROOT / "notebooks"
TESTS_DIR = ROOT / "tests"
```

- **验收标准**: `from config.settings import ROOT; assert ROOT.name == "因子研究"`

- [x] T2: config/settings.py 完成

---

### T3: config/tushare_config.py（Token / 限流）

- **文件**: `E:\code\量化研究\因子研究\config\tushare_config.py`
- **描述**: Tushare 连接配置，Token 从环境变量读取，含积分级别与限流参数
- **输入**: 环境变量 `TUSHARE_TOKEN`
- **输出**: 模块级常量 `TUSHARE_TOKEN`、`POINTS`、`RATE_LIMIT`
- **关键决策**:
  - Token 不可为空；缺失时抛 `RuntimeError("请设置 TUSHARE_TOKEN 环境变量")`
  - 积分级别硬编码默认值 `POINTS = 2000`，可通过环境变量 `TUSHARE_POINTS` 覆盖
  - 限流：每秒最大请求数 `MAX_RPS = 5`（保守值）
- **暴露符号**:

```python
TUSHARE_TOKEN: str       # 从 os.environ["TUSHARE_TOKEN"] 读取
POINTS: int               # 默认 2000
MAX_RPS: int              # 默认 5
MAX_RETRIES: int          # 默认 3
RETRY_DELAY: float        # 默认 1.0
BATCH_SIZE: int           # 默认 5000
```

- **验收标准**: `from config.tushare_config import TUSHARE_TOKEN` 不抛异常（若环境变量已设置）

- [x] T3: config/tushare_config.py 完成

---

### T4: common/logger.py（日志系统）

- **文件**: `E:\code\量化研究\因子研究\common\logger.py`
- **描述**: 标准 logging 配置，控制台 + 文件双输出
- **输入**: 无（通过 `get_logger(__name__)` 获取）
- **输出**: `logging.Logger` 实例
- **关键决策**:
  - 日志文件自动创建于 `output/logs/`，按日期轮转（`factor_research_YYYYMMDD.log`）
  - 控制台级别 `INFO`，文件级别 `DEBUG`
  - 统一格式：`%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`
  - 单例模式：`setup_logging()` 只执行一次
- **暴露函数**:

```python
def setup_logging(log_dir: Path | None = None) -> None:
    """初始化日志系统，幂等"""
    ...

def get_logger(name: str) -> logging.Logger:
    """获取命名 logger，自动继承全局配置"""
    ...
```

- **验收标准**: `get_logger("test").info("hello")` 控制台输出格式正确

- [x] T4: common/logger.py 完成
- [x] T5: common/calendar.py 完成
- [x] T6: common/storage.py 完成

---

### T7: common/loader.py（Tushare 数据拉取桥接）

- **文件**: `E:\code\量化研究\因子研究\common\loader.py`
- **描述**: Tushare 数据拉取桥接层，负责 pandas → polars 转换、分段拉取、限流、重试、缓存
- **输入**: Tushare API（通过 `config/tushare_config.py`）
- **输出**: `pl.DataFrame`，已写入 Parquet
- **关键决策**:
  - 这是整个项目中**唯一**与 Tushare 直接交互的模块
  - pandas → polars 转换在这里做一次（`pl.from_pandas()`）
  - 所有 fetch 函数先检查缓存，命中则跳过拉取
  - 分段拉取：日线按年分段，分钟线按月分段，财报按季度分段
  - 重试仅对网络错误（`ConnectionError`, `Timeout`），不对参数错误重试
  - 使用 `config/tushare_config.py` 中的限流参数
- **暴露函数**:

```python
def init_tushare() -> ts.pro_api:
    """初始化 Tushare Pro API 客户端"""
    ...

def fetch_daily(
    start: str,              # YYYYMMDD
    end: str,
    ts_codes: list[str] | None = None,  # None = 全市场
) -> pl.DataFrame:
    """
    拉取日线行情数据 (daily).
    分段策略: 按年分段，自动缓存到 data/raw/daily/
    返回: trade_date, ts_code, open, high, low, close, pre_close, change, pct_chg, vol, amount
    """
    ...

def fetch_daily_basic(
    start: str,
    end: str,
    ts_codes: list[str] | None = None,
) -> pl.DataFrame:
    """
    拉取每日估值指标 (daily_basic).
    返回: trade_date, ts_code, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_mv, circ_mv
    """
    ...

def fetch_minute(
    ts_code: str,
    freq: str,               # "1min" | "5min" | "15min" | "30min" | "60min"
    start: str,
    end: str,
) -> pl.DataFrame:
    """
    拉取分钟线 (stk_mins).
    逐股票拉取，按月分段.
    返回: ts_code, trade_time, open, high, low, close, vol, amount
    """
    ...

def fetch_finance(
    api_name: str,           # "income" | "balancesheet" | "cashflow" | "fina_indicator" | "forecast" | "express"
    start: str,              # YYYYMMDD
    end: str,
    ts_codes: list[str] | None = None,
    fields: str | None = None,
) -> pl.DataFrame:
    """
    拉取财务报表数据.
    分段策略: 按季度/年度分段.
    """
    ...

def fetch_stock_basic() -> pl.DataFrame:
    """
    拉取全量股票基本信息 (stock_basic).
    缓存到 data/cache/stock_basic.parquet
    返回: ts_code, symbol, name, area, industry, market, list_date
    """
    ...

def fetch_trade_cal(start: str, end: str) -> pl.DataFrame:
    """
    拉取交易日历 (trade_cal)，内部使用，供 calendar.py 调用.
    """
    ...
```

- **验收标准**:
  - `fetch_trade_cal("20260101", "20260513")` 返回包含 cal_date, is_open 的 DataFrame
  - 第二次调用同一区间命中缓存
  - `fetch_daily("20260101", "20260131", ts_codes=["000001.SZ"])` 返回 1 月的日线

- [x] T7: common/loader.py 完成

---

### T8: common/universe.py（股票池过滤）

- **文件**: `E:\code\量化研究\因子研究\common\universe.py`
- **描述**: 股票池构建与过滤，支持多种预设 universe + 自定义过滤链
- **输入**: `stock_basic` 缓存 + `daily` 行情数据
- **输出**: `pl.DataFrame`（符合条件的 ts_code 列表）
- **关键决策**:
  - 索引成分股先硬编码一个初始列表（后续 Phase 2 从 Tushare index_member 动态更新）
  - 过滤器可组合链式调用
  - 每个过滤器返回过滤后的 DataFrame + 剔除原因统计
- **暴露函数**:

```python
def get_stock_basic(use_cache: bool = True) -> pl.DataFrame:
    """
    获取全量 A 股股票列表.
    缓存到 data/cache/stock_basic.parquet
    """
    ...

def get_universe(
    date: str,                # YYYYMMDD
    universe_name: str,       # "all_a" | "csi300" | "csi500" | "csi800" | "lft_default" | "mft_default"
) -> pl.DataFrame:
    """
    获取指定日期的股票池.
    
    预设 universe:
    - all_a: 全 A 股（剔除退市）
    - csi300: 沪深 300 成分股
    - csi500: 中证 500 成分股
    - csi800: 沪深 300 + 中证 500
    - lft_default: 
        1. 全 A 
        2. 剔除 ST
        3. 剔除上市不足 250 个交易日（次新）
        4. 剔除停牌
        5. 剔除涨跌停
    - mft_default: 
        1. lft_default
        2. 剔除日成交额 < 1000 万（流动性过滤）
    """
    ...

def create_universe(
    date: str,
    base: str = "all_a",        # 基础池
    filters: list[str] | None = None,  # ["st", "new_listing", "suspended", "limit", "liquidity:10000000"]
    **filter_kwargs,
) -> pl.DataFrame:
    """
    自定义股票池: 指定基础 pool + 过滤链.
    
    filter_kwargs 支持:
    - min_days: 次新股最少上市天数 (默认 250)
    - min_amount: 最低日成交额 (默认 10_000_000)
    """
    ...

# 单个过滤器（可独立使用）
def filter_st(stocks: pl.DataFrame, date: str) -> pl.DataFrame:
    """剔除 ST / *ST / PT"""
    ...

def filter_new_listing(stocks: pl.DataFrame, date: str, min_days: int = 250) -> pl.DataFrame:
    """剔除上市不足 min_days 个交易日的次新股"""
    ...

def filter_suspended(stocks: pl.DataFrame, date: str) -> pl.DataFrame:
    """剔除当日停牌（基于日线 volume > 0 或 turnover > 0）"""
    ...

def filter_limit(stocks: pl.DataFrame, date: str) -> pl.DataFrame:
    """剔除当日涨跌停（pct_chg ≈ ±10% 或 ±20%）"""
    ...

def filter_liquidity(stocks: pl.DataFrame, date: str, min_amount: float = 10_000_000) -> pl.DataFrame:
    """剔除日成交额低于阈值的股票（MFT 用）"""
    ...

def get_index_members(
    index_code: str,           # "000300.SH" | "000905.SH" | etc.
    date: str,
    use_cache: bool = True,
) -> pl.DataFrame:
    """获取指数成分股（预留，Phase 2 从 Tushare 动态拉取）"""
    ...
```

- **验收标准**:
  - `get_universe("20260513", "all_a")` 返回 5000+ 只股票
  - `get_universe("20260513", "lft_default")` 数量 < all_a（已过滤）
  - `get_universe("20260513", "mft_default")` 数量 <= lft_default
  - 每个 `filter_*` 函数可独立调用且返回有意义的统计信息

- [x] T8: common/universe.py 完成

---

## C. 技术规范

| 规范项 | 要求 |
|--------|------|
| **数据格式** | 全链路 `pl.DataFrame` / `pl.LazyFrame`，不裸用 pandas |
| **Pandas → Polars 桥接** | 仅在 `common/loader.py` 做一次 `pl.from_pandas()` |
| **Parquet 分区** | Hive 分区 `data/raw/{data_type}/year={YYYY}/month={MM}/`，支持谓词下推 |
| **路径管理** | 统一用 `config/settings.py` 中的 `pathlib.Path` |
| **缓存策略** | 交易日历 (`trade_cal.parquet`) 和股票列表 (`stock_basic.parquet`) 缓存到 `data/cache/`，7 天过期自动刷新 |
| **错误处理分层** | 网络错误 → 重试（最多 3 次）；参数错误 → 立即失败（不重试）；权限不足 → 提示用户升级 Tushare 积分 |
| **日志规范** | console: INFO → WARNING → ERROR；file: DEBUG 全量记录 |
| **Type Hints** | 所有公共函数必须有完整的类型标注 |

---

## D. 验收标准

### D1. 环境验收 ✅

- [x] `pixi install` 成功，无依赖冲突
- [x] `pixi run smoke` 输出 `ok`（Polars + Tushare 可 import）

### D2. 模块验收 ✅

- [x] `pixi.toml` 依赖声明完整（13 核心 + 3 dev 放入 pypi-dependencies）
- [x] 每个 `common/*.py` 可独立 `import`
- [x] `from config.settings import ROOT` → ROOT 正常
- [x] `from config.tushare_config import TUSHARE_TOKEN` → 从 .env 自动加载
- [x] `from common.logger import get_logger` → 日志双输出
- [x] `common.calendar.is_trade_date('20260501')` → `False`（劳动节）
- [x] `common.calendar.is_trade_date('20260511')` → `True`
- [x] `common.storage.save_parquet()` + `load_parquet()` → 分区读写正常
- [x] `common.loader.fetch_daily('20260501','20260513',['000001.SZ'])` → 6 行 ✅
- [x] `common.universe.get_universe('20260513', 'all_a')` → 5515 只股票 ✅

### D3. 数据流验收 ✅

- [x] loader.py 拉取 → storage.py 写入 Parquet → partition_exists 验证缓存命中
- [x] 缓存命中后不触发 Tushare 网络请求（calendar 缓存已验证）

### D4. 代码质量

- [x] 所有公共函数有 docstring（Google style）✅
- [x] 所有公共函数有 type hints ✅
- [ ] `ruff check .` 待后续运行

---

## E. 执行顺序

```
T1 (pixi.toml) ──> T2 (settings.py)
T1 (pixi.toml) ──> T3 (tushare_config.py)
                    T2, T3 ──> T4 (logger.py)
                    T2, T3 ──> T5 (calendar.py)
                    T2, T3 ──> T6 (storage.py)
                    T2, T3 ──> T7 (loader.py)  [依赖 T6]
                                T7 ──> T8 (universe.py)
```

T1 可先行。T2/T3 并行的。T4/T5/T6 可在 T2/T3 完成后并行。T7 依赖 T6。T8 依赖 T7。

---

## F. 风险与假设

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| Tushare Token 未配置 | 高 | T3 中有明确的错误提示 |
| Tushare 积分不足导致部分接口不可用 | 中 | loader.py 做接口权限检测，降级提示 |
| Windows 路径兼容性 | 中 | pathlib.Path 自动处理 |
| pixi 依赖冲突（Windows） | 低 | conda-forge 渠道成熟稳定 |
| 分段拉取超时 | 低 | 分段策略 + 重试机制 |
