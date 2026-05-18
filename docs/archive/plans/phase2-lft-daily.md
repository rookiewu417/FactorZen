# Phase 2: LFT 日频因子 + 评估管线

> **状态**: 🔴 待实施
> **创建时间**: 2026-05-13
> **预估工期**: 3-4 个 session
> **前置条件**: Phase 1（common/ 基础设施）✅ 已完成

---

## A. 二期目标

实现 LFT 日频因子的完整链路：**因子计算 → 预处理 → 单因子评估 → 报告输出**。

产出第一个可用的单因子评估报告（IC 分析 + 分层回测 + 换手率），为后续因子筛选、组合优化奠定基础。

### 核心交付物

| 模块 | 内容 | 状态 |
|------|------|------|
| `lft/factors/base.py` | LFTFactor 抽象基类 | 🔴 |
| `lft/data/context.py` | FactorDataContext 数据上下文 | 🔴 |
| `lft/factors/daily/` | 4 个日频因子 | 🔴 |
| `lft/factors/registry.py` | 因子自动发现与注册 | 🔴 |
| `lft/preprocessing/` | 去极值 → 缺失值 → 标准化 → 中性化 | 🔴 |
| `lft/evaluation/` | IC 分析 → 分层回测 → 换手率 → 相关性 | 🔴 |
| `scripts/run_lft_single.py` | 单因子一键评估入口 | 🔴 |

### 不涉及范围

- ❌ MFT 中频因子模块
- ❌ 因子组合 / 合成（四期）
- ❌ 周频 / 月频因子（三期）
- ❌ 复杂中性化（简单截面 OLS 即可）
- ❌ HTML 报告模板（五期）
- ❌ 机器学习因子

---

## B. 任务清单

### B1. 因子框架层（3 个任务）

---

#### T1: `lft/factors/base.py` — LFTFactor 抽象基类

- **文件**: `lft/factors/base.py`
- **描述**: 定义所有 LFT 因子的统一接口契约
- **输入**: 无（纯抽象定义）
- **输出**: `LFTFactor` 抽象类

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lft.data.context import FactorDataContext

class LFTFactor(ABC):
    """LFT 因子抽象基类。
    
    所有因子必须继承此类并实现 compute() 方法。
    类属性声明因子的元信息，供注册中心和管线使用。
    """
    
    # ── 子类必须覆盖的类属性 ──
    name: str                  # 因子唯一标识，如 "momentum_20d"
    category: str              # 因子类别："daily" | "weekly" | "monthly"
    frequency: str = "daily"   # 数据频率
    required_data: list[str] = ["daily"]  # 需要的数据类型
    lookback_days: int = 20    # 因子计算所需回看天数（不含截面日）
    
    # ── 可选类属性 ──
    description: str = ""      # 因子描述（用于报告）
    author: str = ""           # 因子作者
    
    @abstractmethod
    def compute(self, ctx: "FactorDataContext") -> pl.DataFrame:
        """计算因子值。
        
        Args:
            ctx: 数据上下文，提供按需加载的数据访问
            
        Returns:
            pl.DataFrame，必须包含列: trade_date, ts_code, factor_value
        """
        ...
    
    def validate(self, result: pl.DataFrame) -> dict:
        """校验因子计算结果。
        
        Args:
            result: compute() 的输出
            
        Returns:
            dict，包含:
            - coverage: 覆盖率（有因子值的股票占比）
            - n_stocks: 平均股票数
            - n_dates: 日期数
            - null_count: 空值总数
            - inf_count: 无穷值总数
            - warnings: 警告列表
        """
        ...
```

- **关键决策**:
  - `required_data` 声明因子需要的数据类型，`FactorDataContext` 据此自动加载
  - `lookback_days` 让 DataContext 自动扩展日期范围（例如 `[start, end]` 扩展为 `[prev_trade_date(start, lookback_days), end]`）
  - `validate()` 提供默认实现，检查覆盖率 / 空值 / 无穷值
  - Type hints 使用 `TYPE_CHECKING` 避免循环导入

- **验收标准**:
  - `from lft.factors.base import LFTFactor` 无报错
  - 抽象类强制子类实现 `compute()`
  - `validate()` 默认实现在正常 DataFrame 上返回合理的 dict

---

#### T2: `lft/data/context.py` — FactorDataContext

- **文件**: `lft/data/context.py`
- **描述**: 因子计算的数据上下文，根据因子的 `required_data` 声明自动加载对应数据，屏蔽底层存储细节
- **输入**: `start` / `end` 日期 + `LFTFactor` 实例（或 `required_data` + `lookback_days`）
- **输出**: 提供 `ctx.daily`、`ctx.daily_basic` 等属性的惰性 DataFrame 访问器

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FactorDataContext:
    """因子计算数据上下文。
    
    根据因子的 required_data 声明，自动从 Parquet 存储中惰性加载所需数据。
    日期范围自动根据 lookback_days 向前扩展。
    
    使用示例:
        ctx = FactorDataContext(
            start="20240101",
            end="20241231",
            required_data=["daily", "daily_basic"],
            lookback_days=20,
        )
        # 惰性访问
        daily = ctx.daily  # pl.LazyFrame，尚未实际加载
    
    Attributes:
        daily: 日线行情 LazyFrame（如有声明）
        daily_basic: 每日估值 LazyFrame（如有声明）
        start: 请求起始日期
        end: 请求截止日期
        expanded_start: 扩展后的起始日期（考虑了 lookback_days）
    """
    
    start: str                      # "YYYYMMDD"，请求起始日期
    end: str                        # "YYYYMMDD"，请求截止日期
    required_data: list[str] = field(default_factory=lambda: ["daily"])
    lookback_days: int = 20
    
    # 私有属性
    _daily: Optional[pl.LazyFrame] = field(default=None, repr=False)
    _daily_basic: Optional[pl.LazyFrame] = field(default=None, repr=False)
    
    @property
    def expanded_start(self) -> str:
        """返回考虑 lookback_days 后扩展的起始日期。"""
        ...
    
    @property
    def daily(self) -> pl.LazyFrame:
        """惰性加载日线行情数据（含扩展区间）。"""
        ...
    
    @property
    def daily_basic(self) -> pl.LazyFrame:
        """惰性加载每日估值数据（含扩展区间）。"""
        ...
    
    def load_all(self) -> None:
        """强制加载所有声明的数据（用于调试/预热）。"""
        ...
    
    def get_trade_dates(self) -> list[str]:
        """获取 [expanded_start, end] 区间内所有交易日列表。"""
        ...
```

- **关键决策**:
  - 所有数据属性返回 `pl.LazyFrame`（惰性），实际计算在因子 `compute()` 中 `.collect()` 时才触发
  - 日期扩展逻辑：`expanded_start = prev_trade_date(start, lookback_days)`
  - 依赖 `common.storage.load_parquet()` 的惰性加载能力
  - 依赖 `common.calendar.prev_trade_date()` 做日期回推
  - 初始版本仅支持 `daily` 和 `daily_basic`，后续可扩展 `finance` 等

- **验收标准**:
  - `ctx.expanded_start` 比 `ctx.start` 早至少 `lookback_days` 个交易日
  - `ctx.daily` 返回 `pl.LazyFrame`，调用 `.collect().columns` 包含 `['trade_date', 'ts_code', 'close', 'vol', 'amount']`
  - 不声明 `daily_basic` 时，访问 `ctx.daily_basic` 抛 `ValueError`
  - `ctx.load_all()` 执行后所有属性变为具体 DataFrame

---

#### T7: `lft/factors/registry.py` — 因子注册中心

- **文件**: `lft/factors/registry.py`
- **描述**: 自动发现 `lft/factors/daily/` 下所有 `LFTFactor` 子类，提供按名查找和全量列举
- **输入**: Python 包搜索路径（`lft.factors.daily`）
- **输出**: 因子字典 `{name: LFTFactor class}`

```python
from typing import Type

# 全局注册表
_FACTOR_REGISTRY: dict[str, Type[LFTFactor]] = {}

def discover_factors(package: str = "lft.factors.daily") -> dict[str, Type[LFTFactor]]:
    """递归发现指定包下所有 LFTFactor 子类。
    
    使用 importlib 遍历包内模块，收集所有 LFTFactor 的具体子类。
    结果缓存到 _FACTOR_REGISTRY。
    
    Args:
        package: 包路径字符串，默认 "lft.factors.daily"
        
    Returns:
        dict[str, Type[LFTFactor]]: {factor_name: FactorClass}
    """
    ...

def get_factor(name: str) -> Type[LFTFactor]:
    """按名称获取因子类。
    
    Args:
        name: 因子名称，如 "momentum_20d"
        
    Returns:
        LFTFactor 子类
        
    Raises:
        KeyError: 因子未注册
    """
    ...

def list_factors(category: str | None = None) -> list[str]:
    """列出所有已注册因子名称。
    
    Args:
        category: 过滤类别，None 表示全部
        
    Returns:
        因子名称列表
    """
    ...

def register_factor(cls: Type[LFTFactor]) -> Type[LFTFactor]:
    """手动注册因子（可选，通常用 discover_factors 自动发现）。
    
    也可用作装饰器:
        @register_factor
        class Momentum20D(LFTFactor):
            ...
    """
    ...
```

- **关键决策**:
  - 使用 `importlib.import_module()` + `pkgutil.iter_modules()` 自动发现
  - 扫描路径：`lft/factors/daily/*.py` 等
  - 每个因子模块只需在模块末尾实例化或通过 `register_factor` 装饰器注册
  - 建议因子实现文件末尾加 `Momentum20D()` 触发注册（模块级实例化）

- **依赖**: T1（LFTFactor 基类）

- **验收标准**:
  - `discover_factors()` 返回包含已实现因子的 dict
  - `get_factor("momentum_20d")` 返回正确的类
  - `get_factor("nonexistent")` 抛 `KeyError`
  - `list_factors("daily")` 返回日频因子列表

---

### B2. 日频因子实现（4 个任务，可并行）

---

#### T3: `lft/factors/daily/momentum.py` — 20 日动量因子

- **文件**: `lft/factors/daily/momentum.py`
- **描述**: 经典动量因子：当前收盘价相对 20 个交易日前收盘价的涨跌幅
- **公式**: `momentum_20d = close(t) / close(t-20) - 1`
- **输入**: `ctx.daily`（自动含扩展区间）
- **输出**: `pl.DataFrame` 三列 `[trade_date, ts_code, factor_value]`

```python
class Momentum20D(LFTFactor):
    name = "momentum_20d"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 20
    description = "20日动量：close(t) / close(t-20) - 1"
    
    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 惰性：从 ctx.daily LazyFrame 开始
        # 1. 按 ts_code 分组，shift(20) 取 20 天前收盘价
        # 2. factor_value = close / shifted_close - 1
        # 3. 过滤掉 lookback 期间的 NaN (前 20 行每只股票)
        # 4. .collect() 触发计算
        # 5. 只返回 [ctx.start, ctx.end] 区间（去掉扩展部分）
        ...
```

- **Polars 惰性链式操作示意**:
  ```
  ctx.daily
    → .sort(["ts_code", "trade_date"])
    → .with_columns(close_lag20 = pl.col("close").shift(20).over("ts_code"))
    → .with_columns(factor_value = pl.col("close") / pl.col("close_lag20") - 1)
    → .filter(pl.col("trade_date") >= start_dt)    # 去掉扩展区间
    → .select(["trade_date", "ts_code", "factor_value"])
    → .collect()
  ```

- **依赖**: T1（LFTFactor）+ T2（FactorDataContext）

- **验收标准**:
  - `factor.compute(ctx)` 返回非空 DataFrame（3 列）
  - `factor_value` 范围合理（大多在 ±0.5 之间）
  - 前 20 个交易日无值（NaN，已被过滤）
  - `factor.validate(result)["coverage"] > 0.8`

---

#### T4: `lft/factors/daily/reversal.py` — 5 日反转因子

- **文件**: `lft/factors/daily/reversal.py`
- **描述**: 短期反转因子（A 股显著有效）：负的 5 日收益率
- **公式**: `reversal_5d = -(close(t) / close(t-5) - 1)`
- **输入**: `ctx.daily`
- **输出**: `pl.DataFrame` 三列

```python
class Reversal5D(LFTFactor):
    name = "reversal_5d"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 5
    description = "5日反转：-(close(t)/close(t-5) - 1)，负号表示反转信号"
    
    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 同 Momentum20D 模式，lookback=5
        # factor_value = -(close / close.shift(5).over("ts_code") - 1)
        ...
```

- **依赖**: T1 + T2

- **验收标准**: 同上（T3），`lookback_days=5` 正确生效

---

#### T5: `lft/factors/daily/volatility.py` — 20 日波动率因子

- **文件**: `lft/factors/daily/volatility.py`
- **描述**: 历史已实现波动率：20 日对数收益率的标准差
- **公式**: `volatility_20d = std(log_return) over rolling 20 days`
- **输入**: `ctx.daily`
- **输出**: `pl.DataFrame` 三列

```python
class Volatility20D(LFTFactor):
    name = "volatility_20d"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 21  # 需要 21 天数据（20 个收益率）
    description = "20日已实现波动率：std(log_return) over 20 days"
    
    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 1. log_return = log(close / close.shift(1))
        # 2. rolling_std over 20 days (window=20)
        # 3. 过滤扩展区间
        ...
```

- **Polars 关键操作**: `pl.col("close").log().diff().rolling_std(window_size=20).over("ts_code")`

- **依赖**: T1 + T2

- **验收标准**: 同上

---

#### T6: `lft/factors/daily/turnover.py` — 5 日换手率因子

- **文件**: `lft/factors/daily/turnover.py`
- **描述**: 5 日平均换手率（流动性 / 情绪代理变量）
- **公式**: `turnover_5d = avg(vol / float_shares) over 5 days`，近似用 `avg(turnover_rate) over 5 days`，若无则直接用 `avg(vol)`
- **输入**: `ctx.daily`（`vol` 列）或 `ctx.daily_basic`（如有 `turnover_rate` 列）
- **输出**: `pl.DataFrame` 三列

```python
class Turnover5D(LFTFactor):
    name = "turnover_5d"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]  # 仅用日线 vol 列，不强制 daily_basic
    lookback_days = 5
    description = "5日平均换手率：avg(vol) over 5 days"
    
    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 方案 A（简单）: factor_value = vol.rolling_mean(window_size=5).over("ts_code")
        # 方案 B（精确）: 需要 daily_basic.circ_mv 计算 turnover_rate = vol / circ_mv，再 rolling_mean
        # 二期先用方案 A
        ...
```

- **关键决策**:
  - 先用方案 A（直接用 `vol` 做 5 日均值），快速跑通链路
  - 如果 `daily_basic` 数据就绪且 `circ_mv` 可用，可升级为方案 B
  - 方案 A 的 `vol` 需要按 `ts_code` 截面标准化后使用（否则不同市值股票不可比），标准化在预处理阶段统一做

- **依赖**: T1 + T2

- **验收标准**: 同上

---

### B3. 预处理模块（5 个任务，T8-T11 可并行）

所有预处理函数接受 Polars DataFrame（含 `trade_date, ts_code, factor_value`），返回同结构的 DataFrame。

---

#### T8: `lft/preprocessing/outlier.py` — 去极值（MAD 法）

- **文件**: `lft/preprocessing/outlier.py`
- **描述**: 截面 MAD（Median Absolute Deviation）3σ 截尾
- **输入**: `pl.DataFrame` with `[trade_date, ts_code, factor_value]`
- **输出**: 同结构 DataFrame，极端值被截尾

```python
def mad_clip(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    n_sigma: float = 3.0,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """MAD 法去极值。
    
    每日截面:
        中位数 median = median(factor_value)
        MAD = median(|factor_value - median|)
        上限 = median + n_sigma * 1.4826 * MAD
        下限 = median - n_sigma * 1.4826 * MAD
        超出范围的值截尾到边界值
    
    Args:
        df: 因子值 DataFrame
        factor_col: 因子值列名
        n_sigma: σ 倍数，默认 3.0
        date_col: 日期列名
        
    Returns:
        去极值后的 DataFrame
    """
    ...
```

- **关键 Polars 操作**: `pl.col("factor_value").median().over("trade_date")`  + `.clip()`

- **验收标准**:
  - 极端值被截尾到上下边界
  - 中位数不变（MAD 特性）
  - 对 1000+ 只股票的截面，处理时间 < 0.5s

---

#### T9: `lft/preprocessing/missing.py` — 缺失值处理

- **文件**: `lft/preprocessing/missing.py`
- **描述**: 截面中位数填充缺失值
- **输入**: `pl.DataFrame` with `[trade_date, ts_code, factor_value]`
- **输出**: 同结构 DataFrame，缺失值被填充

```python
def fill_cross_sectional_median(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """截面中位数填充。
    
    每日截面:
        缺失值 → 当日该因子的截面中位数
        如果当日所有值都缺失 → 填 0（极端情况）
    
    Args:
        df: 因子值 DataFrame
        factor_col: 因子值列名
        date_col: 日期列名
        
    Returns:
        填充后的 DataFrame
    """
    ...
```

- **验收标准**:
  - 缺失值被填充为当日截面中位数
  - 全为缺失的日期 → 填 0
  - 处理前后行数不变

---

#### T10: `lft/preprocessing/normalizer.py` — 标准化

- **文件**: `lft/preprocessing/normalizer.py`
- **描述**: 截面 Z-score 标准化（每日截面做）
- **公式**: `z = (x - mean(x)) / std(x)` per cross-section

```python
def cross_sectional_zscore(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    date_col: str = "trade_date",
    suffix: str = "_z",
) -> pl.DataFrame:
    """截面 Z-score 标准化。
    
    每日截面:
        z = (factor_value - mean) / std
        
    std=0 时（如当日所有值相同）→ z=0
    
    Args:
        df: 因子值 DataFrame
        factor_col: 输入因子值列名
        date_col: 日期列名
        suffix: 输出列后缀，生成新列而不覆盖原列
        
    Returns:
        DataFrame，新增 {factor_col}{suffix} 列
    """
    ...
```

- **关键决策**: 不覆盖原始 `factor_value`，新增 `factor_value_z` 列（方便调试对比）

- **验收标准**:
  - 每日期望值 ≈ 0，标准差 ≈ 1
  - 原 `factor_value` 列保留不变
  - `std=0` 时不抛异常

---

#### T11: `lft/preprocessing/neutralizer.py` — 中性化

- **文件**: `lft/preprocessing/neutralizer.py`
- **描述**: 行业哑变量 + 对数市值截面 OLS 回归，取残差
- **输入**: `pl.DataFrame` + 行业数据 + 市值数据
- **输出**: 同结构 DataFrame，新增 `factor_value_neutral` 列

```python
def neutralize_ols(
    df: pl.DataFrame,
    factor_col: str = "factor_value_z",       # 通常先标准化再中性化
    industry_col: str = "industry",
    market_cap_col: str = "ln_cap",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """行业 + 对数市值截面 OLS 中性化。
    
    每日截面:
        factor_value ~ industry_dummies + ln(market_cap)
        残差 = factor_value - fitted_value
        残差再截面 Z-score → factor_value_neutral
    
    行业数据来源: common.loader.fetch_stock_basic() 的 industry 字段
    市值数据来源: ctx.daily_basic 的 circ_mv 列
    
    Args:
        df: 因子值 DataFrame（通常已标准化）
        factor_col: 因子值列名
        industry_col: 行业列名
        market_cap_col: 对数市值列名
        date_col: 日期列名
        
    Returns:
        DataFrame，新增 factor_value_neutral 列
    """
    ...
```

- **实现策略**:
  1. 从 `stock_basic` 获取每只股票的行业（`industry` 字段）
  2. 从 `daily_basic` 获取 `circ_mv`（流通市值），取对数
  3. join 到因子 DataFrame
  4. 逐日截面：Polars → numpy array → `statsmodels.OLS` → 残差 → Polars
  5. 残差再做一次 Z-score 标准化
  6. 不考虑行业分类不显著的边缘情况（个股数 << 行业数等，暂不处理）

- **关键决策**:
  - 行业哑变量用 Polars 的 one-hot encoding 转 numpy
  - `statsmodels.OLS` 逐日调用（约 250-500 个截面/年），性能可控
  - 二期不做 Barra 风格因子中性化，只做行业 + 市值

- **依赖**: T8（去极值）/ T9（缺失值）/ T10（标准化），因为中性化通常在标准化之后

- **验收标准**:
  - 每日截面的 `factor_value_neutral` 与行业、市值正交（相关系数 ≈ 0）
  - 某日某行业全缺时不抛异常（skip 该日）
  - 处理 250 天截面的时间 < 30s

---

#### T12: `lft/preprocessing/pipeline.py` — 预处理管线

- **文件**: `lft/preprocessing/pipeline.py`
- **描述**: 串联 T8-T11 四步，配置驱动
- **输入**: 因子原始值 DataFrame + 配置 dict
- **输出**: 处理后的 DataFrame

```python
from dataclasses import dataclass, field

@dataclass
class PreprocessingConfig:
    """预处理管线配置。"""
    outlier_method: str = "mad"       # "mad" | "sigma" | "none"
    outlier_sigma: float = 3.0       # MAD 的 σ 倍数
    fill_method: str = "median"      # "median" | "zero" | "none"
    normalize: bool = True            # 是否做截面 Z-score
    neutralize: bool = True           # 是否做行业+市值中性化
    neutralize_industry_col: str = "industry"
    neutralize_cap_col: str = "ln_cap"

def run_preprocessing(
    df: pl.DataFrame,
    config: PreprocessingConfig | None = None,
    stock_info: pl.DataFrame | None = None,      # 行业信息
    market_cap: pl.DataFrame | None = None,       # 市值数据
    date_col: str = "trade_date",
    factor_col: str = "factor_value",
) -> pl.DataFrame:
    """运行完整预处理管线。
    
    顺序: 去极值 → 缺失值填充 → 标准化 → 中性化
    每步生成新列而非覆盖原列，便于调试。
    
    Args:
        df: 原始因子值
        config: 管线配置，默认全开
        stock_info: 股票行业信息（含 ts_code, industry），中性化需要
        market_cap: 市值数据（含 ts_code, trade_date, circ_mv），中性化需要
        date_col: 日期列名
        factor_col: 因子值列名
        
    Returns:
        处理后的 DataFrame，原始列 + 处理步骤列
    """
    ...
```

- **列命名规范**:
  - 原始值: `factor_value`
  - 去极值后: `factor_value_clip`
  - 缺失值填充后: `factor_value_fill`
  - 标准化后: `factor_value_z`
  - 中性化后: `factor_value_neutral`
  - 最终输出列: `factor_value_neutral`（如未中性化则为 `factor_value_z`）

- **依赖**: T8-T11

- **验收标准**:
  - `run_preprocessing(df)` 返回包含所有中间列的 DataFrame
  - 配置 `neutralize=False` 时跳过中性化步骤
  - 配置 `outlier_method="none"` 时跳过去极值
  - 空 DataFrame 或单只股票不抛异常

---

### B4. 评估模块（4 个任务，T13-T16 可并行）

---

#### T13: `lft/evaluation/ic_analysis.py` — IC 分析

- **文件**: `lft/evaluation/ic_analysis.py`
- **描述**: 每日截面 Rank IC（Spearman 相关系数）分析
- **输入**: 因子值 DataFrame + 未来收益 DataFrame
- **输出**: IC 统计表

```python
@dataclass
class ICAnalysisResult:
    """IC 分析结果。"""
    ic_series: pl.DataFrame           # 每日 IC 值 [trade_date, ic]
    ic_mean: float                    # IC 均值
    ic_std: float                     # IC 标准差
    ir: float                         # Information Ratio = IC_mean / IC_std
    ic_positive_ratio: float          # IC > 0 占比
    icir: float                       # ICIR = IC_mean / IC_std
    ic_decay: dict[int, float]        # IC 衰减 {horizon: mean_ic}
    t_stat: float                     # IC 的 t 统计量
    
    def summary(self) -> str:
        """格式化的 IC 分析摘要。"""
        ...

def compute_rank_ic(
    factor_df: pl.DataFrame,
    forward_return_df: pl.DataFrame,
    factor_col: str = "factor_value_neutral",
    horizons: list[int] = [1, 5, 10, 20],
    date_col: str = "trade_date",
    ts_code_col: str = "ts_code",
) -> ICAnalysisResult:
    """计算 Rank IC。
    
    Args:
        factor_df: 因子值 DataFrame [trade_date, ts_code, factor_col]
        forward_return_df: 未来收益 DataFrame [trade_date, ts_code, ret_1d, ret_5d, ...]
        factor_col: 因子值列名
        horizons: IC 衰减回看期列表，如 [1, 5, 10, 20]
        date_col: 日期列名
        ts_code_col: 股票代码列名
        
    Returns:
        ICAnalysisResult
    """
    ...
```

- **实现要点**:
  - 未来收益 `forward_return_df` 的生成逻辑放在此模块中：对日线 `close` 计算 `ret_Nd = close(t+N) / close(t) - 1`
  - Polars 不支持内置 Spearman 相关系数，需要用 `.rank()` + Pearson correlation 近似，或逐日取 numpy array 用 `scipy.stats.spearmanr`
  - IC Decay：用不同 horizon 的 forward return 分别计算 IC
  - 建议实现一个内部 `_spearman_corr(x, y)` 辅助函数

- **性能考虑**: 逐日截面计算 Spearman（~250 天/年），用 scipy 取值 → 总体 <5s

- **验收标准**:
  - 动量因子在 A 股市场的 Rank IC 应为正（中期动量效应）
  - 反转因子短期 IC 应为负（A 股短期反转显著）
  - IC 序列标准差计算正确

---

#### T14: `lft/evaluation/backtest.py` — 分层回测

- **文件**: `lft/evaluation/backtest.py`
- **描述**: 每日按因子值排序分 10 组，等权配置，计算各组累计净值和多空收益
- **输入**: 因子值 + 日收益率
- **输出**: 分层回测统计

```python
@dataclass 
class BacktestResult:
    """分层回测结果。"""
    group_nav: pl.DataFrame          # 各组累计净值 [trade_date, group_1, ..., group_10]
    long_short_nav: pl.DataFrame     # Top-Bottom 多空净值 [trade_date, nav]
    group_stats: pl.DataFrame        # 各组统计 [group, annual_return, annual_vol, sharpe, max_dd, win_rate]
    long_short_stats: dict           # 多空统计
    
    def summary(self) -> str:
        """格式化的回测摘要。"""
        ...

def run_stratified_backtest(
    factor_df: pl.DataFrame,
    daily_return_df: pl.DataFrame,
    n_groups: int = 10,
    factor_col: str = "factor_value_neutral",
    date_col: str = "trade_date",
    ts_code_col: str = "ts_code",
    return_col: str = "pct_chg",        # 或 "ret_1d"
) -> BacktestResult:
    """运行分层回测。
    
    每日按因子值排序，等分 n_groups 组，等权计算各组次日收益。
    
    Args:
        factor_df: 因子值 DataFrame
        daily_return_df: 日收益率 DataFrame（pct_chg 或 ret_1d）
        n_groups: 分组数，默认 10
        factor_col: 因子值列名
        date_col: 日期列名
        ts_code_col: 股票代码列名
        return_col: 收益率列名（pct_chg 是 Tushare 标准涨幅列）
        
    Returns:
        BacktestResult
    """
    ...
```

- **实现要点**:
  - 因子值（T 日）→ 分组 → T+1 日收益率 → 等权平均 → 各组日收益序列
  - 累计净值 = (1 + daily_return).cumprod()
  - 多空收益 = Group10 - Group1（或 Group1 - Group10，视因子方向而定）
  - 年化收益 = mean(daily_return) * 252
  - Sharpe = mean / std * sqrt(252)
  - 最大回撤用 `(1 + ret).cumprod() / cummax - 1` 的最小值
  - 需要处理因子值为 NaN 的股票（不分配入任何组）

- **性能考虑**: Polars 惰性 + 分组操作，250 天全市场 ~3s

- **验收标准**:
  - 10 组累计净值曲线单调性合理（多头组 > 空头组 对于正向因子）
  - 多空 Sharpe > 0（对于有效因子）
  - 极端分组（G1 vs G10）收益差异显著

---

#### T15: `lft/evaluation/turnover.py` — 换手率分析

- **文件**: `lft/evaluation/turnover.py`
- **描述**: 分组换手率分析，含迁移矩阵
- **输入**: 因子值 DataFrame（每日截面）
- **输出**: 换手率统计

```python
@dataclass
class TurnoverResult:
    """换手率分析结果。"""
    avg_turnover: float                    # 平均单边换手率
    daily_turnover: pl.DataFrame           # 每日换手率 [trade_date, turnover]
    transition_matrix: pl.DataFrame        # 组迁移矩阵 [from_group, to_group, prob]
    
    def summary(self) -> str:
        """格式化的换手率摘要。"""
        ...

def compute_turnover(
    factor_df: pl.DataFrame,
    n_groups: int = 10,
    factor_col: str = "factor_value_neutral",
    date_col: str = "trade_date",
    ts_code_col: str = "ts_code",
) -> TurnoverResult:
    """计算因子分组换手率。
    
    每日截面按因子值分组。单边换手率 = (T 日组内股票 - T-1 日组内股票交集) / 组内股票数 的平均。
    
    Args:
        factor_df: 因子值 DataFrame
        n_groups: 分组数
        factor_col: 因子值列名
        date_col: 日期列名
        ts_code_col: 股票代码列名
        
    Returns:
        TurnoverResult
    """
    ...
```

- **实现要点**:
  - 单边换手率定义：`sum(|w_t - w_{t-1}|) / 2` → 简化：`1 - |Group_t ∩ Group_{t-1}| / |Group_t|`
  - 迁移矩阵：统计 T-1 日各组股票在 T 日属于哪组的概率
  - 结果用于判断因子调仓频率是否合理（日频因子应有较高换手率）

- **验收标准**:
  - `avg_turnover` 在 0-1 之间
  - 迁移矩阵对角线概率最高（大部分股票组别不变）
  - 对动量因子，换手率应低于反转因子

---

#### T16: `lft/evaluation/correlation.py` — 因子相关性

- **文件**: `lft/evaluation/correlation.py`
- **描述**: 多因子截面相关性分析（单因子评估时跳过，多因子对比时使用）
- **输入**: 多个因子的 DataFrame
- **输出**: 因子相关性矩阵

```python
def compute_factor_correlation(
    factor_dfs: dict[str, pl.DataFrame],  # {factor_name: df with factor_value_neutral}
    date_col: str = "trade_date",
    ts_code_col: str = "ts_code",
    factor_col: str = "factor_value_neutral",
) -> pl.DataFrame:
    """计算多因子截面平均相关性。
    
    每日截面计算因子间 Pearson/Spearman 相关 → 时序平均。
    
    Args:
        factor_dfs: 多因子数据
        date_col: 日期列名
        ts_code_col: 股票代码列名
        factor_col: 因子值列名
        
    Returns:
        相关性矩阵 DataFrame
    """
    ...
```

- **关键决策**: 二期仅实现接口，单因子评估时此模块不会被调用。多因子对比脚本（T18）中使用

- **验收标准**:
  - 两个因子 → 返回 2×2 对称矩阵
  - 对角线为 1.0
  - 与自身相关性为 1.0

---

### B5. 入口脚本（2 个任务，依赖所有上述模块）

---

#### T17: `scripts/run_lft_single.py` — 单因子一键评估入口

- **文件**: `scripts/run_lft_single.py`
- **描述**: CLI 入口，自动完成：加载数据 → 计算因子 → 预处理 → IC → 回测 → 换手率 → 打印摘要
- **用法**: `python scripts/run_lft_single.py --factor momentum_20d --start 20240101 --end 20251231`

```python
"""LFT 单因子评估脚本。

用法:
    python scripts/run_lft_single.py --factor momentum_20d --start 20240101 --end 20251231
    
选项:
    --factor      因子名称（必填）
    --start       起始日期 YYYYMMDD（默认 20240101）
    --end         截止日期 YYYYMMDD（默认 20251231）
    --universe    股票池（默认 lft_default）
    --skip-neut   跳过中性化
    --output      输出目录（默认 output/lft/）
    --verbose     详细日志
"""

import argparse
import sys
from pathlib import Path

def main():
    args = parse_args()
    
    # 1. 初始化日志
    setup_logging()
    logger = get_logger(__name__)
    
    # 2. 注册因子
    from lft.factors.registry import discover_factors, get_factor
    discover_factors()
    FactorClass = get_factor(args.factor)
    factor = FactorClass()
    
    # 3. 构建数据上下文
    from lft.data.context import FactorDataContext
    ctx = FactorDataContext(
        start=args.start,
        end=args.end,
        required_data=factor.required_data,
        lookback_days=factor.lookback_days,
    )
    
    # 4. 获取股票池
    from common.universe import get_universe
    # 需要在因子计算后按日期过滤股票池... 
    # 实际流程：先算全市场因子，再按股票池过滤
    
    # 5. 计算原始因子值
    raw_factor = factor.compute(ctx)
    factor.validate(raw_factor)
    
    # 6. 预处理
    from lft.preprocessing.pipeline import run_preprocessing, PreprocessingConfig
    config = PreprocessingConfig(neutralize=not args.skip_neut)
    
    # 加载中性化所需数据
    stock_info = None
    market_cap = None
    if not args.skip_neut:
        stock_info = fetch_stock_basic()
        market_cap = ctx.daily_basic.select(["trade_date", "ts_code", "circ_mv"]).collect()
    
    processed = run_preprocessing(raw_factor, config, stock_info, market_cap)
    
    # 7. IC 分析
    from lft.evaluation.ic_analysis import compute_rank_ic
    # 先生成 forward return
    from lft.evaluation.ic_analysis import _build_forward_returns
    fwd_ret = _build_forward_returns(ctx.daily, horizons=[1, 5, 10, 20])
    ic_result = compute_rank_ic(processed, fwd_ret)
    
    # 8. 分层回测
    from lft.evaluation.backtest import run_stratified_backtest
    daily_ret = ctx.daily.select(["trade_date", "ts_code", "pct_chg"]).collect()
    bt_result = run_stratified_backtest(processed, daily_ret)
    
    # 9. 换手率
    from lft.evaluation.turnover import compute_turnover
    to_result = compute_turnover(processed)
    
    # 10. 输出结果
    # 10a. 打印摘要
    print(ic_result.summary())
    print(bt_result.summary())
    print(to_result.summary())
    
    # 10b. 落盘结果
    save_results(args, factor, ic_result, bt_result, to_result)
    
    logger.info("评估完成")

if __name__ == "__main__":
    main()
```

- **输出目录结构**:
  ```
  output/lft/
  ├── factors/
  │   └── momentum_20d/
  │       └── 20240101_20251231.parquet        # 原始因子值
  ├── results/
  │   └── momentum_20d/
  │       ├── ic_20240101_20251231.parquet     # IC 序列
  │       ├── backtest_20240101_20251231.parquet # 回测结果
  │       └── turnover_20240101_20251231.parquet # 换手率
  └── charts/
      └── momentum_20d/                         # 后续可视化的图表
  ```

- **依赖**: T1-T16 全部

- **验收标准**:
  - `python scripts/run_lft_single.py --factor momentum_20d --start 20240101 --end 20251231` 运行成功
  - 控制台输出：IC Mean / IC Std / IR / 分层回测各组年化收益 / 多空 Sharpe / 换手率
  - 结果文件生成到 `output/lft/results/momentum_20d/` 下
  - 全程无未捕获异常

---

#### T18: `scripts/run_lft_compare.py` — 多因子对比脚本（可选）

- **文件**: `scripts/run_lft_compare.py`
- **描述**: 多因子 IC 对比 + 相关性矩阵（不跑完整回测，节省时间）
- **用法**: `python scripts/run_lft_compare.py --factors momentum_20d,reversal_5d,volatility_20d --start 20240101 --end 20251231`

- **流程**:
  1. 加载多个因子
  2. 逐个计算 + 预处理（并行程度有限，因为 Polars 本身已多线程）
  3. IC 对比表（横向）
  4. 因子相关性矩阵
  5. 打印对比摘要

- **输出**: 控制台对比表 + `output/lft/results/compare_*.parquet`

- **依赖**: T1-T16 + T17（复用 T17 的核心逻辑）

- **验收标准**:
  - 支持 2-5 个因子的对比
  - IC 对比表清晰易读
  - 相关性矩阵正确

---

## C. 技术规范

| 规范项 | 要求 |
|--------|------|
| **数据格式** | 全链路 `pl.DataFrame` / `pl.LazyFrame`，不裸用 pandas |
| **惰性求值** | 因子计算中利用 LazyFrame 链式操作，在 `.collect()` 时一次性触发 |
| **列命名规范** | `trade_date`（日期）, `ts_code`（股票代码）, `factor_value`（原始因子值）, `factor_value_{step}`（处理步骤后缀） |
| **因子输出格式** | 三列：`trade_date`, `ts_code`, `factor_value` |
| **日期格式** | 内部统一 `pl.Date` 类型，外部接口接受 `str "YYYYMMDD"` |
| **输出落盘** | `output/lft/factors/{factor_name}/` 放因子值, `output/lft/results/{factor_name}/` 放评估结果 |
| **日志规范** | 所有模块使用 `common.logger.get_logger(__name__)` |
| **Type Hints** | 所有公共函数必须有完整的类型标注 |
| **错误处理** | 数据缺失优雅降级（如停牌日不抛异常），配置错误快速失败 |
| **并行策略** | 因子实现（T3-T6）可并行编写，预处理模块（T8-T11）可并行编写，评估模块（T13-T16）可并行编写 |

---

## D. 依赖图与执行顺序

```
                        T1 (LFTFactor 基类)
                       /  \
                      /    \
              T2 (DataContext)   T7 (Registry - 依赖 T1)
             /    |    \    \
            /     |     \    \
    T3 (动量)  T4 (反转)  T5 (波动率)  T6 (换手率)
            \     |     /    /
             \    |    /    /
              ↓   ↓   ↓   ↓
         T8 (去极值)  T9 (缺失值)  T10 (标准化)  T11 (中性化)
                     \    |    /
                      \   |   /
                     T12 (预处理管线)
                    /    |    \    \
                   /     |     \    \
          T13 (IC)  T14 (回测)  T15 (换手率)  T16 (相关性)
                   \    |    /    /
                    \   |   /    /
                   T17 (单因子评估入口)
                        |
                   T18 (多因子对比)
```

### 阶段划分

| 阶段 | 任务 | 可并行 | 预估时间 |
|------|------|--------|----------|
| **阶段 1: 框架** | T1, T2 | ✅ T1/T2 可并行 | 1 session |
| **阶段 2: 因子实现** | T3, T4, T5, T6, T7 | ✅ 全部可并行 | 1 session |
| **阶段 3: 预处理** | T8, T9, T10, T11, T12 | ✅ T8-T11 可并行, T12 最后 | 1 session |
| **阶段 4: 评估 + 脚本** | T13, T14, T15, T16, T17, T18 | ✅ T13-T16 可并行 | 1-2 sessions |

---

## E. 数据流全景

```
┌─────────────────────────────────────────────────────────────┐
│                    scripts/run_lft_single.py                 │
│  --factor momentum_20d --start 20240101 --end 20251231      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. FactorDataContext(start, end, required_data, lookback)   │
│    ├─ ctx.daily → LazyFrame (load_parquet via storage)      │
│    └─ ctx.daily_basic → LazyFrame (if required)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Momentum20D.compute(ctx)                                 │
│    ├─ LazyFrame 链式: sort → shift → compute → filter       │
│    ├─ .collect() 触发实际计算                                │
│    └─ 返回 DataFrame[trade_date, ts_code, factor_value]     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Preprocessing Pipeline                                    │
│    ├─ mad_clip() → factor_value_clip                         │
│    ├─ fill_median() → factor_value_fill                      │
│    ├─ zscore() → factor_value_z                              │
│    └─ neutralize_ols() → factor_value_neutral                │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┬────────────┐
              ▼            ▼            ▼            ▼
┌─────────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ 4a. IC Analysis │ │4b. Backtest│ │4c. Turnover│ │4d. Corr │
│ Rank IC + Decay │ │ 10-group │ │ Migration │ │ Matrix  │
└────────┬────────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │               │            │            │
         └───────────────┴────────────┴────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Console Summary + output/lft/ 落盘                        │
│    IC Mean: 0.035  |  IC Std: 0.12  |  IR: 0.29            │
│    Long-Short Sharpe: 1.52  |  MaxDD: -12.3%               │
│    Avg Turnover: 0.65 (daily one-sided)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## F. 验收标准

### F1. 模块级验收

| # | 验收项 | 预期 |
|---|--------|------|
| 1 | `from lft.factors.base import LFTFactor` | 无报错 |
| 2 | `from lft.data.context import FactorDataContext` | 无报错 |
| 3 | `FactorDataContext("20240101", "20241231", lookback_days=20).daily` | 返回 LazyFrame |
| 4 | `Momentum20D().compute(ctx)` | 返回 3 列 DataFrame |
| 5 | `discover_factors()` | 返回包含所有已实现因子的 dict |
| 6 | 每个因子 `compute()` 输出 `validate()` 通过 | coverage > 0.8 |
| 7 | `mad_clip(df)` 极端值被截尾 | 中位数不变 |
| 8 | `fill_cross_sectional_median(df)` 无 NaN | 全填 |
| 9 | `cross_sectional_zscore(df)` 每日期望≈0, std≈1 | 截面标准化 |
| 10 | `neutralize_ols(df, stock_info, market_cap)` | 残差与行业/市值正交 |
| 11 | `compute_rank_ic(df, fwd_ret)` 返回 ICAnalysisResult | IR 合理 |
| 12 | `run_stratified_backtest(df, daily_ret)` 返回 10 组净值 | 单调性合理 |
| 13 | `compute_turnover(df)` 返回 TurnoverResult | avg_turnover ∈ (0,1) |
| 14 | `compute_factor_correlation(factor_dict)` 返回对称矩阵 | 对角线 1.0 |

### F2. 端到端验收

- [ ] `python scripts/run_lft_single.py --factor momentum_20d --start 20240101 --end 20251231` 运行成功
- [ ] 控制台输出：IC 表 + 分层回测统计 + 换手率统计
- [ ] `output/lft/factors/momentum_20d/20240101_20251231.parquet` 存在且可读
- [ ] `output/lft/results/momentum_20d/ic_*.parquet` 存在且可读
- [ ] `output/lft/results/momentum_20d/backtest_*.parquet` 存在且可读
- [ ] `output/lft/results/momentum_20d/turnover_*.parquet` 存在且可读

### F3. 代码质量

- [ ] `ruff check lft/` 零错误
- [ ] 所有公共函数有 docstring（Google style）
- [ ] 所有公共函数有 type hints
- [ ] 无循环导入
- [ ] 无全局可变状态（Registry 除外，其为有意的模块级单例）

### F4. 性能验收

| 操作 | 预期时间 |
|------|----------|
| 单因子计算（全市场 250 天） | < 10s |
| 预处理（4 步全开） | < 30s |
| IC 分析（4 个 horizon） | < 5s |
| 分层回测（10 组） | < 5s |
| 端到端（--factor momentum_20d） | < 60s |

---

## G. 文件清单

二期需要创建的所有文件（不含 `__init__.py`，这些已存在）：

| 文件 | 任务 | 行数估算 |
|------|------|----------|
| `lft/factors/base.py` | T1 | ~60 | ✅ |
| `lft/data/context.py` | T2 | ~80 | ✅ |
| `lft/factors/daily/momentum.py` | T3 | ~40 |
| `lft/factors/daily/reversal.py` | T4 | ~40 |
| `lft/factors/daily/volatility.py` | T5 | ~40 |
| `lft/factors/daily/turnover.py` | T6 | ~40 |
| `lft/factors/registry.py` | T7 | ~60 |
| `lft/preprocessing/outlier.py` | T8 | ~50 |
| `lft/preprocessing/missing.py` | T9 | ~40 |
| `lft/preprocessing/normalizer.py` | T10 | ~40 |
| `lft/preprocessing/neutralizer.py` | T11 | ~80 |
| `lft/preprocessing/pipeline.py` | T12 | ~80 |
| `lft/evaluation/ic_analysis.py` | T13 | ~120 |
| `lft/evaluation/backtest.py` | T14 | ~120 |
| `lft/evaluation/turnover.py` | T15 | ~80 |
| `lft/evaluation/correlation.py` | T16 | ~60 |
| `scripts/run_lft_single.py` | T17 | ~150 |
| `scripts/run_lft_compare.py` | T18 | ~100 |

**总计约 18 个文件，~1280 行代码。**

---

## H. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| daily 数据缺失某些日期 | 中 | 因子计算空值增多 | lookback 区间内缺失日 skip，因子值 NaN 在预处理阶段统一填 |
| Polars 窗口函数性能问题 | 低 | 因子计算慢 | 预排序 + LazyFrame 优化；实测 Polars 1.x 窗口操作很快 |
| 中性化 daily_basic 数据不完整 | 中 | 某些股票无法中性化 | 无市值/行业数据时 skip 该日该股（标记为 NaN，预处理填充） |
| statsmodels OLS 逐日调用性能 | 低 | 中性化慢 | 250 天 × N 行业哑变量在秒级内完成；如有性能问题可缓存回归权重 |
| 行业分类变动 | 低 | 中性化不准 | 使用 stock_basic 的静态 industry 字段（短期稳定） |
| 因子注册发现机制故障 | 低 | 因子无法加载 | 支持手动 register_factor + 提供明确的错误信息 |

---

## I. 与后续 Phase 的接口预留

| 后续 Phase | 本期预留 |
|------------|----------|
| Phase 3（周频/月频因子） | `LFTFactor.category` 已区分 daily/weekly/monthly |
| Phase 4（因子组合） | 因子输出格式统一，评估模块可复用 |
| Phase 5（报告模板） | 每个评估模块返回 `@dataclass` 结果 + `.summary()` 方法 |
| MFT 中频因子 | LFT 架构（基类 → 上下文 → 因子 → 预处理 → 评估）可作为 MFT 参考 |
