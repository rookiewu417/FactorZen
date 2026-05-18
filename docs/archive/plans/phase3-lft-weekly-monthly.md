# Phase 3: LFT 周频/月频因子

> **状态**: 🔴 待实施
> **创建时间**: 2026-05-13
> **预估工期**: 3-4 个 session
> **前置条件**: Phase 1（common/ 基础设施）✅ 已完成 | Phase 2（LFT 日频因子）🔴 待/正实施

---

## A. 三期目标

扩展 LFT 因子体系：从**纯日频因子** → 增加**周频因子**（日线聚合下采样）和**月频因子**（财务数据 + PIT 对齐），使框架覆盖 LFT 全调仓频率（daily / weekly / monthly）。

### 核心交付物

| 模块 | 内容 | 状态 |
|------|------|------|
| `common/calendar.py`（扩展） | `get_weekly_snapshot_dates()` / `get_monthly_snapshot_dates()` | 🔴 |
| `lft/data/context.py`（增强） | `snapshot_mode` + `weekly`/`monthly` 属性 | 🔴 |
| `lft/data/pit.py` | Point-In-Time 财务数据对齐 | 🔴 |
| `lft/factors/weekly/` | 3 个周频因子（momentum_weekly / volatility_weekly / turnover_weekly） | 🔴 |
| `lft/factors/monthly/` | 3 个月频因子（pe_ttm / pb / roe_ttm） | 🔴 |
| `lft/factors/registry.py`（增强） | 多包自动发现（daily + weekly + monthly） | 🔴 |
| `lft/evaluation/`（适配） | IC/回测/换手率适配周频和月频调仓 | 🔴 |
| `scripts/run_lft_single.py`（扩展） | `--frequency weekly|monthly` 参数 | 🔴 |

### 不涉及范围

- ❌ MFT 中频因子模块
- ❌ 因子组合 / 合成（四期）
- ❌ HTML 报告模板（五期）
- ❌ 机器学习因子
- ❌ HFT 高频因子
- ❌ 复杂的 Barra 风格因子中性化

---

## B. 任务清单

### B1. 基础设施增强（2 个任务）

---

#### T1: `common/calendar.py` 扩展 — 周/月末交易日函数

- **文件**: `common/calendar.py`（追加，不修改现有函数）
- **描述**: 新增两个函数，从交易日列表中提取每周/每月的最后一个交易日，作为周频/月频快照日期
- **输入**: `start` / `end` 日期（`YYYYMMDD` 字符串）
- **输出**: `list[date]` — 周频或月频的快照日列表

**新增函数**:

```python
def get_weekly_snapshot_dates(start: str, end: str) -> list[date]:
    """返回 [start, end] 区间内每周最后一个交易日列表。
    
    规则: ISO 周编号分组，取每周最大值作为快照日期。
    确保周频因子在"周末"调仓（A 股实际是周五或本周最后交易日）。
    
    Args:
        start: 起始日期 "YYYYMMDD"
        end: 截止日期 "YYYYMMDD"
        
    Returns:
        list[date]: 每周快照日期，按时间升序排列
        
    Example:
        2026年5月: 5月4日(周一)~5月8日(周五) → 快照日 5月8日
                   5月11日(周一)~5月15日(周五) → 快照日 5月15日
    """
    ...

def get_monthly_snapshot_dates(start: str, end: str) -> list[date]:
    """返回 [start, end] 区间内每月最后一个交易日列表。
    
    规则: 按年-月分组，取每月最大值作为快照日期。
    确保月频因子在"月末"调仓。
    
    Args:
        start: 起始日期 "YYYYMMDD"
        end: 截止日期 "YYYYMMDD"
        
    Returns:
        list[date]: 每月快照日期，按时间升序排列
        
    Example:
        2026年1月: 最后一个交易日 ~1月30日
        2026年2月: 最后一个交易日 ~2月27日（春节影响）
    """
    ...
```

- **实现策略**:
  1. 调用 `get_trade_dates(start, end)` 获取区间内所有交易日
  2. 周频：用 `date.isocalendar()[1]`（ISO 周号）+ 年份分组，每组取 `max`
  3. 月频：用 `(date.year, date.month)` 分组，每组取 `max`
  4. 使用 Polars 的 `group_by` + `max` 聚合或纯 Python `itertools.groupby`

- **关键决策**:
  - 不引入新的外部依赖，纯基于已有 `get_trade_dates()` 实现
  - 周频使用 ISO 周号（跨年周边界已正确处理）
  - 返回类型与 `get_trade_dates()` 一致（`list[date]`）

- **依赖**: 无（仅依赖已有 `get_trade_dates()`）

- **验收标准**:
  - `get_weekly_snapshot_dates("20260101", "20260513")` 返回约 19 个日期
  - `get_monthly_snapshot_dates("20260101", "20260513")` 返回约 5 个日期（1~5 月）
  - 返回的每个日期都是交易日（`is_trade_date(d) == True`）
  - 每周/月恰有一个快照日期（无遗漏无重复）

---

#### T2: `lft/data/context.py` 增强 — 周频/月频快照模式

- **文件**: `lft/data/context.py`（扩展现有 `FactorDataContext`）
- **描述**: 为 `FactorDataContext` 增加 `snapshot_mode` 参数和周/月频数据访问属性，使因子可以在日频全量数据上计算，再下采样到周/月频截面

**新增字段**:

```python
@dataclass
class FactorDataContext:
    # ── 现有字段（不变）──
    start: str
    end: str
    required_data: list[str] = field(default_factory=lambda: ["daily"])
    lookback_days: int = 20
    universe: Optional[list[str]] = None
    
    # ── 新增字段 ──
    snapshot_mode: str = "daily"  # "daily" | "weekly" | "monthly"
    
    # ── 现有私有属性（不变）──
    _daily: Optional[pl.LazyFrame] = field(default=None, repr=False)
    _daily_basic: Optional[pl.LazyFrame] = field(default=None, repr=False)
    
    # ── 新增私有属性 ──
    _weekly_snapshot: Optional[pl.LazyFrame] = field(default=None, repr=False)
    _monthly_snapshot: Optional[pl.LazyFrame] = field(default=None, repr=False)
    _snapshot_dates: Optional[list[date]] = field(default=None, repr=False)
```

**新增属性**:

```python
@property
def snapshot_dates(self) -> list[date]:
    """根据 snapshot_mode 返回快照日期列表。
    
    - "daily": 返回 get_trade_dates(start, end) 的全部交易日
    - "weekly": 返回每周最后一个交易日
    - "monthly": 返回每月最后一个交易日
    """
    ...

@property
def weekly(self) -> pl.LazyFrame:
    """日线数据下采样到周频快照。
    
    先加载完整日线数据（含 lookback 扩展），再过滤到仅保留
    快照日所在行。因子计算仍需使用 ctx.daily 获取完整序列
    （用于 rolling/shift 计算），weekly 仅用于最终截面对齐。
    """
    ...

@property  
def monthly(self) -> pl.LazyFrame:
    """日线数据下采样到月频快照。逻辑同 weekly。"""
    ...

@property
def weekly_basic(self) -> pl.LazyFrame:
    """daily_basic 下采样到周频快照。"""
    ...

@property
def monthly_basic(self) -> pl.LazyFrame:
    """daily_basic 下采样到月频快照。"""
    ...
```

- **关键决策**:
  - `ctx.daily` 保持返回**完整日线**（含 lookback 扩展区间）—— 因子计算需要全量序列做滚动操作
  - `ctx.weekly` / `ctx.monthly` 是日线的**下采样视图**—— 仅用于月频因子直接取 `daily_basic` 值（pe_ttm/pb 不需要滚动计算）
  - `ctx.snapshot_dates` 供周频因子在 `compute()` 末尾做 `trade_date ∈ snapshot_dates` 过滤
  - 惰性求值保持一致：所有新增属性返回 `pl.LazyFrame`

- **周频因子计算流程**:
  ```
  ctx.daily (全量日线 + lookback)
    → 因子公式计算（shift/rolling_std/rolling_mean）
    → .collect() 触发计算
    → .filter(trade_date ∈ ctx.snapshot_dates)   # 只保留周频快照日
    → 返回周频因子值
  ```

- **月频因子计算流程（以 pe_ttm 为例）**:
  ```
  ctx.monthly_basic (daily_basic 下采样到月末)
    → .select(["trade_date", "ts_code", pe_ttm])
    → .collect()
    → 返回月频因子值
  ```

- **依赖**: T1（calendar 新函数）

- **验收标准**:
  - `FactorDataContext(start, end, snapshot_mode="weekly").snapshot_dates` 返回周频日期列表
  - `FactorDataContext(start, end, snapshot_mode="monthly").snapshot_dates` 返回月频日期列表
  - `ctx.weekly` 返回 LazyFrame，`collect()` 后行数 = 股票数 × 周数
  - `ctx.monthly_basic` 返回 LazyFrame，包含 `pe_ttm`, `pb` 等列
  - `ctx.daily` 行为不变（向后兼容，默认 `snapshot_mode="daily"`）

---

### B2. 周频因子实现（3 个任务，可并行）

> **核心原则**: 周频因子**不重复实现**因子公式——复用日频因子的计算逻辑，仅在最终输出前按周频快照日期过滤（"日频公式 + 截面下采样"）。

---

#### T3: `lft/factors/weekly/momentum.py` — 周频动量因子

- **文件**: `lft/factors/weekly/momentum.py`
- **描述**: 20 日动量因子的周频版本。与 `Momentum20D` 共享完全相同公式，仅输出频率不同
- **公式**: `momentum_weekly = close(t) / close(t-20) - 1`（同 `momentum_20d`）
- **输入**: `ctx.daily`（全量日线）
- **输出**: `pl.DataFrame` 三列 `[trade_date, ts_code, factor_value]`，`trade_date` 为周频日期

```python
"""周频动量因子。复用日频公式，下采样到周频快照。"""

import polars as pl
from lft.factors.base import LFTFactor
from lft.data.context import FactorDataContext


class MomentumWeekly(LFTFactor):
    name = "momentum_weekly"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 30  # 比日频多留一些 buffer
    description = "周频 20 日动量（日频公式 + 周频采样）"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 1. 在完整日线上计算动量（同 Momentum20D）
        daily = ctx.daily
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(20).over("ts_code") - 1.0)
                .alias("factor_value")
            )
            .filter(
                pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d")
            )
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        # 2. 下采样到周频快照日期
        snapshot_dates = ctx.snapshot_dates
        result = result.filter(pl.col("trade_date").is_in(snapshot_dates))
        return result
```

- **关键决策**:
  - `lookback_days=30`（比日频的 25 多 5 天）—— 确保周频快照日有足够的前序数据
  - 公式 100% 复用 `Momentum20D` 的逻辑（复制代码而非 import，避免产生硬依赖）
  - `trade_date` 输出为周频日期（如 `2026-05-08`, `2026-05-15` 等）

- **依赖**: T2（context 增强）

- **验收标准**:
  - `factor.compute(ctx)` 返回 3 列 DataFrame
  - `trade_date` 去重后约等于区间周数（~52 周/年）
  - `factor_value` 范围与日频动量大体一致
  - `factor.validate(result)["warnings"]` 无低覆盖率警告

---

#### T4: `lft/factors/weekly/volatility.py` — 周频波动率因子

- **文件**: `lft/factors/weekly/volatility.py`
- **描述**: 20 日波动率的周频版本
- **公式**: 20 日对数收益率标准差（同 `volatility_20d`）
- **输入**: `ctx.daily`
- **输出**: 三列周频 DataFrame

```python
class VolatilityWeekly(LFTFactor):
    name = "volatility_weekly"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 30
    description = "周频 20 日波动率（日频公式 + 周频采样）"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code")).log()
                .alias("log_ret")
            )
            .with_columns(
                pl.col("log_ret").rolling_std(20, min_periods=10).over("ts_code")
                .alias("factor_value")
            )
            .filter(
                pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d")
            )
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        result = result.filter(pl.col("trade_date").is_in(ctx.snapshot_dates))
        return result
```

- **依赖**: T2

- **验收标准**: 同上（T3）

---

#### T5: `lft/factors/weekly/turnover.py` — 周频换手率因子

- **文件**: `lft/factors/weekly/turnover.py`
- **描述**: 5 日平均成交量（换手率 proxy）的周频版本
- **公式**: `log1p(rolling_mean(vol, 5))`，同 `turnover_5d`
- **输入**: `ctx.daily`
- **输出**: 三列周频 DataFrame

```python
class TurnoverWeekly(LFTFactor):
    name = "turnover_weekly"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 15  # 5 日均线 + 缓冲
    description = "周频 5 日平均成交量"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                pl.col("vol").rolling_mean(5, min_periods=3).over("ts_code")
                .log1p()
                .alias("factor_value")
            )
            .filter(
                pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d")
            )
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        result = result.filter(pl.col("trade_date").is_in(ctx.snapshot_dates))
        return result
```

- **依赖**: T2

- **验收标准**: 同上（T3）

---

### B3. PIT 对齐 + 月频因子（3 个任务）

---

#### T6: `lft/data/pit.py` — Point-In-Time 财务数据对齐

- **文件**: `lft/data/pit.py`
- **描述**: 为月频因子提供 PIT 对齐能力——确保在每月末调仓时，**仅使用已公告的财务数据**，杜绝未来信息泄露
- **输入**: 
  - `fina_df`: 财务指标 DataFrame（来自 `fetch_finance("fina_indicator", ...)`），必须包含 `ts_code`, `end_date`（报告期）, `ann_date`（公告日）及指标列
  - `snapshot_dates`: 月频快照日期列表 `list[date]`
- **输出**: PIT 对齐后的 DataFrame，列: `snapshot_date`, `ts_code`, `end_date`（对齐到的报告期）, `ann_date`, 指标列

```python
"""Point-In-Time 财务数据对齐模块。

确保每月末调仓时只使用已公告的财务数据，
杜绝未来信息（look-ahead bias）。

使用场景:
    fina_indicator 数据中，2025Q4 财报的 end_date=20251231，
    但 ann_date=20260430（公告日）。如果在 2026-03-31 调仓，
    不应使用该 Q4 数据（尚未公告）。
"""

import polars as pl
from datetime import date


def pit_align(
    fina_df: pl.DataFrame,
    snapshot_dates: list[date],
) -> pl.DataFrame:
    """对财务数据做 Point-In-Time 对齐。
    
    对每个月频快照日期 snapshot_date:
      找出每只股票的「最新已公告」财务报告——
      即 ann_date <= snapshot_date 中 end_date 最大的那条。
    
    Args:
        fina_df: 财务数据，必须包含列:
            ts_code, end_date (报告期 Date), ann_date (公告日 Date),
            以及财务指标列（如 roe, roa 等）。
        snapshot_dates: 月频快照日期列表（升序）。
        
    Returns:
        pl.DataFrame，列:
            snapshot_date (Date)  - 调仓日
            ts_code               - 股票代码
            end_date              - 对齐到的报告期
            ann_date              - 公告日
            <indicator columns>   - 财务指标列
        
    Example:
        快照日 2026-03-31:
          股票 A: 2025Q3 财报 ann_date=20251030 ✓ 可用
                  2025Q4 财报 ann_date=20260430 ✗ 尚未公告
          → 对齐到 2025Q3 (end_date=20250930)
        
        快照日 2026-05-31:
          股票 A: 2025Q4 财报 ann_date=20260430 ✓ 已公告
          → 对齐到 2025Q4 (end_date=20251231)
    """
    ...
```

- **实现策略**:
  1. 对每个 `snapshot_date`，在 `fina_df` 中筛选 `ann_date <= snapshot_date` 的记录
  2. 按 `ts_code` 分组，取 `end_date` 最大的那条
  3. 合并所有 `snapshot_date` 的结果
  4. 优化：可以先按 `(ts_code, end_date)` 排序，然后做 asof join

- **Polars 实现思路**:
  ```python
  # 对每个 snapshot_date 做 asof join
  # 将 snapshot_dates 构造为 DataFrame
  snapshots = pl.DataFrame({"snapshot_date": snapshot_dates})
  
  # 对于每只股票，找到 ann_date <= snapshot_date 的最新财务数据
  # 方法：笛卡尔积 + 过滤 + group_by 取 max
  aligned = (
      fina_df
      .join(snapshots, how="cross")
      .filter(pl.col("ann_date") <= pl.col("snapshot_date"))
      .sort(["ts_code", "snapshot_date", "end_date"], descending=[False, False, True])
      .group_by(["ts_code", "snapshot_date"])
      .first()
  )
  ```

- **关键决策**:
  - 简单但正确的实现优先于性能优化（月频数据量小：5000 股票 × 12 月 = 60000 行）
  - `ann_date` 可能为 `None`（未公告），这些记录在 PIT 对齐时**排除**
  - 财务数据需预先通过 `fetch_finance("fina_indicator", ...)` 拉取并缓存
  - 不对 `end_date` 的发布延迟做复杂建模（如业绩快报/预告提前反映）

- **依赖**: T1（snapshot dates 可用）

- **验收标准**:
  - 2026-03-31 快照日：不包含 `ann_date > 2026-03-31` 的财报
  - 2026-05-31 快照日：包含 `ann_date <= 2026-05-31` 的最新财报
  - 同一快照日每只股票最多一条记录
  - `snapshot_date` 列名与月频因子的 `trade_date` 可对齐

---

#### T7: `lft/factors/monthly/value.py` — 估值因子（pe_ttm / pb）

- **文件**: `lft/factors/monthly/value.py`
- **描述**: 月频估值因子，直接从 `daily_basic` 下采样到月末快照
- **因子**:
  - `pe_ttm`: 滚动市盈率（来自 `daily_basic.pe_ttm`）
  - `pb`: 市净率（来自 `daily_basic.pb`）
- **输入**: `ctx.monthly_basic`（daily_basic 下采样到月末）
- **输出**: `pl.DataFrame` 三列 `[trade_date, ts_code, factor_value]`

```python
"""月频估值因子：pe_ttm 和 pb。直接从 daily_basic 月末快照提取。"""

import polars as pl
from lft.factors.base import LFTFactor
from lft.data.context import FactorDataContext


class PeTtmMonthly(LFTFactor):
    name = "pe_ttm"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5  # daily_basic 不需要长回看，月末值即可
    description = "月频滚动市盈率（PE-TTM），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        monthly_basic = ctx.monthly_basic
        result = (
            monthly_basic
            .select([
                pl.col("trade_date"),
                pl.col("ts_code"),
                pl.col("pe_ttm").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
            .collect()
        )
        return result


class PbMonthly(LFTFactor):
    name = "pb"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "月频市净率（PB），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        monthly_basic = ctx.monthly_basic
        result = (
            monthly_basic
            .select([
                pl.col("trade_date"),
                pl.col("ts_code"),
                pl.col("pb").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
            .collect()
        )
        return result


# 模块级实例化（两个因子都注册）
PeTtmMonthly()
PbMonthly()
```

- **关键决策**:
  - `pe_ttm` / `pb` 已在 `daily_basic` 中由 Tushare 计算好，不需要自行推导
  - 月末估值就是月末那天的截面值，不涉及跨期计算，直接用 `ctx.monthly_basic` 即可
  - `pe_ttm` 为负值时保留原值（负 PE 本身是有效信号），预处理阶段再做处理
  - 两个因子放在同一个文件中，均通过模块级实例化注册

- **依赖**: T2（`ctx.monthly_basic`）, T6 可选（pe_ttm/pb 不需要 PIT）

- **验收标准**:
  - `factor.compute(ctx)` 返回月频三列 DataFrame
  - `trade_date` 为月末交易日
  - `factor_value` 包含合理的正/负 PE 值和正 PB 值
  - `factor.validate(result)["coverage"] > 0.85`

---

#### T8: `lft/factors/monthly/profitability.py` — 盈利因子（roe_ttm）

- **文件**: `lft/factors/monthly/profitability.py`
- **描述**: 月频 ROE TTM 因子，需要 PIT 对齐确保无未来信息
- **公式**: `roe_ttm`（来自 `fina_indicator`，PIT 对齐）
- **输入**: 
  - `ctx.daily_basic`（用于获取月末截面股票列表）
  - `fina_indicator` Parquet 数据（通过 `common.storage.load_parquet` 加载）
  - `ctx.snapshot_dates`（月频快照日）
- **输出**: `pl.DataFrame` 三列 `[trade_date, ts_code, factor_value]`

```python
"""月频 ROE TTM 因子。使用 PIT 对齐确保无未来信息。"""

import polars as pl
from lft.factors.base import LFTFactor
from lft.data.context import FactorDataContext
from lft.data.pit import pit_align
from common.storage import scan_parquet


class RoeTtmMonthly(LFTFactor):
    name = "roe_ttm"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "月频 ROE TTM（PIT 对齐），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 1. 加载财务数据（fina_indicator）
        fina_lf = scan_parquet("finance")
        fina_df = (
            fina_lf
            .filter(pl.col("end_date").is_not_null())
            .select(["ts_code", "end_date", "ann_date", "roe"])
            .collect()
        )
        
        # 2. PIT 对齐到月频快照日
        snapshot_dates = ctx.snapshot_dates
        pit_df = pit_align(fina_df, snapshot_dates)
        
        # 3. 提取 roe 作为因子值
        result = (
            pit_df
            .select([
                pl.col("snapshot_date").alias("trade_date"),
                pl.col("ts_code"),
                pl.col("roe").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
        )
        return result


# 模块级实例化
RoeTtmMonthly()
```

- **关键决策**:
  - 财务数据从 `data/raw/finance/` 加载（Phase 2 的 `fetch_finance` 已支持 `fina_indicator`）
  - PIT 对齐确保本因子**绝不使用未来财报**（否则模拟月频调仓时会高估收益）
  - `roe` 值可能为负（亏损公司），保留原值
  - 如果 `fina_indicator` 数据未就绪，优雅降级（返回空 DataFrame + warning）

- **依赖**: T6（pit.py）, T2（snapshot_dates）

- **验收标准**:
  - `factor.compute(ctx)` 返回月频 ROE 值
  - 对 2025Q4 财报：2026-04-30 前的快照日不含该 Q4 ROE，2026-05 后的快照日包含
  - PIT 对齐验证：手动检查 2-3 个快照日的 ROE 值与实际公告时间一致
  - `factor.validate(result)["coverage"] > 0.7`（ROE 覆盖率略低，部分公司财报延迟）

---

### B4. 注册中心 + 评估适配 + 脚本扩展（4 个任务，可部分并行）

---

#### T9: `lft/factors/registry.py` 增强 — 多包自动发现

- **文件**: `lft/factors/registry.py`（修改现有）
- **描述**: 扩展 `discover_factors()` 支持从**多个包**（daily + weekly + monthly）发现因子

**修改点**:

```python
# 现有（仅搜索 daily）:
def discover_factors(package: str = "lft.factors.daily") -> dict[str, Type[LFTFactor]]:
    ...

# 改为（接受包列表）:
def discover_factors(
    packages: list[str] | None = None,
) -> dict[str, Type[LFTFactor]]:
    """从多个包中递归发现所有 LFTFactor 子类。
    
    Args:
        packages: 包路径列表，默认 ["lft.factors.daily", "lft.factors.weekly", "lft.factors.monthly"]
        
    Returns:
        dict[str, Type[LFTFactor]]: {factor_name: FactorClass}
    """
    if packages is None:
        packages = [
            "lft.factors.daily",
            "lft.factors.weekly",
            "lft.factors.monthly",
        ]
    
    for pkg in packages:
        try:
            pkg_mod = importlib.import_module(pkg)
            for _, mod_name, _ in pkgutil.iter_modules(pkg_mod.__path__, prefix=pkg + "."):
                try:
                    mod = importlib.import_module(mod_name)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, LFTFactor)
                            and attr is not LFTFactor
                        ):
                            instance = attr()
                            _FACTOR_REGISTRY[instance.name] = attr
                            # 不再 break，允许一个模块注册多个因子（如 value.py 中的 pe_ttm + pb）
                except Exception:
                    pass
        except ModuleNotFoundError:
            pass
    
    _discovered = True
    return _FACTOR_REGISTRY
```

- **关键变更**:
  1. `packages` 参数从单个 `str` → `list[str]`
  2. 移除 `break` 语句——允许一个模块注册**多个**因子（如 `value.py` 中的 `PeTtmMonthly` + `PbMonthly`）
  3. 向后兼容：`get_factor("momentum_20d")` 仍正常工作
  4. 自动发现范围覆盖 `daily/`, `weekly/`, `monthly/` 三个目录

- **依赖**: T3-T8（因子实现）

- **验收标准**:
  - `discover_factors()` 返回包含 ~10 个因子（4 daily + 3 weekly + 3 monthly）
  - `get_factor("momentum_weekly")` 返回 `MomentumWeekly` 类
  - `get_factor("pe_ttm")` 返回 `PeTtmMonthly` 类
  - `get_factor("roe_ttm")` 返回 `RoeTtmMonthly` 类
  - `list_factors("weekly")` 返回 3 个周频因子
  - `list_factors("monthly")` 返回 3 个月频因子

---

#### T10: `lft/evaluation/` 适配 — 周频/月频评估

- **文件**: 修改 `ic_analysis.py`, `backtest.py`, `turnover.py`
- **描述**: 为三个评估模块增加多频率支持，使 IC/回测/换手率能正确处理周频和月频数据

---

##### T10a: IC 分析适配

- **文件**: `lft/evaluation/ic_analysis.py`
- **变更**:
  1. `compute_rank_ic()` 新增参数 `frequency: str = "daily"` 
  2. 根据频率自动设置不同的 `horizons` 默认值和前向收益计算逻辑
  3. 新增 `compute_fwd_returns_multi_freq()` 辅助函数

```python
# 频率 → horizons 映射
_FREQ_HORIZONS = {
    "daily": [1, 5, 10, 20],      # 1/5/10/20 天
    "weekly": [1, 2, 4],           # 1/2/4 周
    "monthly": [1, 3, 6],          # 1/3/6 月
}

def compute_rank_ic(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,       # 仍是日收益
    factor_col: str = "factor_clean",
    frequency: str = "daily",
    horizons: list[int] | None = None,
) -> ICAnalysisResult:
    """计算 Rank IC，支持多频率。
    
    Args:
        frequency: "daily" | "weekly" | "monthly"
            - daily: 因子值 vs T+1 日收益
            - weekly: 因子值 vs 下周收益（cumulative 5 日）
            - monthly: 因子值 vs 下月收益（cumulative ~21 日）
    """
    if horizons is None:
        horizons = _FREQ_HORIZONS.get(frequency, [1, 5, 10, 20])
    
    # 对于周频/月频，前向收益需要将日收益累计
    if frequency == "weekly":
        forward_ret = _compute_weekly_fwd_returns(daily_ret, horizons)
    elif frequency == "monthly":
        forward_ret = _compute_monthly_fwd_returns(daily_ret, horizons)
    else:
        forward_ret = compute_fwd_returns(daily_ret, horizons)
    ...
```

- **周频前向收益**:
  - `fwd_ret_1w` = 后 5 个交易日的累计收益
  - `fwd_ret_2w` = 后 10 个交易日的累计收益
  - 用 `pl.col("ret").shift(-i).over("ts_code")` 累加

- **月频前向收益**:
  - `fwd_ret_1m` = 后 ~21 个交易日的累计收益
  - 通过合并月末快照日期来实现

---

##### T10b: 回测适配

- **文件**: `lft/evaluation/backtest.py`
- **变更**:
  1. `run_stratified_backtest()` 新增参数 `frequency: str = "daily"`
  2. 周频回测：仅在周频快照日做截面分组，持有至下个快照日
  3. 月频回测：仅在月末快照日做截面分组，持有至下月末

```python
def run_stratified_backtest(
    factor_df: pl.DataFrame,
    daily_ret: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    frequency: str = "daily",
) -> BacktestResult:
    """分层回测，支持多频率调仓。
    
    - daily: 每日调仓，T日因子 → T+1日收益
    - weekly: 周频调仓，快照日因子 → 持有到下周快照日（累计收益）
    - monthly: 月频调仓，月末因子 → 持有到下月末（累计收益）
    """
    if frequency == "daily":
        # 现有逻辑：每日分组 + T+1 日收益
        ...
    elif frequency == "weekly":
        # 仅在 factor_df 的 trade_date（周频快照日）分组
        # forward_return = 从当前快照日到下一个快照日的累计收益
        ...
    elif frequency == "monthly":
        # 仅在月末分组
        # forward_return = 从当前月末到下月末的累计收益
        ...
```

- **关键实现**:
  - 周频/月频的 `forward_return` 通过合并 `daily_ret` 计算区间累计收益
  - 年化收益计算需根据频率调整：周频 × 52，月频 × 12
  - `summary()` 中标注频率信息

---

##### T10c: 换手率适配

- **文件**: `lft/evaluation/turnover.py`
- **变更**:
  1. `compute_turnover()` 新增 `frequency` 参数
  2. 周频：相邻快照日之间计算分组变更率（~52 个观测/年）
  3. 月频：相邻月末之间计算分组变更率（~12 个观测/年）
  4. `summary()` 输出中标注样本量极少的风险

```python
def compute_turnover(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    frequency: str = "daily",
) -> TurnoverResult:
    """计算分组换手率。
    
    Args:
        frequency: "daily" | "weekly" | "monthly"
        
    Note:
        周频约 52 个观测/年，月频仅 12 个观测/年。
        月频换手率统计稳定性较差，仅作参考。
    """
    ...
```

- **依赖**: T3-T8（因子有数据可供测试）

- **验收标准**:
  - 周频 IC：`ic_result.summary()` 输出 `n_periods` ≈ 周数（~52/年）
  - 月频 IC：`ic_result.summary()` 输出 `n_periods` ≈ 月数（~12/年）
  - 周频回测：`bt_result.summary()` 各组年化收益基于 52 周计算
  - 月频换手率：`summary()` 包含样本量提示（"月频换手率仅 12 个观测，统计不稳健"）

---

#### T11: `scripts/run_lft_single.py` 扩展 — 多频率支持

- **文件**: `scripts/run_lft_single.py`（修改现有）
- **描述**: 增加 `--frequency` 参数，自动选择因子目录、配置 Context、适配评估频率
- **用法**: 
  ```
  python scripts/run_lft_single.py --factor momentum_weekly --frequency weekly --start 20240101 --end 20251231
  python scripts/run_lft_single.py --factor pe_ttm --frequency monthly --start 20240101 --end 20251231
  ```

**新增参数**:

```python
parser.add_argument(
    "--frequency", 
    default="daily",
    choices=["daily", "weekly", "monthly"],
    help="因子频率，影响注册发现、数据上下文和评估频率"
)
```

**关键变更点**:

```python
def main():
    args = parser.parse_args()
    
    # 1. 根据频率注册对应包的因子
    from lft.factors.registry import discover_factors, get_factor
    if args.frequency == "weekly":
        discover_factors(["lft.factors.daily", "lft.factors.weekly"])
    elif args.frequency == "monthly":
        discover_factors(["lft.factors.daily", "lft.factors.weekly", "lft.factors.monthly"])
    else:
        discover_factors(["lft.factors.daily"])
    
    # 2. Context 构建时传入 snapshot_mode
    ctx = FactorDataContext(
        start=args.start,
        end=args.end,
        required_data=factor.required_data,
        lookback_days=factor.lookback_days,
        universe=ts_codes,
        snapshot_mode=args.frequency,  # NEW
    )
    
    # 3. 评估时传入频率
    ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
    bt_result = run_stratified_backtest(
        clean_df, ret_df.select(["trade_date", "ts_code", "ret"]), 
        frequency=args.frequency,
    )
    to_result = compute_turnover(clean_df, frequency=args.frequency)
    
    # 4. 输出目录按频率分
    # output/lft/factors/weekly/momentum_weekly/...
    freq_dir = OUTPUT_LFT_FACTORS / args.frequency
```

- **关键决策**:
  - `--frequency` 默认为 `"daily"`，保持 Phase 2 脚本的向后兼容
  - 输出目录按频率分层：`output/lft/factors/{frequency}/{factor_name}/`
  - 日志中标注频率信息

- **依赖**: T1-T10 全部

- **验收标准**:
  - `python scripts/run_lft_single.py --factor momentum_weekly --frequency weekly --start 20240101 --end 20251231` 运行成功
  - `python scripts/run_lft_single.py --factor pe_ttm --frequency monthly --start 20240101 --end 20251231` 运行成功
  - `python scripts/run_lft_single.py --factor momentum_20d --start 20240101 --end 20251231`（无 --frequency）行为与 Phase 2 一致
  - 控制台输出 IC Mean / IC Std / IR + 分层回测各组年化收益 + 换手率

---

## C. 技术规范

| 规范项 | 要求 |
|--------|------|
| **数据格式** | 全链路 `pl.DataFrame` / `pl.LazyFrame`，不裸用 pandas |
| **惰性求值** | Context 新增属性保持 LazyFrame，`.collect()` 在因子 `compute()` 中触发 |
| **周频因子公式** | **不重复实现**公式——复制日频因子的 Polars 链，末尾下采样到周频快照 |
| **PIT 对齐** | 简单 `ann_date <= snapshot_date` 过滤 + group_by 取 max，不对公告延迟建模 |
| **输出列** | 统一三列：`trade_date`, `ts_code`, `factor_value`；`trade_date` 为对应频率的快照日期 |
| **因子元信息** | `category` = "weekly" / "monthly"；`frequency` = "weekly" / "monthly" |
| **Type Hints** | 所有公共函数必须有完整的类型标注 |
| **错误处理** | 财务数据未就绪时优雅降级（warning + 空结果），不抛未捕获异常 |

---

## D. 依赖图与执行顺序

```
                               Phase 2 (已完成/就绪)
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
              T1 (calendar)       现有 base.py      现有 evaluation
              周/月末交易日      LFTFactor ABC      ic/backtest/turnover
                    │                 │                 │
                    ▼                 │                 │
              T2 (context 增强)       │                 │
         snapshot_mode + weekly/      │                 │
         monthly 属性                 │                 │
          │            │              │                 │
   ┌──────┼──────┐     │              │                 │
   │      │      │     │              │                 │
   ▼      ▼      ▼     │              │                 │
  T3     T4     T5     │              │                 │
动量周  波动周  换手周  │              │                 │
   │      │      │     │              │                 │
   └──────┼──────┘     │              │                 │
          │            │              │                 │
          ▼            │              │                 │
          │     T6 (PIT 对齐)         │                 │
          │            │              │                 │
          │       ┌────┴────┐         │                 │
          │       ▼         ▼         │                 │
          │      T7        T8         │                 │
          │   pe_ttm+pb  roe_ttm      │                 │
          │       │         │         │                 │
          └───────┴────┬────┴─────────┘                 │
                       │                                │
                       ▼                                │
                 T9 (registry)                          │
            多包自动发现（daily+weekly+monthly）          │
                       │                                │
                       ▼                                │
                 T10 (evaluation 适配)                   │
            IC/回测/换手率 → 周频+月频调仓              │
                       │                                │
                       ▼                                │
                 T11 (脚本扩展)                          │
            --frequency weekly|monthly                   |
```

### 阶段划分

| 阶段 | 任务 | 可并行 | 预估时间 |
|------|------|--------|----------|
| **阶段 1: 基础设施** | T1, T2 | ✅ T1→T2 串行（T2 依赖 T1） | 0.5 session |
| **阶段 2: 周频因子** | T3, T4, T5 | ✅ 三个可并行 | 0.5 session |
| **阶段 3: PIT + 月频因子** | T6, T7, T8 | ⚠️ T7 可和 T6 并行，T8 依赖 T6 | 1 session |
| **阶段 4: 注册 + 评估 + 脚本** | T9, T10, T11 | ⚠️ T9/T10 可并行，T11 最后 | 1 session |

---

## E. 数据流全景

```
┌──────────────────────────────────────────────────────────────────┐
│              scripts/run_lft_single.py --frequency weekly         │
│         --factor momentum_weekly --start 20240101 --end 20251231  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. FactorDataContext(snapshot_mode="weekly")                      │
│    ├─ ctx.daily → 全量日线 LazyFrame（含 lookback 扩展）           │
│    └─ ctx.snapshot_dates → [2024-01-05, 2024-01-12, ...]         │
│        (每周最后一个交易日)                                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. MomentumWeekly.compute(ctx)                                    │
│    ├─ ctx.daily → sort → shift(20) → momentum 公式 → .collect()  │
│    └─ .filter(trade_date ∈ snapshot_dates) → 周频因子值            │
│    返回: DataFrame[trade_date(周频), ts_code, factor_value]       │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Preprocessing (与 Phase 2 相同管线)                             │
│    mad_clip → fill_median → zscore (周频 截面标准化)              │
└──────────────────────────────┬───────────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
┌───────────────────┐ ┌─────────────────┐ ┌───────────────────┐
│ 4a. IC Analysis   │ │ 4b. Backtest    │ │ 4c. Turnover       │
│ (frequency=weekly)│ │ (frequency=weekly)│ │ (frequency=weekly) │
│ 周频 IC: 因子 vs  │ │ 周频调仓: 快照日  │ │ 周间分组迁移       │
│ 下周累计收益      │ │ 分组 → 下周收益   │ │ (~52 观测/年)     │
└────────┬──────────┘ └────────┬────────┘ └────────┬──────────┘
         │                     │                    │
         └─────────────────────┴────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Console Summary                                                │
│    IC Mean: 0.028  |  IC Std: 0.10  |  IR: 0.28 (weekly)        │
│    Long-Short Sharpe: 1.35 (weekly rebalance)                     │
│    Avg Turnover: 0.42 (weekly, N=52 periods) ⚠️ 周频样本较少     │
└──────────────────────────────────────────────────────────────────┘

─── 月频同理 ───

┌──────────────────────────────────────────────────────────────────┐
│ 1. FactorDataContext(snapshot_mode="monthly")                     │
│    ├─ ctx.monthly_basic → daily_basic 月末快照 LazyFrame          │
│    └─ ctx.snapshot_dates → [2024-01-31, 2024-02-29, ...]         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. PeTtmMonthly.compute(ctx)  /  RoeTtmMonthly.compute(ctx)      │
│    pe_ttm: ctx.monthly_basic → select pe_ttm → .collect()        │
│    roe_ttm: fina_indicator → PIT 对齐 → select roe → .collect() │
│    返回: DataFrame[trade_date(月频), ts_code, factor_value]       │
└──────────────────────────────────────────────────────────────────┘
```

---

## F. 验收标准

### F1. 模块级验收

| # | 验收项 | 预期 |
|---|--------|------|
| 1 | `get_weekly_snapshot_dates("20260101", "20260513")` | 返回约 19 个日期，均为交易日 |
| 2 | `get_monthly_snapshot_dates("20260101", "20260513")` | 返回约 5 个日期，均为交易日 |
| 3 | `ctx.snapshot_dates` when `snapshot_mode="weekly"` | 返回周频日期列表 |
| 4 | `ctx.weekly.collect()` | 行数 = 股票数 × 周数 |
| 5 | `ctx.monthly_basic.collect()` | 含 `pe_ttm`, `pb` 等列 |
| 6 | `MomentumWeekly().compute(ctx)` | 返回周频三列 DataFrame |
| 7 | `VolatilityWeekly().compute(ctx)` | 返回周频三列 DataFrame |
| 8 | `TurnoverWeekly().compute(ctx)` | 返回周频三列 DataFrame |
| 9 | `PIT 对齐验证`：2026-03-31 快照日不含 `ann_date > 2026-03-31` 的财报 | ✓ |
| 10 | `PeTtmMonthly().compute(ctx)` | 返回月频三列 DataFrame（pe_ttm 值） |
| 11 | `PbMonthly().compute(ctx)` | 返回月频三列 DataFrame（pb 值） |
| 12 | `RoeTtmMonthly().compute(ctx)` | 返回月频三列 DataFrame（roe 值，PIT 对齐） |
| 13 | `discover_factors()` | 返回包含 10 个因子的 dict（4 daily + 3 weekly + 3 monthly） |
| 14 | `get_factor("momentum_weekly")` | 返回 `MomentumWeekly` 类 |
| 15 | `compute_rank_ic(factor_df, ret, frequency="weekly")` | IC 周期 = 周数（~52/年） |
| 16 | `run_stratified_backtest(factor_df, ret, frequency="monthly")` | 月频调仓回测结果正确 |

### F2. 端到端验收

- [ ] `python scripts/run_lft_single.py --factor momentum_weekly --frequency weekly --start 20240101 --end 20251231` 运行成功
- [ ] `python scripts/run_lft_single.py --factor pe_ttm --frequency monthly --start 20240101 --end 20251231` 运行成功
- [ ] `python scripts/run_lft_single.py --factor momentum_20d --start 20240101 --end 20251231` 向后兼容（无 --frequency 默认 daily）
- [ ] 周频因子输出文件保存到 `output/lft/factors/weekly/`
- [ ] 月频因子输出文件保存到 `output/lft/factors/monthly/`
- [ ] 控制台输出完整的 IC + 回测 + 换手率摘要

### F3. 代码质量

- [ ] 所有新增公共函数有 docstring（Google style）
- [ ] 所有新增公共函数有 type hints
- [ ] 无循环导入
- [ ] 周频因子不重复实现因子公式（复用日频逻辑）

### F4. 性能验收

| 操作 | 预期时间 |
|------|----------|
| T1: 周频/月频快照日期计算 | < 0.1s |
| T3-T5: 单周频因子计算（全市场 250 天 → ~52 周输出） | < 10s |
| T6: PIT 对齐（5000 股 × 12 月） | < 5s |
| T7: 月频 pe_ttm/pb 计算 | < 3s |
| T8: 月频 roe_ttm + PIT | < 10s |
| T11: 端到端周频评估 | < 60s |
| T11: 端到端月频评估 | < 60s |

---

## G. 文件清单

三期需要创建或修改的所有文件：

| 文件 | 任务 | 操作 | 行数估算 |
|------|------|------|----------|
| `common/calendar.py` | T1 | **修改**（追加 2 函数） | +50 |
| `lft/data/context.py` | T2 | **修改**（追加字段+属性） | +80 |
| `lft/data/pit.py` | T6 | **新建** | ~70 |
| `lft/data/__init__.py` | — | 不变 | 0 |
| `lft/factors/weekly/momentum.py` | T3 | **新建** | ~40 |
| `lft/factors/weekly/volatility.py` | T4 | **新建** | ~40 |
| `lft/factors/weekly/turnover.py` | T5 | **新建** | ~35 |
| `lft/factors/monthly/value.py` | T7 | **新建** | ~50 |
| `lft/factors/monthly/profitability.py` | T8 | **新建** | ~50 |
| `lft/factors/registry.py` | T9 | **修改**（多包发现） | +20 |
| `lft/evaluation/ic_analysis.py` | T10a | **修改**（多频率） | +40 |
| `lft/evaluation/backtest.py` | T10b | **修改**（多频率） | +50 |
| `lft/evaluation/turnover.py` | T10c | **修改**（多频率） | +30 |
| `scripts/run_lft_single.py` | T11 | **修改**（--frequency） | +30 |

**总计约 14 个文件变更（4 新建 + 7 修改），~585 行新增/修改。**

---

## H. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| `fina_indicator` 财务数据未拉取（data/raw/finance/ 为空） | 高 | T8 roe_ttm 因子无法计算 | T8 中检查数据是否存在 → 不存在时优雅降级（warning + 空结果）；脚本先调用 `fetch_finance("fina_indicator", ...)` 预拉取 |
| PIT 对齐的 `ann_date` 字段可能为空 | 中 | 部分财报无法做 PIT 对齐 | `pit_align()` 过滤掉 `ann_date` 为 null 的行，在 docstring 中说明 |
| 月频换手率样本量极少（12 个观测/年） | 高 | 换手率统计不稳健 | 在 `TurnoverResult.summary()` 中明确标注「月频换手率仅 N 个观测，统计不稳健」，不依赖月频换手率做因子筛选决策 |
| 周频因子 `lookback_days=30` 可能不够 | 低 | 区间起点的快照日因子值缺失 | Context 的 `expanded_start` 已经处理扩展——`lookback_days=30` 意味着回看 30 个交易日，足以覆盖 20 日动量计算 |
| Registry 修改导致 Phase 2 因子无法加载 | 低 | Phase 2 脚本运行失败 | `discover_factors()` 默认包列表包含 `lft.factors.daily`，确保向后兼容；修改后先在 Phase 2 脚本上回归测试 |
| 周频/月频 IC 不稳定（截面对比样本少） | 中 | IC 结果波动大 | 在 `ICAanalysisResult.summary()` 中标注截面数（"N=52 weekly periods"），提示用户与日频 IC 做对比时注意样本量差异 |

---

## I. 与后续 Phase 的接口预留

| 后续 Phase | 本期预留 |
|------------|----------|
| Phase 4（因子组合） | 周频/月频因子输出格式与日频一致（三列），组合模块可直接复用 |
| Phase 5（报告模板） | 评估模块的 `@dataclass` + `.summary()` 模式保持一致 |
| 自定义因子扩展 | `lft/factors/custom/` 目录已预留，`category` 字段支持 "custom" |
| MFT 中频因子 | LFT 的多频率架构（daily/weekly/monthly）可作为 MFT 参考 |

---

## J. 快速检查清单（实施前）

- [ ] Phase 2 已基本完成（日频因子 + 预处理 + 评估可以跑通）
- [ ] `data/raw/daily_basic/` 分区数据已拉取（至少覆盖研究区间）
- [ ] `data/raw/finance/` 分区数据已拉取（`fina_indicator`，至少覆盖研究区间 + 1 年）
- [ ] `common/calendar.py` 的 `get_trade_dates()` 功能正常

---

