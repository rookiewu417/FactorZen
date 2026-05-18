## Task 1: cross_sectional_zscore zero-std fix (2026-05-14)

### Bug
When all factor values in a cross-section are identical, std=0 causes division by zero → inf/NaN.

### Fix
Used pl.when(std == 0).then(0.0).otherwise(...) pattern to guard against zero std.

Key observations:
- Polars .std() with ddof=1 returns 0.0 for groups with identical values (e.g., [5.0, 5.0, 5.0])
- For single-element groups, .std() returns null (not 0), but the pl.when(null) is falsy so it falls to .otherwise() which does (val - mean) / null → null, not inf/NaN
- The fix handles both cases correctly: std=0 → 0.0, null std → null (no crash)

### Test file: tests/test_normalizer.py
- test_zero_std: 3 identical values → all 0.0
- test_single_stock: 1 stock → no crash, returns reasonable value
- test_normal_case: varying values → correct z-scores

### Status
All 3 tests pass. Fix verified.

## Task 5-11: 7 advanced single-factor evaluation functions (2026-05-14)

### File created: lft/evaluation/advanced.py (~550 lines)

### 7 Dataclasses + 7 Functions implemented:

1. **ICDecayResult + compute_ic_decay** — per-horizon IC decay. Auto-detects horizons from `fwd_ret_*d` column names. Returns `list[ICDecayResult]`.

2. **MonotonicityResult + compute_monotonicity** — quantile return monotonicity. Uses rank ordinal → qcut grouping (pattern from backtest.py). Computes monotonicity_score = fraction of consecutive quantile pairs with consistent direction. Includes OLS slope.

3. **SectorICResult + compute_sector_ic** — sector-stratified IC. Single DataFrame with factor_col, ret_col, sector_col. Returns pl.DataFrame by default, SectorICResult with `return_object=True`.

4. **SizeICResult + compute_size_ic** — size-stratified IC. Uses cap_col rank → qcut into n_buckets (default 3: Small/Mid/Large). Returns pl.DataFrame (cap_bucket, ic) by default.

5. **CrowdingResult + compute_factor_crowding** — multi-factor crowding. Takes `dict[str, pl.DataFrame]` (like correlation.py). Computes cross-sectional correlation matrix averaged across dates. Scores >0.7=High, >0.4=Moderate. Marked EXPERIMENTAL.

6. **MarketRegimeICResult + compute_market_regime_ic** — regime-stratified IC. Supports "direction" (up/down) and "volatility" modes. Auto-computes market return if market_df not provided.

7. **RankAutocorrResult + compute_rank_autocorr** — rank autocorrelation. Ranks per date → Spearman between consecutive periods. Has `get_lag(lag)` method. Computes half_life = -ln(2)/ln(autocorr).

### Key test discrepancies from plan:

- Tests use single-DataFrame pattern (factor_col + ret_col in same df) rather than separate factor_df + daily_ret
- `compute_ic_decay` returns `list[ICDecayResult]` (one per horizon), not single dict
- `compute_factor_crowding` takes `dict[str, pl.DataFrame]` (multi-factor), not single factor with top/bottom groups
- `compute_rank_autocorr` supports `lags` list, `autocorr_values` has one per lag, `get_lag()` method

### Bug fix: test_advanced.py helper

The original test used `pl.lit([list], dtype=pl.Float64)` which the newer Polars rejects (list + typed dtype). Fixed to use `pl.Series(name, list, dtype=pl.Float64)`.

### Test results

ALL 38 tests pass (29 advanced eval + 9 existing). No regressions.
Warnings: factor_crowding synthetic test data has identical factor_clean=0.5 for "momentum" factor, causing np.corrcoef divide-by-zero — expected with synthetic data, harmless.

## Task 12: MFT 框架基础 — MFTFactor + MFTDataContext

**日期**: 2026-05-14
**结果**: 全部通过 ✅

**创建文件**:
| 文件 | 说明 |
|------|------|
| `mft/factors/base.py` | `MFTFactor(ABC)` 抽象基类，含 `compute()` / `validate()` |
| `mft/data/context.py` | `MFTDataContext` dataclass，惰性加载分钟线数据 |
| `tests/test_mft_factor_base.py` | 8 个测试：抽象类验证、默认属性、validate 各类场景 |
| `tests/test_mft_data_context.py` | 7 个测试：构造、expanded_start、required_data 校验、惰性加载、universe 过滤 |

**设计要点**:
- `MFTFactor` 与 `LFTFactor` 结构对齐，差异在于 `frequency="minute"`、`bar_size="1min"`、`lookback_bars=500`（取代 lookback_days）
- `MFTDataContext.minute` 使用 `load_parquet("minute", ..., date_col="trade_time")` — 分钟线以 `trade_time` 为时间戳列
- `expanded_start` 向前减 5 天（确保有足够回溯 bar），与 LFT 的 `prev_trade_date` 范式不同但原理一致
- 惰性加载 + 缓存模式与 `FactorDataContext` 一致：`_minute` 仅在首次 `.minute` 访问时填充

**测试策略**:
- 因子测试全部使用合成 DataFrame，无需真实数据
- 数据上下文测试使用 `unittest.mock.patch` 替换 `load_parquet`，避免空白录扫描失败
- `test_validate_missing_factor_column` 覆盖了 `factor_value` 列缺失的边界情况

**关键发现**:
- `pl.scan_parquet()` 在空目录上生成 LazyFrame 时不会立即报错（仅在 `.collect()` 时失败），但为测试稳定性选择了 mock 方案
- `MFTDataContext` 比 `FactorDataContext` 更精简：只管理 minute 一种数据源，无 weekly/monthly 快照下采样
- MFT 相关 `__init__.py` 存根在 Task 0-11 中已就位，本次无需修改

## Task 15b: MFT Demo Factor — Momentum1Min (2026-05-14, Sisyphus-Junior)

**结果**: 全部通过 ✅ 6 new tests + 87 total (0 regressions)

**创建的文件**:
| 文件 | 描述 |
|------|------|
| mft/factors/demo/__init__.py | demo 包入口 |
| mft/factors/demo/momentum_1min.py | @dataclass Momentum1Min(MFTFactor) — 1分钟 bar 5-bar 动量 |
| tests/test_mft_demo.py | 6 tests: compute 正确性 / null 过滤 / 列验证 / registry 发现 / 默认属性 / 类型检查 |

**测试用例** (tests/test_mft_demo.py, 6 tests, 0.18s):
- `test_momentum_1min_compute()` — 2 只股票各 10 条 bar，验证 5-bar 动量精确值
- `test_momentum_1min_no_null()` — 确认 filter(is_not_null) 生效
- `test_momentum_1min_columns()` — 输出三列 trade_time/ts_code/factor_value
- `test_registry_can_find_momentum_1min()` — registry 从 mft.factors.demo 自动发现
- `test_factor_default_attributes()` — name/bar_size/lookback_bars/frequency/description
- `test_factor_is_mftfactor()` — issubclass 检查

**关键决策**:
- 因子类必须加 `@dataclass` 装饰器，因为父类 `MFTFactor` 是 dataclass。不加的话 class-level 属性赋值不会传给 dataclass `__init__`，导致 `name=""` 
- 与 LFT 模式不同：LFTFactor 不是 dataclass，子类直接用 `name = "..."` 即可；MFTFactor 是 dataclass，子类必须 `@dataclass` + 类型标注
- 测试中 mock `ctx.minute` 必须用 `PropertyMock(return_value=...)` 而非 `Mock(return_value=...)`，因为 `ctx.minute` 是 property 而非 callable
- 合成数据关闭格线性递增（10..19 和 100,102..118），便于手工验证动量公式

## Task: reporting/tear_sheet.py — Jinja2 模板 Bug 修复 (2026-05-14)

### Bug
模板 `tear_sheet.html` 中对 `metrics` dict 使用属性访问语法（如 `metrics.sector_ic`、`metrics.decay_table`），当 key 不存在时，Jinja2 的 `getattr(obj, attr)` 对 `dict` 对象抛出 `UndefinedError`：
```
jinja2.exceptions.UndefinedError: 'dict object' has no attribute 'sector_ic'
```
Jinja2 在 `obj.attr` 查找链中是先 `getattr` 再 `__getitem__`，但对于 plain dict，`getattr` 会先失败并直接抛异常，而非回退到 `__getitem__`。

### Fix
将所有条件性 key（可能不存在于 metrics dict 中的 key）改为使用 `metrics.get('key')` 而非 `metrics.key`：
- `metrics.decay_table` → `metrics.get('decay_table')`
- `metrics.sector_ic` → `metrics.get('sector_ic')`
- `metrics.size_buckets` → `metrics.get('size_buckets')`
- `metrics.monotonicity_score` → `metrics.get('monotonicity_score')`
- `metrics.rank_autocorr` → `metrics.get('rank_autocorr')`
- `metrics.half_life` → `metrics.half_life or 0` (在 `{% if %}` 块内安全)

### 受影响文件
- `reporting/templates/tear_sheet.html` — 6 处修改（第 91/114/120/129/138/139 行附近）

### 新增测试
- `test_generate_html_contains_factor_name` — 验证 HTML 含 `<html>` 标签和因子名
- `test_html_size_under_5mb` — 验证 HTML < 5MB

### 测试结果
ALL 16 tests pass (14 original + 2 new). No regressions.

### 项目路径说明
正确的工作目录是 `E:\code\量化研究\因子研究\`（非 `E:\code\量化研究\量化研究\`）。
后者包含旧的重复文件，`lft/evaluation/` 等模块仅存在于 `因子研究\` 下。
