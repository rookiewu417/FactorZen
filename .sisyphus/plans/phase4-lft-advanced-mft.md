# Phase 4-6: LFT 单因子评估增强 + MFT/HFT 框架 + 详细因子报告

> **状态**: ✅ 已完成
> **创建时间**: 2026-05-14
> **前置条件**: Phase 1（common/ 基础设施）✅ | Phase 2（LFT 日频因子）✅ | Phase 3（LFT 周频/月频因子）✅

---

## TL;DR

> **Quick Summary**: 在已有 LFT 完整管线基础上升级单因子评估能力（7 个高级指标），搭建 MFT 分钟频框架（骨架+Demo），为 HFT 留桩，最后构建标准化 6 面板 Tear Sheet 报告输出。全部采用 TDD。
>
> **Deliverables**:
> - `lft/evaluation/advanced.py` — 7 个高级评估指标（IC Decay / Monotonicity / Sector IC / Size IC / Crowding / Market Regime IC / Rank Autocorrelation）
> - Bug 修复 3 项（normalizer std=0 / pit_align 性能 / CSI 指数成分股）
> - `mft/` 完整框架骨架 + 1 个 Demo 因子
> - `hft/` 占位桩
> - `reporting/` 6 面板 Tear Sheet 报告引擎 + CLI
> - `tests/` 测试体系（目标：新代码 ≥80%，评估模块 ≥60%）
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: Task 0(数据验证) → Task 1-3(Bug修复) → Task 4(TDD桩) → Task 5-10(高级评估) → Task 15-16(报告) → Task 17(最终QA)

---

## Context

### Original Request
用户要求完善项目功能：增强单因子评估、搭建 MFT/HFT 框架、构建详细报告系统。用户明确：现有 10 个因子仅作测试用，实际因子库自行构建；不涉及多因子模型（Fama-MacBeth/Barra/组合优化）。

### Interview Summary
**Key Discussions**:
- 战略方向：不做因子本身，做平台/框架功能完善
- Phase 4: 单因子评估增强 + Bug 修复 + TDD
- Phase 5: MFT 分钟频框架 + HFT Demo
- Phase 6: 详细因子报告（Tear Sheet + HTML + 图表）
- 不涉及：多因子模型、新因子构造、因子组合/合成

**Research Findings**:
- Librarian 研究确认：Alphalens 标准包含 IC Decay、Monotonicity、Rank Autocorrelation、Quantile Turnover、Sector/Size 分层分析
- Qlib 4 层架构：Infrastructure → Alpha Research → Risk Model → Execution（当前处于 Layer 2）
- MFT 与 LFT 差异：数据量（2.88亿行/年）、存储分区（按日+股票）、预处理（微结构噪声）、成本敏感

### Metis Review
**Identified Gaps** (addressed):
- 数据可用性验证：MFT 分钟数据 Tushare 权限需预确认 → Task 0
- Bug 修复必须先于新功能 → Wave 2
- 高级评估不修改现有文件，全部放新文件 → `lft/evaluation/advanced.py`
- 因子拥挤度定义为 MVP 单指标 → 分位内成对相关性
- MFT 框架仅骨架+Demo → 1 个因子，10 只股票，1 个月
- HFT 桩仅占位 → NotImplementedError
- Phase 6 报告为静态 HTML + 内嵌 PNG → 不用交互式 JS
- 不新增 pip 依赖

---

## Work Objectives

### Core Objective
在 LFT 完整管线基础上，补全「高级单因子评估 → MFT/HFT 框架 → 标准化报告」三大能力缺口，使平台成为可独立使用的量化因子研究框架。

### Concrete Deliverables
- `lft/evaluation/advanced.py` — 7 个高级评估指标模块
- `lft/preprocessing/normalizer.py` — std=0 除零修复
- `lft/data/pit.py` — join_asof 性能重构
- `common/universe.py` — CSI300/500/800 动态指数成分股
- `mft/factors/base.py` + `mft/data/context.py` — MFT 框架层
- `mft/preprocessing/pipeline.py` — 分钟频预处理桩
- `mft/factors/demo/momentum_1min.py` — Demo 因子
- `mft/evaluation/__init__.py` — 评估桩
- `hft/__init__.py` — HFT 占位类
- `reporting/tear_sheet.py` + `reporting/templates/tear_sheet.html` — 报告引擎
- `scripts/generate_report.py` — 报告 CLI
- `tests/` — 完整测试体系（14+ 测试文件）

### Definition of Done
- [ ] `pixi run pytest tests/ -v` 全部通过，新代码覆盖率 ≥ 80%
- [ ] `pixi run python scripts/generate_report.py --factor momentum_20d --start 20250101 --end 20250513` 生成有效 HTML（< 5MB）
- [ ] `pixi run python -c "from mft.data.context import MFTDataContext; print('OK')"` 无报错
- [ ] `pixi run python -c "from hft import HFTFactor; HFTFactor().compute(None)"` 抛出 NotImplementedError

### Must Have
- 所有新评估指标独立于现有模块（不修改 ic_analysis.py/backtest.py 等签名）
- MFT 框架可独立运行（不破坏 LFT 管线）
- TDD：每个新 .py 文件对应一个 test_*.py
- Bug 修复带回归测试

### Must NOT Have (Guardrails)
- ❌ 不修改 `ic_analysis.py`/`backtest.py`/`turnover.py` 现有签名
- ❌ 不实现 MFT 微观结构预处理（买卖价差反弹、异步性校正）
- ❌ 不实现 MFT 完整评估管线（回测/IC/换手率）
- ❌ 不为 HFT 桩写任何实现代码
- ❌ 不添加新的 pip 依赖
- ❌ 不为同一个因子创建多个报告变体
- ❌ 因子拥挤度仅为单一 MVP 指标，不扩展为多指标体系
- ❌ Phase 6 报告不使用交互式 JavaScript

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest in pixi.toml, empty tests/ directory)
- **Automated tests**: TDD (RED-GREEN-REFACTOR)
- **Framework**: pytest
- **Coverage target**: 新代码 ≥80%，评估模块 ≥60%

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Backend/Module**: Use Bash (bun/node REPL or python -c) — Import, call functions, compare output
- **CLI/TUI**: Use Bash (pixi run) — Run command, validate output, check exit code
- **API/Backend**: Use Bash (curl) — Send requests, assert status + response fields

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (数据验证 - 先决条件):
├── Task 0: Tushare 数据可用性验证 [quick]

Wave 1 (Bug 修复 - 高优先级，相互独立):
├── Task 1: normalizer.py std=0 修复 [quick]
├── Task 2: pit_align 性能重构 [deep]
└── Task 3: CSI 指数成分股动态加载 [deep]

Wave 2 (TDD 桩 + 高级评估 - 最大并行):
├── Task 4: tests/ 目录桩文件创建 [quick]
├── Task 5: IC Decay 增强分析 [deep]
├── Task 6: Monotonicity 分析 [deep]
├── Task 7: Sector-stratified IC [deep]
├── Task 8: Size-stratified IC [deep]
├── Task 9: Factor Crowding 检测 [deep]
├── Task 10: Market Regime IC [deep]
└── Task 11: Rank Autocorrelation [deep]

Wave 3 (MFT/HFT 框架 - 依赖 Wave 0):
├── Task 12: MFT Factor 基类 + DataContext [deep]
├── Task 13: MFT 预处理管线桩 [deep]
├── Task 14: MFT Demo 因子 [deep]
├── Task 15: MFT 评估桩 + HFT 桩 [quick]
└── Task 16: MFT 注册与发现 [quick]

Wave 4 (报告 - 依赖 Wave 1 + Wave 2):
├── Task 17: Tear Sheet 报告引擎 [visual-engineering]
├── Task 18: 报告 CLI 脚本 [writing]
└── Task 19: 报告模板 HTML/CSS [visual-engineering]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 0 → Task 1-3 → Task 4 → Task 5-11 → Task 17-18 → Task 20 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 8 (Wave 2)
```

### Agent Dispatch Summary

- **0**: **1** — T0 → `quick` (tushare)
- **1**: **3** — T1 → `quick`, T2 → `deep`, T3 → `deep` (tushare)
- **2**: **8** — T4 → `quick`, T5-T11 → `deep`
- **3**: **5** — T12-T16 → `deep`/`quick`
- **4**: **3** — T17 → `visual-engineering`, T18 → `writing`, T19 → `visual-engineering`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

---

### Wave 0: 数据可用性验证（先决条件）

- [x] 0. Tushare 数据可用性验证

  **What to do**:
  - 验证 `stk_mins` API 是否可用（免费层限制、历史回溯窗口、单次最大行数）
  - 验证 `data/raw/daily_basic/` 和 `data/raw/finance/` Parquet 分区覆盖目标日期范围
  - 验证 `stock_basic.industry` 字段内容（是否申万分类、是否有 null、"未知"等异常值）
  - 对 1 只代表性股票（如 `000001.SZ`）测试 `stk_mins` 调用，记录返回行数和耗时
  - 输出结果到 `data/data_availability_report.txt`

  **Must NOT do**:
  - 不要在此步骤拉取全量分钟数据
  - 不要修改任何代码文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`tushare`]
  - **Reason**: 数据验证任务，需要 Tushare 技能理解 API 限制

  **Parallelization**:
  - **Can Run In Parallel**: NO（先决条件，阻塞所有后续 Wave）
  - **Blocks**: Task 1-19（所有后续任务）

  **QA Scenarios**:
  ```
  Scenario: 验证 stk_mins API 可用性
    Tool: Bash
    Steps:
      1. pixi run python -c "from common.loader import init_tushare; pro=init_tushare(); df=pro.stk_mins(ts_code='000001.SZ', freq='1min', start_date='20260501', end_date='20260510'); print(f'Rows: {len(df)}' if df is not None and not df.empty else 'EMPTY')"
      2. 检查输出不为 EMPTY，行数 > 0
    Expected Result: 打印行数，无报错
    Evidence: .sisyphus/evidence/task-0-stk-mins-check.txt

  Scenario: 验证 daily_basic Parquet 缓存
    Tool: Bash
    Steps:
      1. pixi run python -c "from pathlib import Path; from config.settings import DATA_RAW; d=DATA_RAW/'daily_basic'; print('EXISTS' if d.exists() else 'MISSING'); [print(p) for p in sorted(d.rglob('*.parquet'))[:10]]"
      2. 确认至少存在 2025 年的数据
    Expected Result: EXISTS + 至少列出数个 parquet 文件
    Evidence: .sisyphus/evidence/task-0-daily-basic-check.txt
  ```

  **Commit**: NO（仅验证，不产出代码）

---

### Wave 1: Bug 修复（高优先级，相互独立）

- [x] 1. Bug 修复：normalizer.py std=0 除零保护

  **What to do**:
  - 在 `cross_sectional_zscore()` 中添加防护：当截面 std=0 时返回 0.0 而非 inf
  - 实现方式：`(col - mean) / std.fill_nan(0).replace({float('inf'): 0})` 或使用 `when/then/otherwise`
  - 创建 `tests/test_normalizer.py`，包含：
    - `test_zero_std()` — 3 只股票因子值全为 5.0 → 所有 zscore = 0.0
    - `test_single_stock()` — 截面仅 1 只股票 → 优雅降级不崩溃
    - `test_normal_case()` — 验证正常标准化结果与预期一致

  **Must NOT do**:
  - 不要修改函数签名
  - 不要改变正常数据的输出结果

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 单文件小修改 + 简单测试，快速任务

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2, 3 并行）
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 17（报告依赖正确的标准化输出）
  - **Blocked By**: Task 0（需确认环境可用）

  **References**:
  - `lft/preprocessing/normalizer.py:12` — 当前实现，需修复 `(pl.col(col) - mean) / std` 行
  - `lft/preprocessing/pipeline.py:49-54` — `quick_preprocess()` 调用链确认不破坏

  **QA Scenarios**:
  ```
  Scenario: std=0 时返回 0
    Tool: Bash (pixi run python -c)
    Steps:
      1. pixi run python -c "
  import polars as pl
  from lft.preprocessing.normalizer import cross_sectional_zscore
  df = pl.DataFrame({'trade_date': ['20240101']*3, 'ts_code': ['A','B','C'], 'val': [5.0,5.0,5.0]})
  result = cross_sectional_zscore(df, col='val')
  print(result['val_z'].to_list())
  "
      2. 断言输出为 [0.0, 0.0, 0.0]（非 NaN/inf）
    Expected Result: [0.0, 0.0, 0.0]
    Evidence: .sisyphus/evidence/task-1-zero-std.txt

  Scenario: 单股票不崩溃
    Tool: Bash
    Steps:
      1. pixi run python -c "
  import polars as pl
  from lft.preprocessing.normalizer import cross_sectional_zscore
  df = pl.DataFrame({'trade_date': ['20240101'], 'ts_code': ['A'], 'val': [5.0]})
  result = cross_sectional_zscore(df, col='val')
  print(result['val_z'].to_list())
  "
      2. 不抛异常，输出合理的值
    Expected Result: 不崩溃，输出列表
    Evidence: .sisyphus/evidence/task-1-single-stock.txt
  ```

  **Commit**: YES
  - Message: `fix(normalizer): handle std=0 in cross_sectional_zscore`
  - Files: `lft/preprocessing/normalizer.py`, `tests/test_normalizer.py`
  - Pre-commit: `pixi run pytest tests/test_normalizer.py -v`

- [x] 2. Bug 修复：pit_align 笛卡尔积性能重构

  **What to do**:
  - 将 `pit_align()` 中的笛卡尔积 join 替换为 `join_asof` 或按快照日分块后过滤 join
  - 推荐方案：对每个 `snapshot_date`，过滤 `fina_df` 中 `ann_date <= snapshot_date`，再 group_by `ts_code` 取 `end_date.max()`
  - 创建 `tests/test_pit.py`，包含：
    - `test_pit_align_correctness()` — 验证 PIT 对齐结果正确（不引入未来信息）
    - `test_pit_align_performance()` — 对 10 年模拟财务数据，处理时间 < 5 秒
    - `test_pit_align_empty()` — 无匹配财报时优雅返回空 DataFrame

  **Must NOT do**:
  - 不要改变函数签名（`pit_align(fina_df, snapshot_dates)`）
  - 不要修改调用方（`roe_ttm` 因子中的用法）

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 需要深入理解 Polars join 语义 + 性能分析

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocked By**: Task 0

  **References**:
  - `lft/data/pit.py:1-42` — 当前实现
  - `lft/factors/monthly/profitability.py` — `roe_ttm` 调用方，确认不破坏
  - Polars `join_asof` 文档：`https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html`

  **QA Scenarios**:
  ```
  Scenario: PIT 对齐正确性验证
    Tool: Bash (pixi run python -c)
    Steps:
      1. pixi run python -c "
  from lft.data.pit import pit_align
  import polars as pl
  from datetime import date
  fina = pl.DataFrame({
      'ts_code': ['A','A','A'],
      'end_date': [date(2024,3,31), date(2024,6,30), date(2024,9,30)],
      'ann_date': [date(2024,4,15), date(2024,7,15), date(2024,10,15)],
      'roe': [0.05, 0.06, 0.07]
  })
  snapshots = [date(2024,6,15)]
  result = pit_align(fina, snapshots)
  print(result['end_date'].to_list())
  "
      2. 断言 end_date = [2024-06-30]（6 月快照只能用 6 月财报，不能用 9 月）
    Expected Result: [date(2024, 6, 30)] — 不含未来信息
    Evidence: .sisyphus/evidence/task-2-correctness.txt
  ```

  **Commit**: YES
  - Message: `perf(pit): replace cartesian join with efficient PIT alignment`
  - Files: `lft/data/pit.py`, `tests/test_pit.py`
  - Pre-commit: `pixi run pytest tests/test_pit.py -v`

- [x] 3. Bug 修复：CSI300/500/800 指数成分股动态加载

  **What to do**:
  - 在 `universe.py` 中实现 `_load_index_members(index_code, date)` 内部函数，调用 Tushare `index_member` API
  - 修改 `get_universe()` 中 `csi300`/`csi500`/`csi800` 的逻辑：从 fallback `all_a` 改为动态拉取
  - 添加缓存：按指数代码 + 月份缓存到 `data/cache/index_member_{code}_{YYYYMM}.parquet`
  - 优雅降级：API 调用失败时 fallback 到 `all_a` 并记录 warning
  - 创建 `tests/test_universe.py`，包含：
    - `test_get_index_members_csi300()` — 验证返回成分股列表非空
    - `test_index_fallback()` — 模拟 API 失败时的降级行为

  **Must NOT do**:
  - 不要修改现有 LFT 管线的默认 universe 行为
  - 不要在因子计算中硬编码指数成分股

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: [`tushare`]
  - **Reason**: 涉及 Tushare API 调用 + 缓存策略

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocked By**: Task 0

  **References**:
  - `common/universe.py:85-105` — CSI 指数当前 fallback 实现
  - `common/loader.py:451-503` — `fetch_stock_basic()` 缓存模式可参考
  - Tushare `index_member` 接口文档

  **QA Scenarios**:
  ```
  Scenario: 验证 CSI300 成分股加载
    Tool: Bash (pixi run python -c)
    Steps:
      1. pixi run python -c "
  from common.universe import get_universe
  csi = get_universe('20260501', 'csi300')
  print(f'CSI300: {len(csi)} stocks')
  print(csi['ts_code'].head(5).to_list())
  "
      2. 确认返回 250-300 只股票（非 5515 全A）
    Expected Result: 200-350 stocks，列表包含沪深300成分股
    Evidence: .sisyphus/evidence/task-3-csi300.txt

  Scenario: CSI800 = CSI300 + CSI500
    Tool: Bash
    Steps:
      1. pixi run python -c "
  from common.universe import get_universe
  c8 = get_universe('20260501', 'csi800')
  c3 = get_universe('20260501', 'csi300')
  c5 = get_universe('20260501', 'csi500')
  print(f'CSI800:{len(c8)} CSI300:{len(c3)} CSI500:{len(c5)}')
  "
      2. 确认 CSI800 数量 ≈ CSI300 + CSI500（允许少量交集重叠）
    Expected Result: CSI800 数量大于单一指数但不超过两者之和
    Evidence: .sisyphus/evidence/task-3-csi800.txt
  ```

  **Commit**: YES
  - Message: `feat(universe): implement dynamic CSI300/500/800 index member loading`
  - Files: `common/universe.py`, `tests/test_universe.py`
  - Pre-commit: `pixi run pytest tests/test_universe.py -v`

---

### Wave 2: TDD 桩 + 高级评估指标（最大并行 8 个任务）

- [x] 4. TDD 桩：创建所有测试文件框架

  **What to do**:
  - 在 `tests/` 下创建以下文件，每个文件包含至少 1 个失败测试（RED）：
    - `tests/test_advanced.py` — IC Decay 测试（Task 5）
    - `tests/test_monotonicity.py` — Monotonicity 测试（Task 6）
    - `tests/test_sector_ic.py` — Sector IC 测试（Task 7）
    - `tests/test_size_ic.py` — Size IC 测试（Task 8）
    - `tests/test_factor_crowding.py` — Crowding 测试（Task 9）
    - `tests/test_market_regime_ic.py` — Market Regime IC 测试（Task 10）
    - `tests/test_rank_autocorr.py` — Rank Autocorrelation 测试（Task 11）
  - 每个测试文件用合成数据构造：已知输入 → 预期输出
  - 运行 `pixi run pytest tests/ -v` 确认所有新测试 FAIL（红色状态）

  **Must NOT do**:
  - 不要在此任务中实现任何评估函数
  - 不要在测试文件中 import 尚不存在的模块（用 `pytest.importorskip` 或直接 import 让测试失败）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 模板化创建测试桩文件，机械性工作

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Task 5-11 的前半部分并行）
  - **Blocks**: Task 5-11（作为 TDD RED 阶段的前置）
  - **Blocked By**: Task 0

  **QA Scenarios**:
  ```
  Scenario: 所有测试桩文件存在且测试失败
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_advanced.py tests/test_monotonicity.py tests/test_sector_ic.py tests/test_size_ic.py tests/test_factor_crowding.py tests/test_market_regime_ic.py tests/test_rank_autocorr.py -v --tb=line
      2. 确认 7 个测试文件都被发现，所有测试状态为 FAILED
    Expected Result: 7 files discovered, all tests FAILED (RED phase)
    Evidence: .sisyphus/evidence/task-4-tdd-red.txt
  ```

  **Commit**: YES
  - Message: `test(tdd): create test stubs for 7 advanced evaluation metrics`
  - Files: `tests/test_advanced.py`, `tests/test_monotonicity.py`, `tests/test_sector_ic.py`, `tests/test_size_ic.py`, `tests/test_factor_crowding.py`, `tests/test_market_regime_ic.py`, `tests/test_rank_autocorr.py`

- [x] 5. 高级评估：IC Decay 增强分析

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_ic_decay_analysis()` 函数
  - 输出 `ICDecayResult` dataclass：`{horizon_days: (ic_mean, ic_std, ir, pos_ratio)}` 映射 + `is_monotonic_decay` 标志 + `warnings` 列表
  - 支持至少 4 个时间范围：1d, 5d, 10d, 20d（可配置 `horizons` 参数）
  - 检测 IC 是否单调衰减：如果 IC_10d > IC_5d，添加 warning "Non-monotonic IC decay detected"
  - 实现 `summary()` 方法输出格式化字符串
  - 实现 `tests/test_advanced.py` 中的测试（从 RED → GREEN）

  **Must NOT do**:
  - 不要在 `ic_analysis.py` 中添加代码
  - 不要修改 `ICAnalysisResult` 的现有字段

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 涉及统计计算 + dataclass 设计

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Task 6-11 并行）
  - **Blocked By**: Task 4（TDD 桩）
  - **Blocks**: Task 17（报告）

  **References**:
  - `lft/evaluation/ic_analysis.py:12-36` — `compute_fwd_returns()` 前向收益预计算模式
  - `lft/evaluation/ic_analysis.py:64-142` — `compute_rank_ic()` IC 计算模式

  **QA Scenarios**:
  ```
  Scenario: IC Decay 验证单调衰减
    Tool: Bash (pixi run pytest)
    Steps:
      1. pixi run pytest tests/test_advanced.py::test_ic_decay_monotonic -v
    Expected Result: PASS — 验证合成数据（monotonic decay）输出 is_monotonic_decay=True
    Evidence: .sisyphus/evidence/task-5-ic-decay.txt

  Scenario: IC Decay 检测非单调
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_advanced.py::test_ic_decay_non_monotonic -v
    Expected Result: PASS — 非单调数据产生 warnings
    Evidence: .sisyphus/evidence/task-5-non-monotonic.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add IC Decay enhanced analysis to advanced metrics`
  - Files: `lft/evaluation/advanced.py`, `tests/test_advanced.py`
  - Pre-commit: `pixi run pytest tests/test_advanced.py -v`

- [x] 6. 高级评估：Monotonicity（分位单调性）分析

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_monotonicity()` 函数
  - 输出 `MonotonicityResult` dataclass：分位收益向量（n_groups 个元素的 list）、单调性得分（0.0-1.0，基于连续分位间收益方向一致性）、OLS 斜率（最低分位→最高分位的线性拟合）
  - 分组逻辑复用 `backtest.py` 的分位排名模式（`rank ordinal → qcut`）
  - 实现测试：`tests/test_monotonicity.py` 从 RED → GREEN

  **Must NOT do**:
  - 不要在 `backtest.py` 中添加代码

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 需要理解分位收益计算 + OLS 统计

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 4
  - **Blocks**: Task 17

  **References**:
  - `lft/evaluation/backtest.py:48-53` — 分位排名 + group 分配模式（复用）
  - Alphalens `mean_return_by_quantile()` 概念

  **QA Scenarios**:
  ```
  Scenario: 完美单调因子得分为 1.0
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_monotonicity.py::test_perfect_monotonic -v
    Expected Result: PASS — score=1.0, positive OLS slope
    Evidence: .sisyphus/evidence/task-6-perfect.txt

  Scenario: 随机因子得分接近 0
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_monotonicity.py::test_random_monotonic -v
    Expected Result: PASS — score < 0.3
    Evidence: .sisyphus/evidence/task-6-random.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add Monotonicity (quantile return) analysis`
  - Files: `lft/evaluation/advanced.py`, `tests/test_monotonicity.py`
  - Pre-commit: `pixi run pytest tests/test_monotonicity.py -v`

- [x] 7. 高级评估：Sector-stratified IC（行业分层 IC）

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_sector_ic()` 函数
  - 使用 `stock_basic.industry` 作为行业标签（合并因子 + 收益 + 行业后分行业计算 Rank IC）
  - 输出 `SectorICResult` dataclass：`{sector_name: ICAnalysisResult}` 映射 + `n_low_sample_warnings` 列表（样本量 < 30 的行业触发警告）
  - 已知限制：`stock_basic.industry` 是静态分类（不含时点信息），在 docstring 中标注此前视偏差风险
  - 实现测试：`tests/test_sector_ic.py` 从 RED → GREEN

  **Must NOT do**:
  - 不要在 `ic_analysis.py` 中添加代码
  - 不要引入新的行业分类数据源

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 涉及多分组统计 + 数据合并

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 3（CSI 成分股修复）+ Task 4
  - **Blocks**: Task 17

  **References**:
  - `lft/preprocessing/neutralizer.py:40-80` — 行业分类获取模式（`stock_basic.industry` 列）
  - `lft/evaluation/ic_analysis.py:64-142` — `compute_rank_ic()` 复用思路

  **QA Scenarios**:
  ```
  Scenario: 行业分层 IC 输出行业→IC映射
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_sector_ic.py::test_sector_ic_output -v
    Expected Result: PASS — 验证输出包含 {sector: ICAnalysisResult} 字典
    Evidence: .sisyphus/evidence/task-7-sector-ic.txt

  Scenario: 小样本行业触发 warning
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_sector_ic.py::test_low_sample_warning -v
    Expected Result: PASS — 样本量 < 30 的行业产生警告
    Evidence: .sisyphus/evidence/task-7-low-sample.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add Sector-stratified IC analysis`
  - Files: `lft/evaluation/advanced.py`, `tests/test_sector_ic.py`
  - Pre-commit: `pixi run pytest tests/test_sector_ic.py -v`

- [x] 8. 高级评估：Size-stratified IC（市值分层 IC）

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_size_ic()` 函数
  - 使用 `total_mv`（来自 `daily_basic`）将股票按市值三分位数分为 Large/Mid/Small 三组
  - 分三组各自计算 Rank IC，输出 `SizeICResult` dataclass：`{size_group: ICAnalysisResult}` 映射
  - 支持配置：`split_method="tercile"`（三分位数）或 `split_method="median"`（中位数二分）
  - 实现测试：`tests/test_size_ic.py` 从 RED → GREEN

  **Must NOT do**:
  - 不要在 `ic_analysis.py` 中添加代码

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 涉及双数据源合并（因子 + daily_basic）

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 3 + Task 4
  - **Blocks**: Task 17

  **References**:
  - `common/loader.py:208-277` — `fetch_daily_basic()` 提供 `total_mv` 列

  **QA Scenarios**:
  ```
  Scenario: 市值分层 IC 输出三组结果
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_size_ic.py::test_size_ic_three_groups -v
    Expected Result: PASS — 输出包含 Large/Mid/Small 三个 key
    Evidence: .sisyphus/evidence/task-8-size-ic.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add Size-stratified IC analysis`
  - Files: `lft/evaluation/advanced.py`, `tests/test_size_ic.py`
  - Pre-commit: `pixi run pytest tests/test_size_ic.py -v`

- [x] 9. 高级评估：Factor Crowding（因子拥挤度）检测

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_factor_crowding()` 函数
  - MVP 实现：计算极端分位（top/bottom 组）股票之间历史收益的成对平均相关性
  - 输出 `CrowdingResult` dataclass：`crowding_score`（0.0-1.0）+ `interpretation`（"Low"/"Moderate"/"High"）+ `warnings`
  - 得分 > 0.7 标记为 "High" 并触发 warning
  - 标记为 **实验性指标**（docstring + summary 中注明）
  - 实现测试：`tests/test_factor_crowding.py` 从 RED → GREEN

  **Must NOT do**:
  - 不要实现多个拥挤度指标（仅此一个 MVP 指标）
  - 不要声称此指标有学术/行业标准支持

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 需要创新的统计方法设计

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 4
  - **Blocks**: Task 17

  **QA Scenarios**:
  ```
  Scenario: 高相关因子产生高拥挤度得分
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_factor_crowding.py::test_high_crowding -v
    Expected Result: PASS — crowding_score > 0.5
    Evidence: .sisyphus/evidence/task-9-high-crowding.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add Factor Crowding detection (MVP, experimental)`
  - Files: `lft/evaluation/advanced.py`, `tests/test_factor_crowding.py`
  - Pre-commit: `pixi run pytest tests/test_factor_crowding.py -v`

- [x] 10. 高级评估：Market Regime IC（牛熊市 IC 分离）

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_market_regime_ic()` 函数
  - 使用等权市场收益（所有股票的 `ret` 均值）作为市场状态指标：`ret > 0` = Up，`ret <= 0` = Down
  - 分别计算 Up/Down 状态下的 Rank IC Mean/Std/IR/Positive Ratio
  - 输出 `MarketRegimeICResult` dataclass：`up: ICAnalysisResult` + `down: ICAnalysisResult` + `up_periods` + `down_periods`
  - 实现测试：`tests/test_market_regime_ic.py` 从 RED → GREEN

  **Must NOT do**:
  - 不要使用复杂的状态定义（如移动平均线交叉）— 仅用 `ret > 0`

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 涉及条件统计 + 分组聚合

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 4
  - **Blocks**: Task 17

  **QA Scenarios**:
  ```
  Scenario: 牛熊市 IC 分离输出两个子结果
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_market_regime_ic.py::test_up_down_separation -v
    Expected Result: PASS — 输出包含 up 和 down 两个 ICAnalysisResult
    Evidence: .sisyphus/evidence/task-10-market-regime.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add Market Regime IC separation (Up/Down market)`
  - Files: `lft/evaluation/advanced.py`, `tests/test_market_regime_ic.py`
  - Pre-commit: `pixi run pytest tests/test_market_regime_ic.py -v`

- [x] 11. 高级评估：Rank Autocorrelation（因子排序自相关）

  **What to do**:
  - 在 `lft/evaluation/advanced.py` 中创建 `compute_rank_autocorrelation()` 函数
  - 计算相邻两期因子 rank 的 Spearman 相关系数（测度因子信号的时序稳定性）
  - 输出 `RankAutocorrResult` dataclass：`mean_autocorr`（平均自相关）+ `autocorr_series`（每期）+ `half_life_est`（估计半衰期 = `-ln(2)/ln(autocorr)`）
  - 实现测试：`tests/test_rank_autocorr.py` 从 RED → GREEN

  **Must NOT do**:
  - 不要在 `turnover.py` 中添加代码

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - **Reason**: 时序统计 + 信号衰减分析

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 4
  - **Blocks**: Task 17

  **References**:
  - Alphalens `factor_rank_autocorrelation()` 概念

  **QA Scenarios**:
  ```
  Scenario: 高持续性因子自相关 > 0.8
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_rank_autocorr.py::test_high_persistence -v
    Expected Result: PASS — mean_autocorr > 0.8, half_life > 3
    Evidence: .sisyphus/evidence/task-11-high-autocorr.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add Rank Autocorrelation analysis`
  - Files: `lft/evaluation/advanced.py`, `tests/test_rank_autocorr.py`
  - Pre-commit: `pixi run pytest tests/test_rank_autocorr.py -v`

---

### Wave 3: MFT 分钟频框架 + HFT 桩（依赖 Wave 0 数据验证）

- [x] 12. MFT：Factor 基类 + DataContext
- [x] 13. MFT：预处理管线桩
- [x] 15. MFT：评估桩 + 因子注册 + HFT 桩

  **What to do**:
  - 创建 `mft/evaluation/__init__.py` — 定义空 dataclass（MFTICResult, MFTBacktestResult），仅含 docstring
  - 创建 `mft/factors/registry.py` — 基于 `lft/factors/registry.py` 模式的因子自动发现（搜索 `mft.factors.demo`）
  - 更新 `hft/__init__.py` — 创建 `HFTFactor(ABC)` 类，所有方法抛 `NotImplementedError("HFT framework not yet implemented")`
  - 创建 `tests/test_mft_evaluation_stubs.py` + `tests/test_hft_stub.py`

  **Must NOT do**:
  - 不要实现任何 MFT 评估逻辑
  - 不要为 HFT 桩写任何实现

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 占位桩 + 简单注册，复制已有模式

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 12 的因子基类）
  - **Parallel Group**: Wave 3
  - **Blocked By**: Task 12

  **QA Scenarios**:
  ```
  Scenario: HFT 桩抛出 NotImplementedError
    Tool: Bash
    Steps:
      1. pixi run python -c "from hft import HFTFactor; h=HFTFactor(); h.compute(None)" 2>&1
    Expected Result: 输出包含 NotImplementedError
    Evidence: .sisyphus/evidence/task-15-hft-stub.txt
  ```

  **Commit**: YES
  - Message: `feat(mft): add MFT evaluation stubs, registry, and HFT stub`
  - Files: `mft/evaluation/__init__.py`, `mft/factors/registry.py`, `hft/__init__.py`, `tests/test_mft_evaluation_stubs.py`, `tests/test_hft_stub.py`
  - Pre-commit: `pixi run pytest tests/test_mft_evaluation_stubs.py tests/test_hft_stub.py -v`

---

### Wave 4: 详细因子报告（Phase 6，依赖 Wave 1 + Wave 2）

- [x] 16. Phase 6：Tear Sheet 报告引擎

  **What to do**:
  - 创建 `reporting/tear_sheet.py` — 报告生成核心
    - 函数：`generate_tear_sheet(factor_name, ic_result, bt_result, to_result, advanced_results) -> str`（返回 HTML 字符串）
    - 6 个面板（Panels）：
      1. **Overview** — 因子名称/频率/日期范围/Coverage/基础统计
      2. **Returns Analysis** — 分组收益柱状图 + 多空净值曲线（matplotlib → base64 PNG）
      3. **IC Analysis** — IC 时序图 + IC Decay 柱状图 + IC 分布直方图（plotly → PNG）
      4. **Turnover Analysis** — 换手率时序 + 迁移矩阵热力图
      5. **Risk Attribution** — 行业暴露 + 市值暴露（如高级评估数据可用）
      6. **Summary** — 关键指标一览表
  - 使用 Jinja2 模板（`reporting/templates/tear_sheet.html`）
  - 所有图表用 Matplotlib 渲染为 base64 PNG 内嵌于 HTML
  - HTML 文件必须 < 5MB
  - 创建 `tests/test_reporting.py`

  **Must NOT do**:
  - 不要使用交互式 JavaScript（Plotly 仅用于静态 PNG 渲染）
  - 不要引入新的 Python 依赖

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]
  - **Reason**: 涉及可视化设计 + HTML 模板 + 图表渲染

  **Parallelization**:
  - **Can Run In Parallel**: NO（报告引擎核心，阻塞 Task 17, 18）
  - **Parallel Group**: Wave 4（第一个任务）
  - **Blocked By**: Task 1-3（Bug 修复）, Task 5-11（高级评估）
  - **Blocks**: Task 17, 18

  **References**:
  - `lft/evaluation/ic_analysis.py:39-61` — `ICAnalysisResult` dataclass 结构
  - `lft/evaluation/backtest.py:8-26` — `BacktestResult` dataclass 结构
  - `lft/evaluation/turnover.py:7-20` — `TurnoverResult` dataclass 结构
  - Alphalens Tear Sheet 概念参考

  **QA Scenarios**:
  ```
  Scenario: 报告生成非空 HTML
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_reporting.py::test_generate_html_not_empty -v
    Expected Result: PASS — 返回的 HTML 字符串非空 + 包含 <html> 标签
    Evidence: .sisyphus/evidence/task-16-html-output.txt

  Scenario: HTML < 5MB
    Tool: Bash
    Steps:
      1. pixi run pytest tests/test_reporting.py::test_html_size_limit -v
    Expected Result: PASS — HTML 大小 < 5MB
    Evidence: .sisyphus/evidence/task-16-size-check.txt
  ```

  **Commit**: YES
  - Message: `feat(report): add 6-panel Tear Sheet report engine`
  - Files: `reporting/tear_sheet.py`, `reporting/templates/tear_sheet.html`, `tests/test_reporting.py`
  - Pre-commit: `pixi run pytest tests/test_reporting.py -v`

- [x] 17. Phase 6：报告 CLI 脚本

  **What to do**:
  - 创建 `scripts/generate_report.py` — 命令行入口
    - 参数：`--factor`（因子名）, `--start`, `--end`, `--frequency`（daily/weekly/monthly）, `--universe`（默认 lft_default）, `--output`（输出路径，默认自动生成）
    - 编排流程：加载数据 → 因子计算 → 预处理 → 基础评估（IC/回测/换手率）→ 高级评估（7 个指标）→ 报告生成
    - 输出到 `output/reports/{factor_name}_{start}_{end}.html`
  - 添加 `pixi.toml` tasks：`report = "python scripts/generate_report.py"`

  **Must NOT do**:
  - 不要修改 `run_lft_single.py` 或 `run_lft_compare.py`
  - 不要在 CLI 中添加交互式功能

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []
  - **Reason**: CLI 编排 + 脚本编写

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 16 报告引擎）
  - **Parallel Group**: Wave 4
  - **Blocked By**: Task 16

  **QA Scenarios**:
  ```
  Scenario: 端到端报告生成
    Tool: Bash
    Steps:
      1. pixi run python scripts/generate_report.py --factor momentum_20d --start 20250101 --end 20250513
      2. Test-Path -LiteralPath "output/reports/momentum_20d_*.html"
    Expected Result: 命令成功退出（exit 0），HTML 文件存在
    Evidence: .sisyphus/evidence/task-17-report-cli.txt
  ```

  **Commit**: YES
  - Message: `feat(report): add report generation CLI script`
  - Files: `scripts/generate_report.py`, `pixi.toml`
  - Pre-commit: `pixi run python scripts/generate_report.py --factor momentum_20d --start 20250101 --end 20250513`

- [x] 18. 最终 QA：全测试套件 + 覆盖率

  **What to do**:
  - 运行 `pixi run pytest tests/ -v` → 全部通过
  - 检查新代码覆盖率 ≥ 80%（`pytest --cov`）
  - 修复所有测试失败和覆盖率不足的问题
  - 验证所有 QA scenarios 的 evidence 文件都存在
  - 运行 `pixi run lint` 确保无新增 lint 问题

  **Must NOT do**:
  - 不要跳过失败的测试
  - 不要在未修复问题的情况下标记为"已完成"

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: []
  - **Reason**: 测试执行 + 问题修复

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖所有前置任务）
  - **Blocked By**: Task 0-17

  **QA Scenarios**:
  ```
  Scenario: 全测试套件通过
    Tool: Bash
    Steps:
      1. pixi run pytest tests/ -v --tb=short
      2. 确认 exit code = 0
    Expected Result: ALL TESTS PASS
    Evidence: .sisyphus/evidence/task-18-full-test.txt
  ```

  **Commit**: YES（如有修复）
  - Message: `chore(qa): final test suite verification and coverage`
  - Pre-commit: `pixi run pytest tests/ -v`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, pytest output, pixi run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pixi run lint` (ruff check .). Review all changed files for: AI slop (excessive comments, over-abstraction, generic names), empty catches, unused imports, type annotation gaps. Check that existing module signatures are unchanged.
  Output: `Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Run full test suite: `pixi run pytest tests/ -v`. Run report generation: `pixi run python scripts/generate_report.py --factor momentum_20d --start 20250101 --end 20250513`. Verify HTML output. Run MFT demo: verify factor computes.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Tests [N/N pass] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 0**: `feat(data): verify Tushare data availability for MFT/advanced eval` — data_check.log
- **Wave 1**: `fix: normalizer std=0, pit_align perf, CSI index members` — normalizer.py, pit.py, universe.py
- **Wave 2**: `feat(eval): 7 advanced single-factor evaluation metrics` — lft/evaluation/advanced.py, tests/
- **Wave 3**: `feat(mft): MFT framework skeleton + demo + HFT stub` — mft/, hft/, tests/
- **Wave 4**: `feat(report): Tear Sheet report engine + CLI` — reporting/, scripts/generate_report.py, tests/
- **Wave FINAL**: `chore(qa): verification and quality assurance` — evidence/

---

## Success Criteria

### Verification Commands
```bash
pixi run pytest tests/ -v                    # Expected: ALL PASS, ≥80% new code coverage
pixi run python scripts/generate_report.py --factor momentum_20d --start 20250101 --end 20250513  # Expected: HTML < 5MB
pixi run python -c "from mft.data.context import MFTDataContext; print('OK')"  # Expected: OK
pixi run python -c "from hft import HFTFactor; HFTFactor().compute(None)" 2>&1  # Expected: NotImplementedError
pixi run lint                                   # Expected: clean or pre-existing only
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All tests pass
- [x] TDD cycle complete for all new modules
- [x] Report generation produces valid HTML
- [x] MFT demo factor computes successfully
- [x] HFT stub throws NotImplementedError
- [x] No regression in existing LFT pipeline
