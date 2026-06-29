# M1 · 因子挖掘引擎 MVP — 设计文档

> 状态：设计已评审通过（2026-06-29），待转实现计划。
> 上游：[FactorZen 升级计划](../FactorZen-升级计划.md) 的里程碑 **M1**。
> 定位：把 FactorZen 从「严谨的单因子研究框架」推进到「能自动产出可解释因子」的研究平台，是后续 M2 防过拟合护栏与 M5 Agent 闭环的地基。

---

## 1. 目标与定位

自动生成**可解释的量价 + 基本面因子表达式**，用随机搜索与遗传编程（GP）搜索高质量、低冗余的候选；每个 top-K 候选都能**反序列化成标准 `DailyFactor`**，被现有 `fz factor run` 评估管线**无缝复现**，并附带 **train/valid 样本外对比**与**与因子池去相关**报告。一条 `fz mine search` 命令端到端跑通，全程落 `manifest.json` 保证可复现。

**一句话**：机器自动挖因子 → 现有评估管线验真 → 可解释表达式 + 样本外证据 + 去相关，全程可审计可复现。

### 1.1 已拍板的关键决策（评审结论）

| 决策 | 选择 | 理由 |
|---|---|---|
| MVP 范围 | 务实端到端闭环 | GP 与去相关都在；OOS holdout 永久隔离、PBO/DSR 留 M2 |
| 数据范围 | 价量 + 基本面（`daily_basic`） | 基本面限定为每日估值/市值/换手（PIT 干净）；**不碰 `finance` 财报**（披露滞后陷阱） |
| 样本外 | 带轻量 train/valid 时间切分 | 立刻暴露过拟合；完整 OOS 永久隔离留 M2 |
| 表达式表示 | 自定义 AST ↔ 字符串双向 | GP 友好 + 可解释 + 可序列化复现 |
| 评估架构 | 两段式 | 搜索内循环快速 Rank IC/IR；top-K 才跑完整 `fz factor run` |
| 搜索算法 | random → 遗传编程（GP） | random 先打通端到端，GP 作为第一个增量接入 |

---

## 2. 非目标（明确不做，防 scope 蔓延）

- **完整 OOS holdout 永久隔离** + **PBO / Deflated Sharpe / White Reality Check / Hansen SPA** → M2。
- **`finance` 财报基本面因子**（披露滞后，PIT 复杂）→ 后续。
- **Agent / LLM 闭环挖掘** → M5/M6。
- **Leaderboard HTML 美化展示页** → M7 成果页；MVP 先输出 CSV / markdown 排行榜。
- **全 A 股（~5000 标的）× 10 年性能优化** → M0 性能线；MVP 先在 `csi500` 子时段验证。

---

## 3. 模块结构

```text
src/factorzen/discovery/
├── __init__.py
├── operators.py        算子库：每个算子 = 带类型签名的 AST 节点工厂（时序/截面/算术）
├── expression.py       AST 节点定义 + 编译器（AST → polars 求值）+ str()/parse() 双向序列化
├── factor.py           ExpressionFactor：把表达式包装成标准 DailyFactor（compile → compute）
├── scoring.py          快速 fitness（Rank IC/IR）+ 去相关惩罚 + 复杂度惩罚 + train/valid 评估
├── search/
│   ├── __init__.py
│   ├── base.py         Searcher 抽象基类 + Candidate / SearchResult 数据类
│   ├── random_search.py   类型约束的随机表达式生成 + 随机搜索
│   └── genetic.py      遗传编程：子树交叉/变异/锦标赛选择/精英保留/防膨胀
├── export.py           top-K → workspace/factors/daily/*.py（可读、可 diff、可复现）
└── mining_session.py   编排：生成 → 评估 → 去相关 → 排序 → 落候选 → manifest

src/factorzen/cli/main.py   新增 fz mine search / fz mine leaderboard / fz mine export

workspace/mining_sessions/{session_id}/
├── manifest.json       配置 / seed / 算子集 / 搜索预算 / 尝试总数 N / git SHA / 耗时
├── candidates.csv      全部候选：表达式 · train IC/IR · valid IC/IR · 衰减 · max_corr · 复杂度
└── exported/           top-K 导出的 *.py（也可由 fz mine export 落到 workspace/factors/daily/）

tests/
├── test_discovery_operators.py    每个算子 polars 编译正确性（对拍手写）
├── test_discovery_expression.py   round-trip 序列化 + PIT/前视安全
├── test_discovery_scoring.py      fitness 一致性 + 去相关 + 复杂度惩罚
├── test_discovery_search.py       random 采样 + GP 交叉/变异合法性
└── test_discovery_session.py      端到端 smoke（同 seed 同结果，产物齐全）
```

---

## 4. 引擎心脏：算子库 + 表达式 AST

### 4.1 叶子（数据接口）

- **价量**（来自 `ctx.daily`，列已含复权）：`close_adj` `open_adj` `high_adj` `low_adj` `vol` `vwap` `returns(n)` `log_vol`
- **基本面**（来自 `ctx.daily_basic`，按 `trade_date` 对齐）：`total_mv` `circ_mv` `pb` `pe_ttm` `turnover_rate` `volume_ratio`
- **常数 / 窗口参数**：整数窗口（如 5/10/20/60）与浮点常数

> `vwap` 若行情无现成列，由 `amount / vol` 派生；`returns(n)` 由 `close_adj / close_adj.shift(n) - 1` 派生。派生列在算子层定义，叶子节点统一暴露。

### 4.2 算子（每个节点知道 arity / 返回类型 / 时序还是截面 / 如何 compile）

- **时序算子**（编译时 `.over("ts_code")`）：`ts_mean` `ts_std` `ts_rank` `ts_sum` `ts_min` `ts_max` `ts_corr` `ts_decay_linear` `delta` `delay`，均带窗口参数
- **截面算子**（编译时 `.over("trade_date")`）：`rank` `zscore` `scale` `neutralize`（`neutralize` 复用现有 `neutralize_ols`）
- **算术算子**：`+ - * /` `abs` `log` `sign` `sqrt` `power` `min` `max`

### 4.3 表达式 AST

- 内部为带类型的 AST 树；GP 直接对树做子树交叉/变异。
- `str(ast)` 输出可读表达式，`parse(s)` 还原为等价 AST（**round-trip 等价，有测试保证**）。
- **类型约束生成**：随机生成与 GP 变异时按算子类型签名约束（如 `ts_corr` 需两个时序子树、窗口为正整数、`log` 输入加正值保护），从源头减少无效表达式，而非事后丢弃。

示例（可解释）：

```text
rank(ts_corr(returns(1), log(vol), 20))
zscore(ts_mean(close_adj, 5) / ts_mean(close_adj, 60))        # 均线比值
neutralize(ts_rank(turnover_rate, 20) * sign(delta(pb, 5)))   # 价量 × 基本面
```

### 4.4 编译与求值

AST → 一串 polars `with_columns` 链 → 在预加载的 `daily`（已 join `daily_basic`）上求值 → 输出 `[trade_date, ts_code, factor_value]`，与手写因子 `compute()` 的返回完全同构。

---

## 5. 数据流（端到端，含两段式评估）

```text
1. 预加载一次 ctx.daily + ctx.daily_basic（train+valid 全区间 + lookback 预热）→ 内存缓存
   ★ 性能关键：所有候选复用同一份数据，避免逐候选重新 IO
2. 预计算一次 forward returns（复用 compute_fwd_returns）→ 缓存
3. 搜索内循环（random / GP），只在 train 段：
     生成/演化 AST → 编译求值 factor_value → 快速预处理（cross_sectional_zscore / rank）
     → 快速 fitness = Rank IC / IR（复用 compute_rank_ic 核心，不跑回测）
     → 去相关惩罚：与内置因子池 + 已入选候选的 max_corr（复用 compute_factor_correlation）
     → fitness_adj = IR_train − λ·max_corr − γ·complexity
     → GP：锦标赛选择 / 子树交叉 / 变异 / 精英；random：按预算采样
4. 收敛取 top-K：
     valid 段重算 IC / IR（样本外）· 完整去相关 · 记录尝试总数 N（多重检验记账种子）
5. 落地：
     每个 top-K → export 成 workspace/factors/daily/*.py（registry 自动发现 → fz factor run 复现）
     candidates.csv + manifest.json
6. fz mine leaderboard 展示；fz factor run <挖出因子> 跑完整 tear sheet
```

---

## 6. 接口契约（对接现有系统，复用而非重造）

> 以下接口均已由代码探索验证，直接复用。

| 用途 | 复用接口 | 位置 |
|---|---|---|
| 因子基类 | `DailyFactor.compute(ctx) -> pl.DataFrame[trade_date, ts_code, factor_value]` | `daily/factors/base.py:18` |
| 数据上下文 | `FactorDataContext`，`ctx.daily` / `ctx.daily_basic`（LazyFrame） | `daily/data/context.py:12,35,65` |
| 注册发现 | 模块扫描 + 类属性 `name` + 模块级实例化；`get_factor` / `list_factors` | `daily/factors/registry.py` |
| Rank IC | `compute_rank_ic(clean_df, ret_df)` / `compute_ic(df, factor_col, ret_col, method)` | `daily/evaluation/ic_analysis.py:~310,244` |
| 前向收益 | `compute_fwd_returns(daily)` → `fwd_ret_1d/5d/10d/20d` | `daily/evaluation/ic_analysis.py:24` |
| 截面预处理 | `cross_sectional_zscore` / `cross_sectional_rank` / `mad_clip` / `winsorize_percentile` | `daily/preprocessing/{normalizer,outlier}.py` |
| 中性化 | `neutralize_ols(df, col, stock_basic, daily_basic)` | `daily/preprocessing/neutralizer.py:14` |
| 去相关 | `compute_factor_correlation(factor_dict, factor_col)` | `daily/evaluation/correlation.py:26` |
| 多重检验 | `apply_fdr_correction(p_values, method)` | `daily/evaluation/advanced/correlation.py:92` |
| 停牌掩码 | `get_universe_snapshot(date, universe)` → `is_suspended` 等（M0 新增） | `core/universe.py` |

**`ExpressionFactor`（新增）** 实现 `DailyFactor` 接口：持有表达式字符串/AST，`compute(ctx)` 内编译执行，类属性 `name` 由表达式 hash 或导出时指定。这样挖出的因子与手写因子在注册、评估、报告全链路完全等价。

---

## 7. PIT / 前视安全（不可妥协）

- 时序算子只用历史（`shift` / `rolling` + `min_periods`）；编译强制按 `ts_code` 排序后 `.over("ts_code")`。
- `daily_basic` 按当日 `trade_date` 对齐（每日估值，PIT 安全）；**不碰 `finance` 财报**。
- **复用 M0 `get_universe_snapshot`**：停牌日（`is_suspended`，`vol==0`）的量价在算子层置 `null`，避免污染 rolling —— M0 → M1 的直接衔接。
- **train/valid 按时间先后切分**；valid 段在搜索内循环**完全不可见**，仅 top-K 验收用一次。
- 专门的**前视安全测试**：构造已知泄漏未来信息的表达式，断言被拦截或产出预期 IC 符号。

---

## 8. 防过拟合（MVP 最小集，为 M2 留 hook）

- **多重检验记账**：记录搜索尝试总数 N，排行榜显式标注「从 N 个候选中选出 top-K」。
- **去相关**：候选必须报告与因子池 `max_corr`，超阈值淘汰/降权。
- **train/valid 样本外对比**：IC 衰减一眼可见。
- **复杂度惩罚**：fitness 减去表达式复杂度项，抑制 GP 表达式膨胀（bloat）。
- `scoring.py` 预留接口，供 M2 接入 PBO / DSR / Reality Check。

---

## 9. CLI（保持 `fz` 一致风格，全部写 manifest）

```bash
fz mine search --config <yaml>            # 或 --set key=value 覆盖；默认内置研究级配置
                --method random|genetic    # 搜索算法
                --budget N                 # 评估预算（候选数 / GP 代数）
                --top-k K --seed 42
fz mine leaderboard <session_id>          # 展示候选排行榜（CSV/markdown）
fz mine export <session_id> --expr <id>   # 把指定候选导出成 workspace/factors/daily/*.py
```

---

## 10. 测试策略

- **一致性测试（关键）**：用挖掘引擎对已知因子（如 `momentum`）算出的 IC == `fz factor run` 的 IC —— 证明引擎与主线评估**同口径**。
- **round-trip 序列化**：`parse(str(ast))` 与原 AST 等价。
- **算子编译对拍**：每个算子的 polars 实现对照手写结果。
- **前视安全测试**：见 §7。
- **GP 合法性**：交叉/变异产物始终是类型合法的表达式。
- **端到端 smoke**：小预算挖掘跑通，产物齐全，同 `seed` 同结果。
- 全部离线 mock 数据，符合 CI 离线可重复原则。

---

## 11. 验收标准（Definition of Done）

- [ ] `fz mine search` 一条命令端到端跑通，产出 top-K + `candidates.csv` + `manifest.json`。
- [ ] 挖出的因子能被 `fz factor run` 复现，且其 IC 与挖掘内循环一致（一致性测试通过）。
- [ ] 表达式可读、可序列化、round-trip 等价。
- [ ] 候选报告含 train/valid IC 对比、`max_corr` 去相关、尝试总数 N 记账。
- [ ] 同 `seed` 可复现（manifest 记录全过程）。
- [ ] 复用 M0 停牌掩码与现有 IC / 预处理 / 去相关 / 注册，无重复造轮子。
- [ ] `lint` / `typecheck` / `test` / `coverage` 全绿，`git status` 干净。

---

## 12. 建议实现顺序（为 writing-plans 铺垫）

> 每步独立可交付、有测试、产物可复现。先打通端到端，再加 GP，最后接 CLI 与导出。

1. **表达式 AST + 算子库 + 编译器**（`expression.py` + `operators.py`）：含 round-trip 与算子对拍测试。
2. **ExpressionFactor**（`factor.py`）：表达式 → 标准 `DailyFactor`，能被 `fz factor run` 跑通（含一致性测试）。
3. **scoring**（`scoring.py`）：快速 fitness + train/valid 切分 + 去相关惩罚。
4. **random 搜索 + mining_session**：端到端打通，产出 candidates.csv + manifest（端到端 smoke）。
5. **遗传编程**（`search/genetic.py`）：交叉/变异/选择/防膨胀，作为 random 之上的增量。
6. **CLI + export**：`fz mine search/leaderboard/export`，top-K 落 workspace 因子。

---

*本设计基于 FactorZen 当前 `feature/platform-upgrade` 分支现状制定，对接的现有接口均已代码验证。*
