# 产物布局

> [FactorZen](../../README.md) · [文档](../README.md) · **产物布局**

严格区分两个根目录：

| 根 | 内容 | 规矩 |
|---|---|---|
| `data/` | 所有**数据**：行情原始落盘、数据湖、缓存、派生特征 | 新数据源一律落这里 |
| `workspace/` | 所有**研究产出**：评估 run、挖掘 session、因子库、组合、模拟、报告 | 只放产出，不放数据 |

路径常量的唯一真源是 `src/factorzen/config/settings.py`。本页描述的是**磁盘上真实存在且有产出**的目录。正文默认 A 股研究链路的产物形态；多市场落盘差异见 [多市场](../concepts/multi-market.md) 与 [数据源与口径](data-sources.md)。

> ℹ️ `settings.py` 还声明了 `MINE_AGENT_DIR`、`EXECUTION_DIR`、`OPS_DIR`、`DATA_PROCESSED` 等常量，但这些目录要么尚未跑过对应命令，要么在 `src/` 中没有消费方（`EXECUTION_DIR` / `OPS_DIR` grep 不到引用点）。`PORTFOLIOS_DIR` 被 `fz portfolio build --out-dir` 用作默认值，但目录内部形态未在本页展开。本页不描述它们的内部形态——常量存在 ≠ 产物结构已定型。

---

## 1. `workspace/` 一级目录 ↔ 生产命令

| 目录 | 由谁产生 | 内容 |
|---|---|---|
| `factor_evaluations/` | `fz factor eval` / `fz factor backtest` | 单因子评估/回测 run，每 run 一个子目录 + 跨 run 索引 |
| `mining_sessions/` | `fz mine search` | 随机/遗传搜索 session |
| `mine_team/` | `fz mine team` | 多角色团队挖掘 session（含 lift 准入记录） |
| `combinations/` | `fz combine run / from-session / from-library` | 多因子合成实验 |
| `combine_backtests/` | `fz combine backtest` | 组合回测产物 |
| `risk_models/` | `fz risk build` | 风险模型（暴露/协方差/特质风险） |
| `sim/` | `fz sim run` | 模拟交易 run |
| `reports/` | 全部 HTML 报告集中收口 | `daily/` 因子报告 · `intraday/` · portfolio dashboard（前端「报告」栏单点可见） |
| `factor_library/` | lift 准入、`fz factor-library rebuild` | **因子库登记簿本体，不是 run 目录** |
| `factor_store/` | `fz factor new`、`fz factor-library store sync` | python/expression 因子三件套（meta/py/parquet） |
| `configs/` | 用户手写 | YAML 运行配置模板，非产物（见 [配置参考](configuration.md)） |
| `runs/` | 各命令共同写入 | 全局归档 `runs/artifacts/` + 日志 `runs/logs/` |
| `_ops/` | 长任务与运维脚本（`WORKSPACE_OPS_DIR`） | 运维杂项统一屋，见下表 |

> ℹ️ `workspace/` 根下还会出现散落的 `*.done` / `*.exitcode` / `*.log` 文件。这是长任务的 sentinel 惯例（后台任务完成时 `touch` 一个哨兵文件供轮询），不属于产物结构。

### `_ops/` 子目录（运维，非产品 stage）

常量：`settings.WORKSPACE_OPS_DIR = workspace/_ops`。

| 子目录 | 写入者 | 内容 |
|---|---|---|
| `logs/` | 长任务 shell（`workspace/configs/run_*.sh` 等）、一次性作业 | 文本日志 / sentinel |
| `data_ingest/` | `tools/ingest_minute.py` | 分钟数据导入 manifest + done |
| `data_backfill/` | `data/_tools/card_*.py` 等回填脚本 | 回填 run 目录 |
| `data_maintenance/` | `tools/repair_raw_partition.py` | raw 分区修补 manifest |
| `campaigns/` | 研究战役归档 | 战役材料 |
| `architecture_review/` | 架构评审作业 | 评审产物（含 `job_manifest.txt`） |

---

## 2. 单个 run 目录的真实文件清单

### 2.1 `factor_evaluations/{run_id}/` —— 单因子评估

`run_id` 缺省格式 `{safe(factor)}_{YYYYmmdd_HHMMSS}`，因子名里的非法字符（`[^A-Za-z0-9_.-]+`）被替换成 `_`。

以 `factor_evaluations/momentum_20d_20260717_045904/` 为例：

| 文件 | 体量 | 内容 |
|---|---|---|
| `factor.parquet` | 116 MB | 因子面板（全时段全标的），最大的文件 |
| `universe.parquet` | 7.1 MB | 逐日 universe 快照，PIT 自查用 |
| `report.html` | ~450 KB | 单页报告（eval→信号轨 / backtest→交易轨；各自 8 张图，base64 PNG 内嵌，体积随图数增长） |
| `manifest.json` | 3.6 KB | 复现清单，见 §3.1 |
| `ic.parquet` | 6.2 KB | 逐日 IC 序列 |
| `quality.json` | 1.0 KB | 因子质量指标 |
| `meta.json` | 651 B | 因子元信息 |
| `walk_forward.json` | 42 B | 滚动验证结果；**`walk_forward.enabled=false` 时近乎空文件**（仅 `fz factor backtest`） |
| `signal.json` | ~1 KB | 信号层分层/多空毛口径统计 + 口径元信息（仅 `fz factor eval`） |
| `signal_group_nav.parquet` | ~10 KB | 各分位组累计净值曲线（仅 `fz factor eval`） |

两条轨道产出的文件集不同：`fz factor eval` 出 `signal.*` 不出 `walk_forward.json`，`fz factor backtest` 反之；`factor/universe/ic/quality/meta/report/manifest` 两轨共有。

另有跨 run 索引 `factor_evaluations/experiment_index.jsonl`，每 run 追加一行：`run_id, timestamp, factor, universe, start, end, status, manifest_path`。

> ⚠️ 索引写入失败被静默吞掉（`except: pass`），刻意不让索引故障影响实验本身。**索引缺行不代表 run 失败**，以 run 目录里的 `manifest.json` 为准。

### 2.2 `sim/{run_id}/` —— 模拟交易

```text
manifest.json   metrics.json   nav.parquet
```

三个文件，无更多。`nav.parquet` 是净值序列，`metrics.json` 是汇总指标，`manifest.json` 的 schema 与评估域**完全不同**，见 §3.2。

### 2.3 `mine_team/{run_id}/` —— 团队挖掘 session

以 `mine_team/20260716_031732_team_42_8r/` 为例：

| 文件 | 体量 | 内容 |
|---|---|---|
| `manifest.json` | 441 KB | session 全记录（attempts / candidates / lift 结果），见 §3.3 |
| `lift_test_manifest.json` | 83 KB | lift 准入复审记录（最新一次；每次运行覆写） |
| `lift_test_manifest_{YYYYmmddTHHMMSS}.json` | 83 KB | 同上的**不可变归档**，每次运行新增一份、永不覆写 |
| `candidates.csv` | 38 B | 存活候选摘要（该 run 几乎为空） |

`fz mine team` 还会在 `mine_team/_pool_cache/{key}/` 下建池缓存目录。

### 2.4 `risk_models/{run_id}/`

```text
exposures.parquet        factor_covariance.parquet   factor_returns.parquet
specific_risk.parquet    risk_summary.csv            manifest.json
```

### 2.5 `combinations/{run_id}/`

跑完的合成实验目录：

```text
combined_equal_weight.parquet   combined_ic_weighted.parquet
combined_lgbm.parquet           combined_max_ir.parquet
comparison.csv                  importance.csv
manifest.json                   input_manifest.json      report.md
```

四种合成方式各落一份因子面板，`comparison.csv` 横向对比，`importance.csv` 是 lgbm 的 SHAP/gain 重要度。

> ⚠️ 这个域的目录**形态不齐**：只做了输入准备、尚未跑合成的目录里只有 `input_manifest.json` + `run.sh`（外加一个同名 `.done` sentinel），完全没有 `combined_*.parquet`。判断一次合成是否真跑完，看 `manifest.json` 是否存在，别看目录是否存在。

### 2.6 `mining_sessions/{session_id}/`

```text
candidates.csv   manifest.json   exported/
```

### 2.7 `factor_library/` —— 登记簿本体（非 run 目录）

这是全平台唯一的因子登记簿，按市场分文件：

| 文件 | 内容 |
|---|---|
| `{market}.jsonl` | 登记簿本体，每因子一行；市场取 `ashare` / `crypto` / `futures` / `us` |
| `{market}.md` | 同内容的人类可读渲染（`rebuild` 附带产出） |
| `summary.md` | 跨市场汇总 |
| `rebuild_{market}_manifest.json` | 每次 `fz factor-library rebuild` 的记录：窗口/源/git_sha/时间 + lift 复审计数 |
| `forward_track/{market}.jsonl` | 向前追踪记录 |
| `*.jsonl.bak-*` / `*.bak_legacy_*` | 改动前的历史备份 |

> ⚠️ `lift-test` 与 `forward-review` **默认 dry-run**，只打印裁决不写库。要真正写入登记簿必须显式加 `--apply`。

### 2.8 `runs/` —— 全局归档与日志

```text
runs/logs/                        factor_research.log(+ 按日轮转)
runs/artifacts/daily/factors/     {factor}_{start}_{end}.parquet
runs/artifacts/daily/results/     {factor}_{start}_{end}_{ic,quality,universe,meta,signal,...}.*
runs/artifacts/daily/charts/
runs/artifacts/intraday/{factors,results}/
# HTML 报告不再落 runs/artifacts/，已收口到 workspace/reports/（见 §2.9）
```

> ℹ️ **每次 `fz factor eval` / `fz factor backtest` 产物落两份**：  
> - **run 目录**（`factor_evaluations/<run_id>/`）用**短名**：`ic.parquet`、`signal.json`、`report.html`、`meta.json` 等（见 §2.1）。  
> - **全局归档**：数据类（`_ic.parquet`、`_signal.json` 等长名）落 `runs/artifacts/daily/`；**HTML 报告已收口到 `workspace/reports/daily/`**（`{factor}_{start}_{end}_eval.html` / `.html`），前端「报告」栏单点可见。  
> manifest 的 `outputs` 字段同时记录两组路径。

qlib 系列因子会再分桶：因子名前缀 `qlib_alpha158_` → `qlib158/` 子目录，`qlib_alpha360_` → `qlib360/`（`settings.py` 的 `daily_output_bucket`）。

### 2.9 `workspace/reports/` —— HTML 报告统一收口

所有 HTML 报告集中于此，前端「报告」栏单点可见、可直接在浏览器打开：

```text
workspace/reports/daily/       fz factor eval / backtest 的因子报告（{factor}_{start}_{end}[_eval].html）
workspace/reports/intraday/    日内因子报告
workspace/reports/portfolio_<run_id>.html   fz report portfolio / research run 的组合 dashboard
```

> ℹ️ run 目录内仍各自保留一份短名 `report.html`（run 自包含产物，见 §2.1）；`workspace/reports/` 是跨 run 的集中归档。

---

## 3. `manifest.json` 的真实字段

> ⚠️ **`manifest.json` 不是一套统一 schema。** 不同域由不同写入方产出，字段互不重叠。跨域消费 manifest 前先确认它来自哪个域。

### 3.1 评估域（`core/experiment.py`）

写入 `factor_evaluations/{run_id}/manifest.json`。`run_experiment()` 自管的标准字段全集：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | str | 常量 `"1"` |
| `run_id` | str | 目录名 |
| `git_sha` | str | `git rev-parse HEAD`；超时 5s 或失败 → `"unknown"` |
| `git_dirty` | bool | `git status --porcelain` 非空即 true |
| `pixi_lock_sha256` | str | `pixi.lock` 的 sha256；文件不存在 → `"missing"` |
| `command` | list \| null | 命令行数组 |
| `config` | dict | `RunConfig.model_dump()` 全量 |
| `outputs` | dict | 产物路径表 |
| `start_ts` / `end_ts` | str | 起止时间 |
| `duration_seconds` | float | 保留 3 位 |
| `status` | str | `running` → `success` / `failure` |
| `error` | str \| null | 异常信息 |

> ⚠️ `git_dirty: true` 时会打 warning「无法仅凭 git SHA 复现」。**带脏工作区跑出来的结果，事后无法用 git_sha 精确重跑**——正式实验前先把改动提交掉。

`outputs` 是**双份路径**（全局归档键 + `run_` 前缀本地副本，见 `experiments/run_paths.py` 的 `copy_outputs_to_run_dir()`），键数按轨道不同：

**`fz factor backtest`（交易轨）= 7 全局 + 7 个 `run_*` = 14 键**

- 全局归档 7 键：`factor` / `ic` / `quality_report` / `walk_forward_summary` / `universe_snapshot` / `report` / `meta` → 指向 `runs/artifacts/daily/...`
- run 本地 7 键：`run_factor` / `run_ic` / `run_quality_report` / `run_walk_forward_summary` / `run_universe_snapshot` / `run_report` / `run_meta` → 指向 run 目录内

**`fz factor eval`（信号轨）= 8 全局 + 8 个 `run_*` = 16 键**

- 全局归档 8 键：`factor` / `ic` / `quality_report` / `universe_snapshot` / `signal` / `signal_group_nav` / `report` / `meta` → 指向 `runs/artifacts/daily/...`
- run 本地 8 键：`run_factor` / `run_ic` / `run_quality_report` / `run_universe_snapshot` / `run_signal` / `run_signal_group_nav` / `run_report` / `run_meta` → 指向 run 目录内

（eval 无 `walk_forward_summary`；backtest 无 `signal` / `signal_group_nav`。两轨全局键集合见 `pipelines/daily_single.py` 的 `run_factor_eval` / `run_factor_backtest`。）

**非托管顶层键会被保留**：收尾时先读回磁盘上的 manifest，把 `outputs` 与所有不在托管列表里的顶层键原样留下。这就是 `stage_timings` 这类由 pipeline 自行追加的字段能存活的机制。真实样本：

```json
"stage_timings": {"IC 分析": 1.668, "策略回测": 7.257, "换手率": 0.39, "报告生成": 0.541}
```

追加自定义字段用 `record_experiment_metadata(exp_dir, key, value)`（写顶层）或 `record_experiment_output(exp_dir, key, value)`（写进 `outputs`）。不走完整 `run_experiment` 的 pipeline（如 `fz risk build`）用 `build_manifest_base(command, config)` 复用同一套基础字段，避免各自手写精简版而漏掉 `git_dirty` / `pixi_lock_sha256`。

### 3.2 模拟域（`sim/engine.py`）

写入 `sim/{run_id}/manifest.json`。**与 §3.1 无任何字段重叠**：

| 字段 | 说明 |
|---|---|
| `run_id` | |
| `n_signals` | 信号条数 |
| `git_sha` | |
| `inputs` | 输入 run 目录列表 |
| `start` / `end` | 回测窗口 |
| `n_exec_dates` | 实际执行日数 |
| `cost_model` | 成本参数字典 |
| `config` | 引擎配置字典 |
| `command` | 字符串（如 `"sim run"`），**不是数组** |

实测 `cost_model` = `{commission: 0.00025, stamp_tax: 0.001, slippage: 0.0005, borrow_annual: 0.085}`，与 `config/constants.py` 的 `COMMISSION_RATE` / `STAMP_TAX_RATE` / `SLIPPAGE_RATE` / `BORROW_RATE_ANNUAL` 逐值一致。

`config` 键：`factor_col`、`frequency`、`initial_capital`、`max_participation_rate`、`max_gross_exposure`、`max_abs_weight`、`limit_up_pct`、`limit_down_pct`、`execution_price`、`ret_definition`、`rebalance_threshold`、`strategy_type`、`strategy_params`、`cost_model`、`alpha`、`fallback_adv`。

> ⚠️ **sim manifest 里会出现裸 `Infinity`**（`max_gross_exposure` / `max_abs_weight` 无约束时），那**不是合法 JSON**。`jq`、浏览器 `JSON.parse`、以及多数非 Python 语言的解析器会直接失败。sim 域没有走清洗路径——只有 §3.3 的挖掘域做了处理。Python 的 `json.load` 能读，跨语言消费前需自行预处理。

`sim/engine.py` 的**读**取端也有约定：输入 run 目录缺 `manifest.json` → warning「疑似半成品目录」跳过；manifest 缺 `signal_date` 字段 → 同样跳过（无法作为有效信号执行）。

### 3.3 挖掘 session 域（`agents/manifest.py`）

写入 `mine_agent/{run_id}/` 或 `mine_team/{run_id}/manifest.json`。字段全集：

```text
run_id, seed, n_trials, sharpe_variance, deflation_two_sided,
iterations, params, partial, pbo, attempts[], candidates[],
library_pool_size, n_library_correlated_rejects, n_gray_zone,
n_lift_queue, lift_group, lift_results[],
lift_admissions{added_active, added_probation},
n_lift_evaluated, lift_dropped_coverage[], lift_error, objective, git_sha
[, intraday_scout]
```

关键语义：

| 字段 | 含义 |
|---|---|
| `partial` | `true` = **轮末增量快照，挖掘未跑完**。进程崩溃后留在磁盘上的就是它——消费方据此区分「跑完的结果」与「崩溃现场」 |
| `sharpe_variance` + `n_trials` | 复算候选 `dsr_pvalue` 的必要输入；`partial` 快照写 `null`（那时还没有最终 basis） |
| `deflation_two_sided` | `true` 表示 `effective_trials = 2 × n_trials` |
| `intraday_scout` | 仅当该 session 跑了分钟级探索时出现 |

> ✅ **这个域的 nan/inf 一律清洗成 `null`**：`dump_manifest` 先递归 `_sanitize`，再以 `allow_nan=False` 兜底抛错。理由是 `json.dumps` 默认把 nan 写成裸 `NaN`（同样非法 JSON），而候选数 < 2 时 `pool_pbo` 正常返回 nan 属**常态非异常**。这是三个域里唯一保证输出严格合法 JSON 的。

### 3.4 其它写入点

| 写入点 | 落盘位置 | 备注 |
|---|---|---|
| `research/combination/experiment.py` | `combinations/{run}/manifest.json` | 含 cv 参数 / seed / git_sha |
| `pipelines/research_run.py` | `research_dir/manifest.json` | `fz research run` 编排器 |
| `discovery/mining_session.py` | `mining_sessions/{session}/manifest.json` | |
| `agents/team_orchestrator.py` | `mine_team/{run}/manifest.json` | |
| `discovery/factor_library.py` | `factor_library/rebuild_{market}_manifest.json` | |
| `markets/crypto/lake.py` | `data/crypto_lake/manifest.json` | 见 [数据源与口径](data-sources.md) |
| `daily/data/intraday.py` | `data/derived/intraday_features/{version}/{freq}/manifest.json` | |
| `strategies/trend_timing.py` | 各期 run_dir | 含 `signal_date` + `weights.parquet` |
| `server/artifacts.py` | **读**取端 | 扫 `<workspace>/<domain>/<run_id>/manifest.json` 建索引，供只读展示 server |

变体文件名：`rebuild_{market}_manifest.json`、`input_manifest.json`（combinations）、`lift_test_manifest.json`（mine_team）、`job_manifest.txt`（`_ops/architecture_review`，**非 JSON**）。

lift-test 每次运行落**两份**：稳定名 `lift_test_manifest.json`（latest 指针，覆写）
+ 时间戳归档 `lift_test_manifest_{YYYYmmddTHHMMSS}.json`（永不覆写）。
归档保证成功运行的证据不被后续失败运行抹掉；只读展示 server 按精确名
`manifest.json` 建索引，不扫这两者，故无影响。

---

## 4. `data/` 根布局

```text
data/
├── raw/          Tushare 等原始落盘（分区 parquet）
├── cache/        月度成分快照 + python 因子面板缓存
├── derived/      派生特征
├── crypto_lake/  Binance Vision 数据湖（多市场）
└── _tools/       一次性数据脚本 + logs/
```

### 4.1 `data/raw/` —— 原始行情

分区路径格式 `{base_dir}/{data_type}/year={YYYY}/month={MM}/data.parquet`（`core/storage.py`）。真实子目录与体量（A 股主线在前）：

| 子目录 | 体量 | 内容 |
|---|---|---|
| `minute_1min/` | 27 G | **分钟线**（`DATA_RAW_MINUTE` 指向它） |
| `daily_basic/` | 464 M | 每日指标（市值/换手/估值） |
| `daily/` | 296 M | 日线行情 |
| `moneyflow/` | 221 M | 资金流 |
| `margin_detail/` | 166 M | 两融明细 |
| `hk_hold/` | 35 M | 北向持股 |
| `top_list/` | 12 M | 龙虎榜 |
| `adj_factor/` | 9.7 M | 复权因子 |
| `finance_fina_indicator/` | 7.1 M | 财务指标 |
| `stk_holdernumber/` | 3.3 M | 股东户数 |
| `finance/` | 2.0 M | 财务报表 |
| `index_daily_000905_SH/` | 1.6 M | 中证 500 指数日线 |
| `index_daily_000300_SH/` | 1.6 M | 沪深 300 指数日线 |
| `us_daily/` | 43 M | 美股日线（多市场，见 [多市场](../concepts/multi-market.md)） |
| `fut_daily/` | 34 M | 期货日线（多市场） |
| `fut_mapping/` | 872 K | 期货主力映射（多市场） |
| `fut_meta/` | 8.0 K | 期货元信息（多市场） |

> ⚠️ **指数日线按指数拆目录**：是 `index_daily_000300_SH` / `index_daily_000905_SH`，不是单一的 `index_daily`。写路径拼接逻辑时别当成一个目录。

### 4.2 其余

| 目录 | 内容 |
|---|---|
| `data/cache/` | 261 个 parquet：`index_member_{code}_{YYYYMM}.parquet` 月度成分快照 + `fut_basic_meta.parquet`；同时是 python 因子的面板缓存目录（`DATA_CACHE`） |
| `data/derived/bars_5min/year=YYYY/month=MM/` | 5 分钟 bar，含自己的 `manifest.json` |
| `data/derived/intraday_features/{version}/{freq}/` | 日内特征（`INTRADAY_FEATURES_DIR`），每 `{version}/{freq}` 一份 `manifest.json`；本机为 `v1/5min` |
| `data/crypto_lake/` | 多市场 crypto 数据湖：`klines_1m/` `funding/` `metrics/` `meta.parquet` `manifest.json`，详见 [数据源与口径](data-sources.md#4-多市场数据源) 与 [多市场](../concepts/multi-market.md) |
| `data/_tools/` | 一次性数据脚本（回填/补缺）与 `logs/` |

数据源接口、单位口径与缓存键完整性见 [数据源与口径](data-sources.md)。
