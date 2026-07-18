# 因子挖掘

> [FactorZen](../../README.md) · [文档](../README.md) · **因子挖掘**

挖掘的目标不是「找到 IC 高的表达式」，而是**为因子库找到有增量的候选**。这个定位决定了平台的挖掘链路和常见做法有几处明显不同：默认评估目标是**对现有库正交后的残差 IC**、搜索过程会主动避开库里已覆盖的方向、session 收尾自动把过关候选 upsert 进库并做 lift 裁决。

三个入口，从便宜到贵：

| 入口 | 提案来源 | 需要 LLM | 典型用途 |
|---|---|---|---|
| [`fz mine search`](#表达式搜索) | 随机 / 遗传算法 | ❌ | 大批量筛空间、快速基线、无网络环境 |
| [`fz mine agent`](#llm-单-agent) | LLM 单角色迭代 | ✅ | 带经济直觉的定向探索 |
| [`fz mine team`](#llm-4-角色团队) | LLM 4 角色 + Evaluator 流水线 | ✅ | **主力入口**，长战役、跨 session 记忆 |

---

## 表达式与算子库

所有挖掘产出的都是同一种东西：一棵**表达式 AST**，可以和可读字符串双向转换（`discovery/expression.py`），再编译成 polars 表达式求值。

```text
rank(ts_std(close, 20))          字符串
   ↕  parse_expr / to_expr_string
OpNode("rank", [OpNode("ts_std", [Feature("close")], window=20)])    AST
   ↓  compile
pl.Expr                          求值
```

双向可逆是很多机制的前提：LLM 提案的表达式先 parse 成 AST 做合法性与语义校验，再 `to_expr_string` 规范化——**规范形就是因子库的主键**，`rank(ts_std(close,20))` 和 `rank( ts_std( close , 20 ) )` 会归一成同一条记录，天然去重。

### 算子（31 个，`discovery/operators.py`）

| 类别 | 数量 | 语义 | 算子 |
|---|---|---|---|
| 时序 `ts` | 15 | `.over("ts_code")`，带窗口 | `ts_mean` `ts_std` `ts_sum` `ts_min` `ts_max` `ts_median` `ts_rank` `ts_zscore` `ts_skew` `ts_corr` `ts_cov` `ts_decay_linear` `delay` `delta` `pct_change` |
| 截面 `cs` | 3 | `.over("trade_date")`，无窗口 | `rank` `zscore` `scale` |
| 算术 `arith` | 13 | 逐元素 | `add` `sub` `mul` `div` `abs` `log` `sign` `sqrt` `neg` `inv` `square` `max` `min` |

编译的前提是求值表已按 `(ts_code, trade_date)` 排好序——时序算子靠 `.over("ts_code")` 分组，顺序错了结果就错。

> ℹ️ 除法一律走 `_safe_div`：分母必须 `is_finite()` 且 `abs() > 1e-12`，否则出 null。这挡的是 polars 里 `NaN.abs() > 1e-12` 判 True 导致 NaN 分母穿透守卫的坑。

### 叶子（57 个，`core/feature_schema.py` 是单一真源）

| 族 | 数量 | 例子 |
|---|---|---|
| 行情基础 | — | `open` `high` `low` `close` `vol` `amount` `vwap` `ret_1d` `log_vol` `amplitude` `intraday_ret` `overnight_ret` |
| 估值 / 规模 | 10 | `total_mv` `circ_mv` `pe_ttm` `pb` `ps_ttm` `dv_ttm` `turnover_rate` `turnover_rate_f` `volume_ratio` `float_share` |
| 基本面（PIT 按公告日对齐） | 8 | `roe` `roa` `netprofit_yoy` `or_yoy` `assets_yoy` `debt_to_assets` `grossprofit_margin` `netprofit_margin` |
| 资金流 | 8 | `net_mf_amount` `north_ratio` 等 |
| 两融 | 4 | `margin_balance` `margin_buy_ratio` `margin_ratio` `short_balance` |
| 股东户数 | 2 | `holder_num` `holder_num_chg` |
| 龙虎榜 | 2 | `top_list_flag` `top_list_net_buy` |
| **日内微观结构** | 17 | `i_rv` `i_amihud` `i_smart_money` `i_vwap_dev` …（见[日内叶子](#日内叶子)） |

> ⚠️ 日内的 17 个 `i_*` 叶子**默认不启用**，必须显式加 `--intraday-leaves`，且面板要先构建好。它们计入 57 这个总数，但不加旗标时不会出现在搜索空间里。

---

## 表达式搜索

不调 LLM，纯算法生成候选。

```bash
pixi run -- fz mine search --start 20200101 --end 20241231 --universe csi500 \
  --method genetic --trials 500 --top-k 10 --seed 42
```

### 两种方法

**`--method random`（默认）** —— 按算子类型签名递归生成合法 AST（`discovery/search/random_search.py:23`）。默认树深 3；每层有 25% 概率提前收成叶子，叶子里 10% 概率取常数（`0.5` / `1.0` / `2.0`）；窗口从 `{3, 5, 10, 20, 60}` 里抽。

**`--method genetic`** —— 交叉（把 B 的随机子树接到 A 的随机位置）+ 变异（把随机子树换成新生成的），保留精英（`discovery/search/genetic.py`）。同一个 `--seed` 下 `--workers` 并行与串行结果等价。

> ℹ️ 搜索空间的最大回看 = `max(窗口) × 树深` = `60 × 3` = 180 个交易日。数据准备阶段按这个值设预热前缀，保证空间内任何表达式都不会因预热不足被误拒（`random_search.py:13`）。

### 关键参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--trials` | `200` | 试验次数 |
| `--top-k` | `10` | 保留的头部候选数 |
| `--holdout-ratio` | `0.2` | **永久隔离**的样本外段占比 |
| `--train-ratio` | `0.7` | 挖掘段内 train/valid 切分 |
| `--decorr-threshold` | `0.7` | top-K 贪心去相关的 \|corr\| 门槛 |
| `--min-n-train` | `5` | 候选 train 段最少有效 IC 天数 |
| `--dsr-alpha` | `0.1` | 护栏 `passed` 的 DSR 显著性阈值 |
| `--objective` | `residual` | 评估目标，见下 |

全表见 [CLI 参考](../reference/cli.md#fz-mine-search)。

---

## 评估目标：为什么默认是残差

`--objective` 有两个取值：

- **`residual`（默认）** —— 候选先对库内 `active` 因子做**同日截面正交**，再算残差与前向收益的 Spearman IC。测的是「相对现有库的真增量」。
- **`raw`** —— 裸 RankIC。

残差化严格在**单日截面内**完成（`discovery/residual.py`）：库因子当日做截面 z-score、null 补 0 → 最小二乘拟合候选（含截距）→ 取残差 → 与当日前向收益算 Spearman。没有跨日状态、没有跨日拟合，所以不引入未来信息。

> ℹ️ **库为空时自动退化成 `raw`**，不需要你手工切换。第一次挖掘时看到日志里目标是 `raw` 属正常。

> ⚠️ **因子库自身的 upsert / rebuild 不用残差口径**，走裸 IC + 覆盖门。理由写在 `residual.py` 的 docstring 里：库是参照系，对参照系自身做「对库残差化」是循环定义。残差目标只用于挖掘评估。

残差 IC 天然小于裸 IC（共享方向已被剔除），所以两条通道的地板不同：`DEFAULT_IC_FLOOR = 0.015`（裸）vs `DEFAULT_RESIDUAL_IC_FLOOR = 0.010`（残差），都定义在 `discovery/guardrails.py`。

---

## 去相关的三个层次

容易混淆，逐条区分：

| 层次 | 旗标 | 时机 | 作用 |
|---|---|---|---|
| **top-K 内部去相关** | `--decorr-threshold`（默认 `0.7`） | 排序后取头部时 | 候选之间贪心剔近重复 |
| **库级正交过滤** | `--no-library-orthogonal` 关闭（默认开） | **搜索过程中** | 让提案避开库里已覆盖的方向 |
| **入库去相关打标** | 库侧 `--decorr-threshold` | 收尾 upsert 时 | 超阈值仍收录，打 `correlated` 标 |

> ⚠️ `--no-library` 与 `--no-library-orthogonal` 管两件完全不同的事：前者关**收尾时写库**，后者关**搜索期避让**。关掉一个不影响另一个。

第三层的「高相关仍收录只打标」是刻意设计，理由见[因子库与增量准入](../concepts/factor-library.md#四态状态机)。

---

## 护栏与 `passed`

每个候选评估完会打一个 `passed` 标记，默认参与筛选（`fz mine leaderboard` 默认只显示 `passed`，`--all` 是逃生口）。

判定口径由 `DEFAULT_GATE` 决定，默认 **`library`**：真（holdout 同号）+ 有信号（`|IC| ≥ floor`），**不含**单因子 DSR 显著性——因为 DSR 已挪到组合层，单因子层不再当硬门。另有 `strict` 口径（DSR 显著 + holdout 同号）供需要单因子独立显著时选用。

几个阈值（`discovery/guardrails.py`，改一处全局生效）：

| 常量 | 值 | 含义 |
|---|---|---|
| `DEFAULT_GATE` | `library` | 入池判据口径 |
| `DEFAULT_DSR_ALPHA` | `0.10` | DSR 显著性水平 |
| `DEFAULT_IC_FLOOR` | `0.015` | 裸 IC 下限 |
| `DEFAULT_RESIDUAL_IC_FLOOR` | `0.010` | 残差 IC 下限 |
| `DEFAULT_HOLDOUT_MIN_DAYS` | `60` | holdout 有效 IC 天数下限 |
| `DEFAULT_DUPLICATE_CORR` | `0.95` | 与库内因子「重复」硬拒阈值 |

> ℹ️ `holdout` 天数不足时的分类是「**覆盖不足**」而不是「无预测力」。这个区分很重要：空/稀疏 holdout 的 `ic_mean` 哨兵值 `0.0` 曾被同号门误杀过。覆盖失败也不进负例记忆——它不是方向性证据。

统计原语（bootstrap IC CI / DSR / PBO-CSCV / holdout）的数学细节见[防过拟合护栏](../concepts/guardrails.md)。

---

## LLM 单 Agent

```bash
pixi run -- fz mine agent --start 20200101 --end 20241231 --universe csi500 \
  --iterations 5 --top-k 5 --patience 2
```

单个 LLM 角色迭代提案，每轮走「生成 → 求值 → 护栏 → Critic → 反思」的闭环（`agents/orchestrator.py`）。

`mine agent` 独有的能力是 **`--human-review`**（人工复核环节）——`mine team` 没有这个旗标。

两个自愈机制值得知道：

- **`--heal-rounds`（默认 2）** —— 表达式解析失败时把错误回灌给 LLM 要求修正，最多 N 轮。设 `0` 关闭。
- **`--patience N`** —— 连续 N 轮无新候选就早停；缺省跑满 `--iterations`。

> ⚠️ LLM 挖掘需要 `FACTORZEN_LLM_*` 环境变量。**缺配置直接报错退出，不会静默降级成随机搜索。** 变量全表见[环境变量参考](../reference/environment.md)。

---

## LLM 4 角色团队

主力入口。流水线是 **Librarian → Hypothesis → Coder → Evaluator → Critic**，外加否决回路（`agents/team_orchestrator.py:1`）。

```bash
pixi run -- fz mine team --start 20200101 --end 20241231 --universe csi500 \
  --iterations 8 --structured --hypotheses-per-round 2 --llm-workers 4 --pool-subproc
```

### 4 个角色 + Evaluator

| 角色 | 是否调 LLM | 职责 |
|---|---|---|
| **Librarian** | ✅ | 跨 session 长期记忆的读写：recall 已知无效方向、已覆盖方向、被 lift 拒过的方向；session 末 record |
| **Hypothesis** | ✅ | 提经济直觉方向，注入长期记忆（避开挖穿区、优先未探索区） |
| **Coder** | ✅ | 方向 → 表达式；按 Critic 反馈修正；解析失败时按错误信息重写 |
| **Evaluator** | ❌ **确定性** | 求值 + 护栏判定，不调 LLM |
| **Critic**（Risk Auditor） | ✅ | 读候选指标判过拟合，给 `keep` / `revise_expr` / `revise_hypothesis` / `drop` 四种 verdict，驱动否决回路 |

> ⚠️ 口径是「**4 角色 + Evaluator**」。Evaluator 是确定性的求值与护栏节点，不是第五个 LLM 角色——不要写成「5 角色」。（此外还有一个可选的 **Feature Scout** 角色，只在开 `--intraday-scout` 时参与，见下节。）

Librarian 的记忆不是摆设，它有具体阈值（`agents/roles/librarian.py`）：某个叶子上的方向尝试（排除覆盖失败）≥ **15** 次且 0 次过关 → 判定为「**挖穿区**」，后续提案避开；本 session 存活叶子中历史唯一表达式数 ≤ **2** → 「**未探索区**」，优先考虑。

### 团队独有的参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--index-path` | `workspace/mine_team/experiment_index.jsonl` | 跨 session 实验登记簿 |
| `--structured` | 关 | 结构化假设（机制 / 预期符号 / 证伪判据）+ 任务分解后逐任务翻译 |
| `--hypotheses-per-round` | `1` | 每轮假设数，`>1` 提升单轮产能（护栏 / Critic 仍每轮一次） |
| `--llm-workers` | `4` | 轮内独立 LLM 调用并发度，`1` = 串行零回归 |
| `--no-campaign-prior` | 关 | 关闭跨 session 试验族记账 |
| `--no-auto-lift` | 关 | 关闭 session 末的自动组 lift 裁决 |
| `--lift-se-mult` | `1.0` | lift 准入 SE 乘数 |
| `--lift-workers` | 自适应 | session 末 lift 的并发，上限 4 |
| `--pool-subproc` | 关 | 池构建放子进程，见[内存隔离](#内存隔离pool-prebuild) |

> ℹ️ **`--no-campaign-prior` 关掉的是多重检验记账。** 默认开启时，DSR 的 N 取「历史唯一表达式 ∪ 本 session」，防止分多次挖掘来稀释多重检验惩罚。关掉会让护栏变松——只在明确知道自己在做什么时用。

### session 末的 lift 裁决

团队 session 跑完会自动做一次组 lift 裁决（`--no-auto-lift` 关闭）。这条钩子和 [`fz factor-library lift-test`](../concepts/factor-library.md)、库 rebuild 复审共用**同一个裁决函数** `lift_admission()`——三个消费方不会漂移。

裁决规则与阈值见[因子库与增量准入](../concepts/factor-library.md#裁决规则)。

---

## 日内叶子

17 个日内微观结构特征（`i_*`）可以直接当日频挖掘的叶子用。它们是从 1 分钟 bar 聚合出来的**日频**面板，语义上和 `close`、`turnover_rate` 没有区别——挖掘引擎不知道它们来自分钟数据。

**前置：先构建面板。**

```bash
pixi run -- fz data intraday-features build --start 20200101 --end 20241231 \
  --freq 5min --version v1 --workers 2
```

**然后在挖掘时启用：**

```bash
pixi run -- fz mine team --start 20200101 --end 20241231 --universe csi500 \
  --intraday-leaves --intraday-freq 5min
```

> ⚠️ **`--intraday-freq` 必须与构建面板时的 `--freq` 一致**，否则读不到面板。注意两边的取值形态也不同：面板构建用自由字符串 `5min`，不是 `fz mine --freq` 的 `{1m,5m,15m,1h,daily}` 枚举。

> ⚠️ `--intraday-leaves` **仅 `ashare`**。

> ⚠️ `i_pv_corr` 在 30min 频率下样本数只有 8，恒为 null（代码里自己标注了）。日内特征的可用性随 `--freq` 变化。

---

## LLM Scout：动态日内叶子

比固定的 17 个特征更进一步：让 LLM 直接提案 **bar 级表达式**，平台校验、聚合成日频、注入 session 当叶子用。

```bash
pixi run -- fz mine agent --start 20200101 --end 20241231 --universe csi500 \
  --intraday-scout --scout-k 4 --scout-max-leaves 12
```

`--intraday-scout` 隐含 `--intraday-leaves`，**仅 `ashare`**。

**叶名是内容寻址的**（`discovery/intraday_expr.py:128`）：

```text
ix_{sha1(agg | 规范化表达式 | freq)[:8]}
```

同一个 `(聚合函数, 规范表达式, 频率)` 三元组永远得到同一个叶名——跨 session、跨 run 天然去重，也让入库记录可复现。

**v1 DSL 的边界**（都是显式校验，越界直接 `ValueError`）：

| 维度 | 允许的取值 |
|---|---|
| bar 级叶子 | `open` `high` `low` `close` `vol` `amount` `bar_ret` |
| 算子 | **仅逐元素算术**（`arith` 类），**不允许** `ts_*` 与截面算子 |
| 日聚合 | `sum` `mean` `std` `skew` `min` `max` `first` `last` `median` |

只允许逐元素算子是有意的：bar 级的时序/截面算子语义模糊（截面是指同一分钟的全市场？），先不开。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--scout-k` | `4` | 每轮 Scout 提案条数 |
| `--scout-max-leaves` | `12` | 单个 session 最多注入的 `ix_*` 叶数 |

> ℹ️ 校验、物化、筛选都在 `agents/scout_support.py` 统一完成，`mine agent` 与 `mine team` 共用同一条验证路径——这是刻意防双路径漂移的设计。

---

## 内存隔离：pool-prebuild

全 A 长窗口挖掘时，「构建因子库池」这一步的内存峰值很高，而 Python 进程内的内存不会全额归还操作系统。解法是把池构建放进**独立子进程**——进程退出即全额归还。

**日常用法：只需要一个旗标。**

```bash
pixi run -- fz mine team --start 20200101 --end 20241231 --universe csi500 \
  --pool-subproc
```

`--pool-subproc`（或环境变量 `FACTORZEN_POOL_SUBPROC=1`）会自动派生一个 `fz pool-prebuild` 子进程，产物落在 `workspace/mine_team/_pool_cache/<key>/`。缓存键由**库文件 hash + 窗口 + 票池 + 市场 + holdout 比例 + 日内旗标**共同决定（`cli/main.py:862-871`），命中则直接复用、跳过子进程。

**手工预热：**

```bash
pixi run -- fz pool-prebuild --start 20200101 --end 20241231 --universe csi500 \
  --out workspace/factors/_cache/pool_20260718
```

产物是 `--out` 目录下的 `pool_wide.parquet` + `pool_meta.json`。

> ⚠️ **`fz pool prepare` 不存在。** 真实形态是扁平的顶层命令 `fz pool-prebuild`，无子命令，`--start` / `--end` / `--out` 三个必填。敲 `fz pool` 会直接报 invalid choice。

> ⚠️ **手工 `--out` 指定的目录不会被 `mine team` 自动发现**——自动路径的目录名是按缓存键算出来的哈希。手工预构建主要用于离线检查产物或独立排查，日常直接用 `--pool-subproc` 就好。

> ⚠️ `--pool-subproc` 与 `--no-library-orthogonal` **同时开会跳过子进程**（池根本不会被用到），命令会打印一行提示。子进程失败时会打 warning 并**回退进程内构建**，不中断挖掘。

预构建的窗口 / 票池 / `--holdout-ratio` 必须与随后的挖掘一致，否则口径不匹配。更多内存与耗时数据见[性能与资源](performance.md)。

---

## 看结果与导出

### 排行榜

```bash
pixi run -- fz mine leaderboard workspace/mine_team/20260718_120000
pixi run -- fz mine leaderboard workspace/mine_team/20260718_120000 --all
```

读的是 session 目录下的 `candidates.csv`。默认只列 `passed` 的候选；全部候选一个都没过时会提示用 `--all` 看完整的 N 个。

`candidates.csv` 的列（`discovery/mining_session.py:569`）：

```text
rank · n_trials · expression · ic_train · ir_train · ic_valid · ir_valid
max_corr · complexity · holdout_ic · dsr_pvalue · pbo · ic_ci_low · passed
residual_ic_train · residual_holdout_ic · n_residual_holdout_days
```

### 导出单日 alpha

把某个候选在某个截面日的因子值导出成 `(ts_code, alpha)` parquet，喂给组合优化：

```bash
pixi run -- fz mine export-alpha \
  --session workspace/mine_team/20260718_120000 \
  --rank 1 --date 20241231 --universe all_a \
  --out workspace/alpha/20241231.parquet
```

`--rank` 是 `candidates.csv` 里的名次（1-based）。默认只允许导出 `passed` 的候选，`--all` 放开。产物直接接 [`fz portfolio build --alpha-file`](risk-and-portfolio.md)。

---

## 产物与入库

session 目录（`workspace/mining_sessions/` · `workspace/mine_agent/` · `workspace/mine_team/` 按入口分）：

| 文件 | 内容 |
|---|---|
| `candidates.csv` | 头部候选表，列见上 |
| `manifest.json` | 可复现记录：`seed` / `method` / `n_trials` / `sharpe_variance` / `train_end` / `holdout_start` / `git_sha` / `objective` / `excluded_leaves` / `library_pool_size` / 完整候选详情 |

> ℹ️ `manifest.json` 里同时记 `n_trials`（**真实评估过的唯一表达式数**）和 `cli_n_trials`（你在命令行要求的次数）。DSR 的 deflation 门槛由 `(n_trials, sharpe_variance)` 共同决定，两个都是可复现的必要信息——这也是为什么多重检验记账必须按真实评估数而不是最终存活数。

**入库是自动的。** session 收尾会把 `passed` 候选 upsert 进 `workspace/factor_library/`（`--no-library` 关闭）。整个 upsert 块有 try/except 兜底——写库是收尾副作用，绝不拖垮挖掘产出本身。

没进库的灰区候选也不算白跑：它们可以后续走 [`fz factor-library lift-test --session <session_dir>`](../concepts/factor-library.md#常用命令) 这条第二通道，单独做 lift 实验。

复现一个入库因子：`fz factor-library list` 查到 `name`，然后 `fz factor run <name>`（manifest 的 `reproduce_note` 字段里也写了这条）。

---

## 多市场

挖掘类命令的 `--market` 是 4 值域 `{ashare, crypto, futures, us}`，默认 `ashare`。

| 市场 | 池的选法 | 注意 |
|---|---|---|
| `ashare` | `--universe`（如 `csi500`） | 唯一支持日内叶子与 scout 的市场 |
| `crypto` | `--top-n` 按成交额，或 `--symbols` 指定 | `--freq` 可选分钟级 bar 粒度 |
| `futures` | `--top-n`，或 `--symbols` | 主力连续 + 乘法后复权 |
| `us` | S&P 500 静态池按 `--top-n` 截断 | ⚠️ 见下 |

> ⚠️ **us 的成分是 ~2024 年的静态快照（约 490 个），不是 PIT 历史成分。** 用它回看历史窗口会引入**幸存者偏差**——代码 docstring 自己标注了这一点。这是平台 PIT 铁律的已知例外，A 股侧的 PIT 是严格的。详见[多市场适配](../concepts/multi-market.md)。

> ⚠️ **us / futures 只打通到挖掘 + 因子库 + 过拟合校验**，没有 `fz data fetch` 子命令、没有组合优化、没有风险模型。crypto 与 ashare 是全链路可跑的。

---

## 相关阅读

- [因子库与增量准入](../concepts/factor-library.md) —— 挖出来的候选如何被裁决入库
- [防过拟合护栏](../concepts/guardrails.md) —— DSR / PBO / holdout 的数学与咬合方式
- [多因子组合](combination.md) —— 库里的因子如何组合成策略
- [因子编写](factor-authoring.md) —— 表达式表达不了的想法怎么手写
- [多市场适配](../concepts/multi-market.md) —— 四市场的真实能力边界
- [性能与资源](performance.md) —— 挖掘耗时、内存峰值与并行策略
- [CLI 参考](../reference/cli.md#fz-mine) —— `fz mine` 全参数
