# 端到端教程

> [FactorZen](../../README.md) · [文档](../README.md) · **端到端教程**

从零跑完一条完整的 A 股研究链路：

```text
拉数据 → 挖掘 → 增量准入进库 → 风险模型 → 组合优化 → 模拟交易 → 报告
```

每一步给出**可直接复制的命令**、**预期产物路径**、以及**怎么判断这步做对了**。踩坑就地标注。

**前提**：完成[安装与环境](installation.md)。建议先跑一遍[快速上手](quickstart.md)——那里把前三步讲透了，本页会更快地带过、把篇幅留给后面的风险与组合环节。

**本教程用到的窗口**：数据与研究窗口 `20200101`–`20241231`，票池 `csi500`，组合部分按季度调仓。全程不需要 LLM。

> ℹ️ 全部日期用紧凑格式 `YYYYMMDD`。整个 CLI 只有 `fz live replay --from-date/--to-date` 是例外，用 `YYYY-MM-DD`。

---

## 第 1 步：拉数据

```bash
pixi run fz data fetch daily       --start 20200101 --end 20241231
pixi run fz data fetch daily-basic --start 20200101 --end 20241231
```

`daily` 是日线行情，`daily-basic` 是每日指标（市值 / 换手 / 估值）——**风险模型的风格因子依赖后者，不能只拉 `daily`**。

需要基本面因子时再补：

```bash
pixi run fz data fetch fundamentals --start 20200101 --end 20241231
```

**产物**：`data/raw/{daily,daily_basic}/year=YYYY/month=MM/data.parquet`；`fundamentals` 落的是 `data/raw/finance_fina_indicator/`（Tushare `fina_indicator` 全套质量 / 成长字段，按公告日做 PIT 对齐）。

**怎么判断做对了**：

```bash
pixi run smoke-data --skip-tushare
```

`[审计:daily] OK` 与 `[审计:daily_basic] OK`，行数为百万级。审计基于交易日历覆盖判断，不是「文件存在就算数」。

> ⚠️ **单位口径别踩**：`daily.amount` 是**千元**，`daily_basic.total_mv` / `circ_mv` 是**万元**。任何金额或市值阈值在写进代码前先核对单位。详见[数据源与口径](../reference/data-sources.md)。

---

## 第 2 步：挖掘候选因子

```bash
pixi run fz mine search \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --method genetic --trials 200 --top-k 10 --seed 42
```

窗口会按 `--holdout-ratio`（默认 0.2）切出一段**永久隔离的 holdout**，挖掘过程碰不到它。默认 `--objective residual`：候选的评估目标是**对库内 active 因子截面正交后的残差 IC**（库为空时自动退化为裸 RankIC）——从搜索阶段起就在找「库里没有的东西」。

**产物**：`workspace/mining_sessions/session_42_genetic/`

```text
candidates.csv    头部候选与各项指标
manifest.json     seed / 试验数 / holdout 起点 / 全部候选详情 / n_gray_zone
```

> ⚠️ **session 目录名是 `session_{seed}_{method}`，不含时间戳。** 同 seed 同方法重跑会**原地覆盖**上一次产物。要保留多次实验就换 `--seed`。

**怎么判断做对了**：

```bash
pixi run fz mine leaderboard workspace/mining_sessions/session_42_genetic --all
```

不加 `--all` 只列护栏 `passed` 的候选。护栏 `passed` 的候选在挖掘收尾时**自动 upsert 进因子库**，这是入库第一通道（`--no-library` 可关）。

> ℹ️ 想用 LLM 团队挖掘，把这一步换成 `fz mine team`（4 角色 + Evaluator，session 末还会自动跑一轮组 lift 裁决）。需要配好 `FACTORZEN_LLM_*`，缺配置会直接报错退出。详见[因子挖掘指南](../guides/mining.md)。

---

## 第 3 步：增量准入进库

单因子门槛没过、但方向与库内因子不重合的候选，被标进 **lift 队列**（灰区），走入库第二通道：**组合增量裁决**。

```bash
# 先 dry-run，只看裁决不写库
pixi run fz factor-library lift-test \
  --session workspace/mining_sessions/session_42_genetic \
  --market ashare \
  --start 20200101 --end 20241231 \
  --universe csi500
```

> ⚠️ **`lift-test` 默认就是 dry-run。** 上面这条跑完，因子库一条记录都不会变。加 `--apply` 才写库。`forward-review` 同理。

```bash
# 确认裁决结果后，写库
pixi run fz factor-library lift-test \
  --session workspace/mining_sessions/session_42_genetic \
  --market ashare \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --apply
```

**产物**（仅 `--apply` 时）：`workspace/factor_library/ashare.jsonl` 新增 `status=probation` 记录；lift 拒绝回灌该 session 所属的 `experiment_index.jsonl`（路径优先取 session manifest 里的 `index_path`，否则回退到 session 目录的**父目录**，本例即 `workspace/mining_sessions/experiment_index.jsonl`）。

**怎么判断做对了**：

```bash
pixi run fz factor-library list --market ashare
```

> ℹ️ 通过 lift 的候选默认封顶在 `probation`。转正要靠真实时间：`fz factor-library forward-track` 逐日记录 paper forward RankIC，攒够 `--min-days`（默认 60）后 `fz factor-library forward-review --apply` 裁决。完整机制见[因子库与增量准入](../concepts/factor-library.md)。

> ⚠️ `forward-track` **尚未接进 `fz ops daily` 的无人值守日链路**，probation 因子的每日记录目前需要人工执行。这是已知的接线缺口。

---

## 第 4 步：构建风险模型

Barra 式风险模型：风格 + 行业暴露、Newey-West 修正的因子协方差、收缩后的特质风险。

```bash
pixi run fz risk build \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --cov-half-life 90 --nw-lags 2
```

**你会看到**一行摘要：

```text
[risk] factors=<N> R2=<...> valid_days=<...> n_factor_mismatch=<...> → workspace/risk_models/risk_20200101_20241231
```

**产物**：`workspace/risk_models/risk_{start}_{end}/`

```text
exposures.parquet          因子暴露
factor_covariance.parquet  因子协方差
factor_returns.parquet     因子收益
specific_risk.parquet      特质风险
risk_summary.csv           摘要
manifest.json              复现清单
```

**怎么判断做对了**：`factors` 数量合理（风格 + 行业 one-hot）、`R2` 非零、`valid_days` 接近窗口内交易日数。若 `factors` 明显偏少，通常是 `daily_basic` 没拉全导致滚动风格因子退化。

> ⚠️ 本命令**没有 `--market`，是 A 股专属**。Barra 模型未接入多市场适配层；crypto 有自己独立的风险实现，futures / us 没有风险模型。

> ⚠️ **这一步的产物是给你看的，不是给下一步吃的。** `fz portfolio build` 在进程内**自己重新构建**一次风险模型，并不读取 `workspace/risk_models/` 里的文件。所以严格说第 4 步不是第 6 步的前置——它的价值在于把风险结构落成可审计、可对比的独立产物。跳过它不影响组合优化能跑。

---

## 第 5 步：导出 α 截面

组合优化吃的是**某个截面日的 α 信号文件**，来自挖掘 session 里的某个候选：

```bash
mkdir -p workspace/alpha

pixi run fz mine export-alpha \
  --session workspace/mining_sessions/session_42_genetic \
  --rank 1 \
  --date 20241231 \
  --universe csi500 \
  --out workspace/alpha/20241231.parquet
```

**你会看到**：`[mine] export-alpha: rank=1 expr='...' date=20241231 → ... (N 只股票)`

**产物**：`workspace/alpha/20241231.parquet`，两列 `ts_code` + `alpha`。

**怎么判断做对了**：打印的股票数接近该日 universe 规模。数量明显偏小说明因子在该截面大量缺值。

> ⚠️ 默认只允许导出护栏 `passed` 的候选，`--all` 才放开。`--rank` 是 `candidates.csv` 里的名次（1-based）。
>
> ⚠️ `--lookback`（默认 60 个交易日）要覆盖表达式里最长的时序算子窗口，否则该算子在截面日算不出值。

> ℹ️ **一个真实的能力边界**：α 导出目前只能从**挖掘 session** 走，因子库还没有对应的 α 导出出口。因子库的正式消费出口是 `fz combine from-library`（多因子合成研究，见[多因子组合](../guides/combination.md)），它与组合优化链路暂未打通。

---

## 第 6 步：组合优化

cvxpy 求解带约束的均值-方差问题，并产出归因。

### 单期

```bash
pixi run fz portfolio build \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --alpha-file workspace/alpha/20241231.parquet \
  --lam 1.0 --w-max 0.05 --industry-neutral \
  --run-id 20241231
```

**你会看到**：`[portfolio] status=optimal holdings=<N> → workspace/portfolios/20241231`

**产物**：`workspace/portfolios/20241231/`

```text
weights.parquet    目标权重
attribution.csv    归因明细
risk_summary.csv   风险摘要
manifest.json      含 status 与 signal_date（模拟交易靠它排调仓日程）
```

**怎么判断做对了**：`status=optimal`。若是 `infeasible`，说明约束联立无解——最常见的原因是 `--w-max` 太小、`--turnover` 预算太紧，或行业中性叠加了过强的其它约束。

> ⚠️ **`--industry-neutral` 是中性到票池的等权基准暴露，不是绝对中性到 0。** one-hot 行业列上「绝对中性到 0 + long-only + Σw=1」必然无解。当前实现用等权基准，**不等同于市值加权中性**。

> ⚠️ **不传 `--run-id` 时目录名默认取 `--end` 的日期串。** 多期循环里若忘了传，同一 `--end` 的两期会静默互相覆盖。**多期务必显式传不同的 `--run-id`。**

### 多期（模拟交易需要）

模拟交易的调仓日程 = `--portfolio-dir` 下各个 run 目录的 `signal_date`。想跑出一条像样的净值曲线，得先构建**多期**权重表。按季度调仓的例子：

```bash
for D in 20240329 20240628 20240930 20241231; do
  pixi run fz mine export-alpha \
    --session workspace/mining_sessions/session_42_genetic \
    --rank 1 --date $D --universe csi500 \
    --out workspace/alpha/$D.parquet

  pixi run fz portfolio build \
    --start 20200101 --end $D \
    --universe csi500 \
    --alpha-file workspace/alpha/$D.parquet \
    --lam 1.0 --w-max 0.05 --industry-neutral \
    --run-id $D
done
```

**产物**：`workspace/portfolios/{20240329,20240628,20240930,20241231}/`，四个 run 目录。

> ⚠️ **每一期都会重新构建一次完整风险模型**，耗时随期数线性增长。全 A 长窗口做月频调仓前先估算好时间预算，参考[性能与资源](../guides/performance.md)。

---

## 第 7 步：模拟交易

按权重表跑回测，含交易约束（停牌 / 涨跌停 / ST / 次新 / T+1）与成本（佣金、印花税、滑点、融券费）。

```bash
pixi run fz sim run \
  --portfolio-dir workspace/portfolios \
  --start 20240101 --end 20241231 \
  --run-id tutorial_2024
```

**你会看到**：`[sim] run_dir=workspace/sim/tutorial_2024 sharpe=<...> max_dd=<...> ann_ret=<...>`

**产物**：`workspace/sim/tutorial_2024/`

```text
nav.parquet      净值序列
metrics.json     汇总指标
manifest.json    含 cost_model、n_exec_dates、输入 run 目录列表
```

> ⚠️ **`--portfolio-dir` 在这里传的是「根目录」`workspace/portfolios`**，命令会遍历其下的各 `{run_id}/` 子目录组成调仓日程。**下一步 `fz report portfolio` 的同名参数传的却是「单个 run 目录」。** 同名异义，是本 CLI 最常见的传参错误，两处都要对号入座：
>
> | 命令 | `--portfolio-dir` 传什么 |
> |---|---|
> | `fz sim run` | 根目录 `workspace/portfolios` |
> | `fz report portfolio` | 单 run 目录 `workspace/portfolios/20241231` |

> ⚠️ **不传 `--run-id` 时 sim 输出目录固定叫 `workspace/sim/sim/`**，下次跑会覆盖。建议每次显式命名。

> ⚠️ **有些组合期会被静默跳过**：缺 `manifest.json` 的半成品目录、manifest 缺 `signal_date` 的目录、以及优化 `status` 非成功（如 `infeasible`）的期，都不会进调仓日程，只在日志里留 warning。**`n_exec_dates` 明显少于你构建的期数时，回头查每期的 `status`。**

**怎么判断做对了**：

```bash
pixi run fz sim show --sim-dir workspace/sim/tutorial_2024
```

`n_exec_dates` 与你构建的组合期数一致，净值序列覆盖 `--start`/`--end` 窗口。

---

## 第 8 步：出报告

```bash
pixi run fz report portfolio \
  --sim-dir workspace/sim/tutorial_2024 \
  --portfolio-dir workspace/portfolios/20241231
```

**产物**：`workspace/reports/portfolio_<run_id>.html`（`--out` 可指定别的路径）。

报告包含净值曲线、模拟指标、归因与风险摘要。`--market` 缺省时从 sim 的 `manifest.json` 自动识别。

> ⚠️ 这里的 `--portfolio-dir` 是**单个 run 目录**（见上一步的对照表），它要从中读 `attribution.csv` / `risk_summary.csv` / `manifest.json`。传根目录会读不到文件。

> ⚠️ **归因是 Brinson-Fachler 两项法**（配置效应 + 选股效应，交互项并入选股），不是 BHB 三项法；也不支持日内高频归因。

**怎么判断做对了**：浏览器打开 HTML，净值曲线与 `fz sim show` 的指标对得上，归因表非空。

> ℹ️ 想在浏览器里统一浏览所有 run 的产物，可以起只读展示服务：`pixi run serve`（依赖 fastapi/uvicorn，pixi 默认环境已含）。

---

## 全链路检查清单

跑完回头逐项确认：

| 步骤 | 产物 | 判定标准 |
|---|---|---|
| 1 拉数据 | `data/raw/daily/`、`data/raw/daily_basic/` | `smoke-data --skip-tushare` 审计 OK |
| 2 挖掘 | `workspace/mining_sessions/session_42_genetic/` | `candidates.csv` 非空，`manifest.json` 有 `holdout_start` |
| 3 准入 | `workspace/factor_library/ashare.jsonl` | `factor-library list` 能看到新记录及其 `status` |
| 4 风险 | `workspace/risk_models/risk_20200101_20241231/` | `R2` 非零，`valid_days` 接近交易日数 |
| 5 α | `workspace/alpha/*.parquet` | 行数接近 universe 规模 |
| 6 组合 | `workspace/portfolios/{date}/` | 每期 `manifest.json` 的 `status=optimal` |
| 7 模拟 | `workspace/sim/tutorial_2024/` | `n_exec_dates` = 组合期数 |
| 8 报告 | `workspace/reports/portfolio_*.html` | 曲线与指标一致 |

每一份 `manifest.json` 都记着配置、命令、`git_sha`、seed、窗口与 universe。

> ⚠️ **`git_dirty: true` 时无法凭 `git_sha` 精确重跑。** 正式实验前先把工作区改动提交掉。

---

## 更快的路：一条命令跑完

挖掘 → 组合 → 模拟 → 报告可以用编排器一次跑完，全程贯穿同一个 `run_id`：

```bash
pixi run fz research run \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --method genetic --trials 300 \
  --rebalance-days 20 --w-max 0.05 --industry-neutral
```

**产物**：同一 `run_id` 贯穿挖掘 session、`workspace/portfolios/<run_id>/`、`workspace/sim/<run_id>/`，报告落 `workspace/reports/portfolio_<run_id>.html`。

> ⚠️ **`fz research run` 目前是单因子 + in-sample 编排**，且**没有 `--market`**（A 股专属）。它适合快速看一条链路通不通，**不能替代**上面的分步流程——尤其是它不含增量准入这一步，而准入才是平台的核心主张。

---

## 下一步

- [因子库与增量准入](../concepts/factor-library.md) —— lift 裁决规则、四态状态机、向前确认
- [风险与组合优化](../guides/risk-and-portfolio.md) —— Barra 模型细节、约束体系、归因口径
- [模拟与向前执行](../guides/execution.md) —— 从模拟回测走到向前执行引擎与分歧归因
- [无人值守运营](../guides/operations.md) —— 把日链路自动化
- [CLI 参考](../reference/cli.md) —— 全部命令与参数
- [产物布局](../reference/artifacts.md) —— 每个目录、每个 `manifest.json` 字段
