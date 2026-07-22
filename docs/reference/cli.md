# CLI 参考手册

> [FactorZen](../../README.md) · [文档](../README.md) · **CLI 参考**

FactorZen 的全部功能都通过单一入口 `fz` 暴露。本手册覆盖 **14 个顶层命令 / 45 个叶子命令** 的完整参数面，所有默认值均取自真实 argparse 声明。

## 如何调用

本项目**没有全局 Python**，`fz` 是一个 pixi task，等价于 `python -m factorzen.cli.main`：

```bash
pixi run -- fz --help
pixi run -- fz mine search --help
```

本手册所有示例都带 `pixi run --` 前缀，可直接复制粘贴。

**通用约定：**

- 所有命令组都要求子命令。光敲 `fz factor` 会报错退出，必须写成 `fz factor list`。
- 日期一律 `YYYYMMDD`（如 `20241231`）。**唯一例外**是 `fz live replay` 的 `--from-date/--to-date`，见该节。
- 参数表中「必填」列标 ✅ 的参数缺失时 argparse 直接报错；位置参数在「参数」列以 `<name>` 标注。
- 布尔旗标（`flag`）不带值，出现即为真。

---

## 命令速查表

| 顶层命令 | 一句话用途 |
|---|---|
| [`fz factor`](#fz-factor) | 创建 / 列出 / 评估单个因子，含参数网格扫描 |
| [`fz report`](#fz-report) | 生成单因子报告与组合仪表盘 HTML |
| [`fz data`](#fz-data) | 拉取行情与财务数据、回补 crypto 数据湖、构建日内特征面板 |
| [`fz runs`](#fz-runs) | 列出历史 run 记录 |
| [`fz mine`](#fz-mine) | 因子挖掘：搜索 / Agent / 团队 / 库池预构建 |
| [`fz factor-library`](#fz-factor-library) | 因子库登记簿：重建、查询、lift 准入、向前跟踪与复审 |
| [`fz research`](#fz-research) | 端到端编排：挖掘 → 组合构建 → 模拟 → 报告，同一 run_id |
| [`fz validate`](#fz-validate) | 单因子过拟合检验（Deflated Sharpe + bootstrap CI） |
| [`fz risk`](#fz-risk) | 构建 Barra 风险模型（风格/行业暴露 + 协方差 + 特质风险） |
| [`fz portfolio`](#fz-portfolio) | 组合优化求解 + 归因 |
| [`fz sim`](#fz-sim) | 模拟交易回测与指标查看 |
| [`fz live`](#fz-live) | 向前执行：会话初始化、逐日推进、replay、分歧归因 |
| [`fz combine`](#fz-combine) | 多因子组合的四方法 OOS 对比实验 |
| [`fz ops`](#fz-ops) | 无人值守运营；含 `validate-config` 校验 YAML run config |

---

## fz factor

单因子工作流：模板创建、注册表查询、评估、参数扫描。

### fz factor new

创建一个用户因子的模板文件。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<name>` | str | — | ✅ | 因子名，同时决定模板文件名 |
| `--frequency` / `--freq` | `daily` \| `weekly` \| `monthly` \| `intraday` | `daily` | | 因子注册频率 |
| `--force` | flag | 关 | | 覆盖已存在的同名模板 |

> ⚠️ 这里的 `--freq` 是**因子注册频率**，取值 `daily/weekly/monthly/intraday`，与 `fz mine` / `fz sim` 下表示 **bar 粒度**的 `--freq {1m,5m,15m,1h,daily}` 完全是两回事。本手册全篇会在每个 `--freq` 出现处注明语义。

```bash
pixi run -- fz factor new my_reversal --freq daily
```

**产物**：`workspace/factor_store/ashare/<name>/factor.py` + `meta.json`（三件套脚手架）。

### fz factor list

列出已注册的因子。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--frequency` / `--freq` | `daily` \| `weekly` \| `monthly` \| `intraday` | `daily` | | 因子注册频率（同上，非 bar 粒度） |

```bash
pixi run -- fz factor list --freq daily
```

**产物**：无，仅打印。

### fz factor eval

因子研究评估（信号层，**纯毛口径**）：RankIC / 衰减 / 单调性 / 信号多空分层 / 换手。**不跑**日环撮合、walk-forward、benchmark。

> ⚠️ 本轨**刻意不提供任何成本参数**。粗略的 bps 折算既不是真实撮合，又会让人误以为这里能算净收益。
> 要看成本、约束与可实现性，走 [`fz factor backtest`](#fz-factor-backtest)。换手率仍会给出，但它只是**信号换手强度的度量**，不折算成本。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<name>` | str | 无 | | 因子名；可省略并改由 `--config` 提供 |
| `--start` | str | 无 | | 起始日 `YYYYMMDD` |
| `--end` | str | 无 | | 终止日 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名（如 `csi500`） |
| `--frequency` / `--freq` | `daily` \| `weekly` \| `monthly` | `daily` | | 因子注册频率（**无 `intraday`**，与 `factor new/list` 不同） |
| `--benchmark` | str | 无 | | 保留参数面一致；eval 轨忽略 |
| `--config` | str | 无 | | YAML run config 路径 |
| `--seed` | int | `42` | | 全局随机种子 |
| `--set KEY=VALUE` | str，可重复 | 无 | | 覆盖任意配置字段 |
| `--dry-run` | flag | 关 | | 只打印生效配置，不执行 |
| `--exec-lag` | int | `1` | | 成交滞后（交易日）。默认 1=可实现口径；`0`=旧 close→close（不可实现，仅对照用） |
| `--exec-price-col` | str | `open_adj` | | 成交价格列。默认 `open_adj`（open[t+2]/open[t+1]） |
| `--n-groups` | int | `5` | | 截面分位组数；多空取最高组减最低组 |

```bash
pixi run -- fz factor eval momentum_20 --start 20220101 --end 20241231 \
  --universe csi500 --n-groups 10
```

**产物**（两处落盘，命名规则不同）：  
- **全局归档** `workspace/runs/artifacts/daily/`：长名 `{name}_{start}_{end}_ic.parquet`、`_signal.json`、`_signal_group_nav.parquet`、`_meta.json`、`_quality.json`；报告为 `{name}_{start}_{end}_eval.html`（不覆盖交易轨 `.html`）。  
- **run 目录** `workspace/factor_evaluations/<run_id>/`：短名 `factor.parquet`、`ic.parquet`、`signal.json`、`signal_group_nav.parquet`、`report.html`、`meta.json`、`quality.json`、`universe.parquet`、`manifest.json`。

### fz factor backtest

模拟交易回测（日环撮合，净口径）：策略回测 + 约束/成本 + 换手 + walk-forward + 单调性 + benchmark。**不跑**信号层向量化回测。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<name>` | str | 无 | | 因子名；可省略并改由 `--config` 提供 |
| `--start` | str | 无 | | 起始日 `YYYYMMDD` |
| `--end` | str | 无 | | 终止日 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名（如 `csi500`） |
| `--frequency` / `--freq` | `daily` \| `weekly` \| `monthly` | `daily` | | 因子注册频率（**无 `intraday`**，与 `factor new/list` 不同） |
| `--benchmark` | str | 无 | | 基准指数代码（计算超额收益） |
| `--config` | str | 无 | | YAML run config 路径 |
| `--seed` | int | `42` | | 全局随机种子 |
| `--set KEY=VALUE` | str，可重复 | 无 | | 覆盖任意配置字段 |
| `--dry-run` | flag | 关 | | 只打印生效配置，不执行 |
| `--exec-lag` | int | `1` | | 成交滞后（交易日）。默认 1=可实现口径；`0`=旧 close→close（不可实现，仅对照用） |
| `--exec-price-col` | str | `open_adj` | | 成交价格列。默认 `open_adj`（open[t+2]/open[t+1]） |

```bash
pixi run -- fz factor backtest momentum_20 --start 20220101 --end 20241231 \
  --universe csi500 --set backtest.top_n=30
```

**产物**（两处落盘，命名规则不同）：  
- **全局归档** `workspace/runs/artifacts/daily/`：长名 `{name}_{start}_{end}_ic.parquet`、`_walk_forward.json`、`_meta.json`、`_quality.json`；报告为 `{name}_{start}_{end}.html`（交易轨，与 eval 的 `_eval.html` 并存）。  
- **run 目录** `workspace/factor_evaluations/<run_id>/`：短名 `factor.parquet`、`ic.parquet`、`walk_forward.json`、`report.html`、`meta.json`、`quality.json`、`universe.parquet`、`manifest.json`。

### fz factor sweep

在 `--grid` 指定的维度上做参数网格扫描，按指标排序输出。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<name>` | str | 无 | | 因子名；可改由 `--config` 提供 |
| `--config` | str | 无 | | 基准 YAML run config |
| `--grid KEY=V1,V2,...` | str，可重复 | 无 | | 一个网格维度，重复该旗标可加多维 |
| `--set KEY=VALUE` | str，可重复 | 无 | | 施加到每个组合的固定覆盖 |
| `--start` | str | 无 | | 起始日 `YYYYMMDD` |
| `--end` | str | 无 | | 终止日 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |
| `--sort-by` | str | `ir` | | 排序指标，可填 `ir` / `ic_mean` / `ic_pos` / `t` |

```bash
pixi run -- fz factor sweep momentum_20 --start 20220101 --end 20241231 \
  --grid backtest.top_n=30,50,100 --sort-by ir
```

**产物**：`workspace/factor_evaluations/sweep_<YYYYMMDD_HHMMSS>/sweep_results.csv`。

---

## fz report

报告工作流：单因子报告与组合仪表盘。

### fz report build

生成单因子报告（单页 HTML）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<name>` | str | 无 | | 因子名 |
| `--factor` | str | 无 | | 因子名（与位置参数二选一） |
| `--start` | str | 无 | | 起始日 `YYYYMMDD` |
| `--end` | str | 无 | | 终止日 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |
| `--frequency` / `--freq` | `daily` \| `weekly` \| `monthly` | `daily` | | 因子注册频率 |
| `--reuse` | flag | 关 | | 复用已有产物，不重算 |
| `--benchmark` | str | 无 | | 基准指数代码 |
| `--config` | str | 无 | | YAML run config 路径 |

```bash
pixi run -- fz report build momentum_20 --start 20220101 --end 20241231 \
  --universe csi500 --reuse
```

**产物**：报告 HTML 落 `workspace/runs/artifacts/daily/reports/`（按因子分桶）。

### fz report path

打印某个 run 的报告路径。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<run_id>` | str | — | ✅ | run 标识 |

```bash
pixi run -- fz report path 20260718_120000_momentum_20
```

**产物**：无，仅打印路径。

### fz report portfolio

生成组合仪表盘 HTML（净值、归因、风险摘要）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--sim-dir` | str | 无 | | 模拟产物目录（含 `metrics.json`） |
| `--portfolio-dir` | str | 无 | | 组合构建产物目录（含 `attribution.csv` / `risk_summary.csv` / `manifest.json`） |
| `--out` | str | 无 | | HTML 输出路径；缺省落 `workspace/reports/portfolio_<run_id>.html` |
| `--market` | `ashare` \| `crypto` | **无（自动识别）** | | 市场语境；缺省从 sim manifest 推断。`crypto` = USDT 计价 / 365 日年化 / 计资金费 / 按 sector 归因 |

> ⚠️ **`--portfolio-dir` 在这里是「单个 run 目录」**（即 `workspace/portfolios/20241231/`），而 [`fz sim run`](#fz-sim-run) 的同名参数是**组合产物根目录**（`workspace/portfolios/`，其下才是各 `{run_id}/`）。同名异义，传错会读不到文件。

> ⚠️ 本命令的 `--market` 只有 `ashare` / `crypto` **两个取值**且默认为空（自动识别），而 `fz mine` / `fz factor-library` 等命令的 `--market` 是 `ashare/crypto/futures/us` **四值**、默认 `ashare`。`--market` 不是全局统一参数。

```bash
pixi run -- fz report portfolio \
  --sim-dir workspace/sim/20260718_120000 \
  --portfolio-dir workspace/portfolios/20241231
```

**产物**：`workspace/reports/portfolio_<run_id>.html`（或 `--out` 指定路径）。

---

## fz data

数据工作流：原始数据拉取、crypto 数据湖回补、日内特征面板构建。

### fz data fetch

拉取原始数据进本地 parquet 缓存。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<data_type>` | `daily` \| `daily-basic` \| `fundamentals` \| `flows` \| `margin_detail` \| `stk_holdernumber` \| `top_list` | — | ✅ | 数据类型 |
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |

```bash
pixi run -- fz data fetch daily --start 20200101 --end 20260718
```

**产物**：`data/raw/` 下按类型分目录的 parquet 缓存。需要 `.env` 中的 `TUSHARE_TOKEN`。

### fz data crypto backfill

从 Binance Vision 回补 1 分钟 K 线 / 资金费率 / 持仓量到本地数据湖。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--symbols` | str | 无 | | 逗号分隔的 symbol 列表；缺省 = 按上月成交额 Top-N 自动选池 |
| `--top-n` | int | `50` | | 自动选池规模 |
| `--lake-root` | str | `data/crypto_lake` | | 数据湖根目录 |

```bash
pixi run -- fz data crypto backfill --start 20240101 --end 20241231 --top-n 50
```

**产物**：`data/crypto_lake/` 下分区 parquet。

### fz data intraday-features build

从 1 分钟数据湖构建日频的「日内特征面板」，即挖掘中 `i_*` 叶子的数据来源。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--freq` | str（自由字符串） | `5min` | | **日内面板的 bar 频率**，非 choices 限定 |
| `--version` | str | `v1` | | 特征电池版本 |
| `--codes` | str | 无 | | 逗号分隔的 ts_code 过滤 |
| `--overwrite` | flag | 关 | | 当 battery_hash 与既有 manifest 不符时重写 |
| `--force` | flag | 关 | | 强制重算所有月份（忽略已覆盖月的增量跳过） |
| `--workers` | int | `1` | | 月级进程并行度 |

> ⚠️ 这是 `--freq` 的**第三种语义**：日内特征面板的 bar 频率，取自由字符串（`5min` 而非 `5m`）。它既不是 `factor new` 的注册频率，也不是 `mine search` 的 `{1m,5m,15m,1h,daily}` 枚举。下游消费该面板的命令用的是 `--intraday-freq`，两者取值必须对齐。

> ⚠️ `--workers` 会显著吃内存：单月峰值约 7.6 GiB。24 GiB 内存机器建议最多设 2，设 >2 会打印告警。

```bash
pixi run -- fz data intraday-features build --start 20200101 --end 20260410 \
  --freq 5min --version v1 --workers 2
```

**产物**：`data/derived/intraday_features/` 下按频率与版本分区的 parquet + `manifest.json`。

### fz data intraday-features status

打印日内特征面板的 manifest 与分区覆盖情况。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--freq` | str | `5min` | | 日内面板 bar 频率 |
| `--version` | str | `v1` | | 特征电池版本 |

```bash
pixi run -- fz data intraday-features status --freq 5min --version v1
```

**产物**：无，仅打印。

---

## fz runs

历史 run 记录查询（仅 list；单条 manifest 请直接读 `workspace/factor_evaluations/<run_id>/manifest.json`）。

### fz runs list

列出已记录的 run。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--limit` | int | `20` | | 最多打印多少行 |

```bash
pixi run -- fz runs list --limit 20
```

**产物**：无，仅打印。

---

## fz mine

因子挖掘工作流：从无 LLM 的随机/遗传搜索，到 LLM 单 Agent，再到多角色团队。

> ⚠️ 本组命令的 `--market` 是 **4 值域** `{ashare, crypto, futures, us}`，默认 `ashare`；而 `fz portfolio build` / `fz sim run` / `fz report portfolio` 的 `--market` 只有 `{ashare, crypto}`。跨命令拼脚本时不要假设取值域一致。

> ⚠️ 本组命令的 `--freq {1m,5m,15m,1h,daily}` 指 **crypto 的 bar 粒度**，默认 `daily`；**A 股只支持 `daily`**。它与 `fz factor` 的因子注册频率、`fz data intraday-features` 的面板频率是三种不同语义。

### fz mine search

随机 / 遗传搜索候选因子表达式（不调用 LLM）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名（如 `csi500`） |
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--top-n` | int | `50` | | crypto/futures/us 池规模 |
| `--method` | `random` \| `genetic` | `random` | | 搜索方法 |
| `--trials` | int | `200` | | 试验次数 |
| `--top-k` | int | `10` | | 保留的头部候选数 |
| `--seed` | int | `42` | | 随机种子 |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |
| `--exec-lag` / `--exec-price-col` | | 1 / open_adj | | 成交口径 |
| `--set KEY=VALUE` | str，可重复 | 无 | | 高级覆盖（见[高级覆盖](#高级覆盖--set)） |

```bash
pixi run -- fz mine search --start 20200101 --end 20241231 --universe csi500 \
  --method genetic --trials 500 --top-k 10 --seed 42
# 高级：pixi run -- fz mine search ... --set objective=raw --set workers=4
```

**产物**：`workspace/mining_sessions/<run_id>/`；默认 upsert 到 `workspace/factor_library/`。

### fz mine leaderboard

打印某个挖掘 session 的候选排行榜。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<session_dir>` | str | — | ✅ | 挖掘 session 目录 |
| `--all` | flag | 关 | | 含未过护栏的候选（默认只显示 `passed`） |

```bash
pixi run -- fz mine leaderboard workspace/mining_sessions/20260718_120000 --all
```

**产物**：无，仅打印。

### fz mine export-alpha

把单个候选在某个截面日的 alpha 导出成 `(ts_code, alpha)` parquet，用于喂给 [`fz portfolio build`](#fz-portfolio-build)。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session` | str | — | ✅ | 含 `candidates.csv` 的挖掘 session 目录 |
| `--rank` | int | `1` | | `candidates.csv` 中的名次（1-based） |
| `--date` | str | — | ✅ | 截面日 `YYYYMMDD` |
| `--universe` | str | `all_a` | | 票池名 |
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--top-n` | int | `50` | | crypto/futures 池规模 |
| `--lookback` | int | `60` | | 时序算子的回看交易日数 |
| `--out` | str | — | ✅ | 输出 parquet 路径（列：`ts_code`, `alpha`） |
| `--all` | flag | 关 | | 允许导出未过护栏的候选（默认只允许 `passed`） |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |

```bash
pixi run -- fz mine export-alpha --session workspace/mining_sessions/20260718_120000 \
  --rank 1 --date 20241231 --universe all_a --out workspace/alpha/20241231.parquet
```

**产物**：`--out` 指定的 parquet 文件。

### fz mine agent

LLM 单 Agent 引导的挖掘。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` / `--end` | str | — | ✅ | 窗口 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |
| `--market` | 四选一 | `ashare` | | 市场剖面 |
| `--symbols` | str | 无 | | crypto/futures/us 显式标的 |
| `--top-n` | int | `50` | | 池规模 |
| `--iterations` | int | `5` | | 迭代轮数 |
| `--top-k` | int | `5` | | 头部候选数 |
| `--seed` | int | `42` | | 随机种子 |
| `--human-review` | flag | 关 | | 人工审阅 |
| `--freq` / `--exec-*` | | | | crypto bar / 成交口径 |
| `--set KEY=VALUE` | 可重复 | 无 | | 高级覆盖：`heal_rounds`/`patience`/`objective`/`intraday_*`/`scout_*` 等 |

```bash
pixi run -- fz mine agent --start 20200101 --end 20241231 --universe csi500 --iterations 5
# 高级：--set heal_rounds=0 --set patience=3 --set intraday_scout=true
```

**产物**：`workspace/mine_agent/<run_id>/`。

### fz mine team

多角色团队挖掘（Hypothesis / Coder / Critic / Librarian）。**表面参数 ≤14 + `--set`**。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` / `--end` | str | — | ✅ | 窗口 |
| `--universe` | str | 无 | | 票池 |
| `--market` | 四选一 | `ashare` | | 市场 |
| `--symbols` | str | 无 | | 非 A 股显式标的 |
| `--top-n` | int | `50` | | 池规模 |
| `--iterations` | int | `5` | | 轮数 |
| `--top-k` | int | `5` | | 头部候选 |
| `--seed` | int | `42` | | 种子 |
| `--structured` | flag | 关 | | 结构化假设 + 任务分解 |
| `--pool-subproc` | flag | 关 | | 库池子进程预构建（等价 env `FACTORZEN_POOL_SUBPROC=1`） |
| `--freq` / `--exec-*` | | | | crypto bar / 成交口径 |
| `--set KEY=VALUE` | 可重复 | 无 | | 高级覆盖（见下表） |

```bash
pixi run -- fz mine team --start 20200101 --end 20241231 --universe csi500 \
  --iterations 8 --structured --pool-subproc \
  --set hypotheses_per_round=2 --set llm_workers=4
```

**产物**：`workspace/mine_team/<run_id>/`；默认 upsert 因子库。

### fz mine pool-prebuild

原顶层 `fz pool-prebuild`，现归位 `mine` 组。为 `--pool-subproc` 在独立子进程构建库池。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` / `--end` | str | — | ✅ | 窗口 |
| `--out` | str | — | ✅ | 池缓存输出目录 |
| `--universe` / `--market` / `--symbols` / `--top-n` | | | | 与 team 同源 |
| `--index-path` | str | `workspace/mine_team/experiment_index.jsonl` | | |
| `--library-root` | str | 无 | | 缺省 = index 同级 `factor_library` |
| `--holdout-ratio` | float | `0.2` | | |
| `--intraday-leaves` / `--intraday-freq` | | 关 / `5min` | | |

```bash
pixi run -- fz mine pool-prebuild --start 20200101 --end 20241231 --universe csi500 \
  --out workspace/mine_team/_pool_cache/manual
```

**产物**：`--out` 下 `pool_wide.parquet` + `pool_meta.json`。

### 高级覆盖 `--set`

`mine search` / `agent` / `team` 与 `factor-library lift-test` 支持 `--set KEY=VALUE`（可重复）。未知 KEY **fail-loudly** 并列出合法键。布尔用 `true/false/1/0`。

#### mine team 合法 KEY（旧旗标 → 等价写法）

| KEY | 旧默认 | 旧 CLI | `--set` 示例 |
|---|---|---|---|
| `llm_workers` | 4 | `--llm-workers` | `--set llm_workers=4` |
| `heal_rounds` | 2 | `--heal-rounds` | `--set heal_rounds=0` |
| `objective` | residual | `--objective` | `--set objective=raw` |
| `hypotheses_per_round` | 1 | `--hypotheses-per-round` | `--set hypotheses_per_round=2` |
| `index_path` | workspace/mine_team/experiment_index.jsonl | `--index-path` | `--set index_path=...` |
| `patience` | None | `--patience` | `--set patience=3` |
| `no_library` | false | `--no-library` | `--set no_library=true` |
| `no_library_orthogonal` | false | `--no-library-orthogonal` | `--set no_library_orthogonal=true` |
| `no_campaign_prior` | false | `--no-campaign-prior` | `--set no_campaign_prior=true` |
| `no_auto_lift` | false | `--no-auto-lift` | `--set no_auto_lift=true` |
| `no_sleeve_gate` | false | `--no-sleeve-gate` | `--set no_sleeve_gate=true` |
| `lift_se_mult` | 1.0 | `--lift-se-mult` | `--set lift_se_mult=1.5` |
| `lift_workers` | None | `--lift-workers` | `--set lift_workers=1` |
| `intraday_leaves` | false | `--intraday-leaves` | `--set intraday_leaves=true` |
| `intraday_freq` | 5min | `--intraday-freq` | `--set intraday_freq=5min` |
| `intraday_scout` | false | `--intraday-scout` | `--set intraday_scout=true` |
| `scout_k` | 4 | `--scout-k` | `--set scout_k=5` |
| `scout_max_leaves` | 12 | `--scout-max-leaves` | `--set scout_max_leaves=16` |

#### mine search 合法 KEY

`workers` / `holdout_ratio` / `train_ratio` / `decorr_threshold` / `min_n_train` / `dsr_alpha` / `no_library` / `no_library_orthogonal` / `objective` / `intraday_leaves` / `intraday_freq`

#### mine agent 合法 KEY

`patience` / `heal_rounds` / `no_library_orthogonal` / `objective` / `intraday_leaves` / `intraday_freq` / `intraday_scout` / `scout_k` / `scout_max_leaves`

#### factor-library lift-test 合法 KEY

`top_m` / `queue_ic_floor` / `include_sub_floor` / `threshold` / `library_root` / `se_mult` / `allow_active` / `horizon` / `lift_workers` / `intraday_leaves` / `intraday_freq`

---

## fz factor-library

因子库登记簿：分市场、全信息、自动维护。库内每条记录带 `status`（`active` / `probation` / `correlated` / `no_lift`）与证据字段。

> ⚠️ 本组所有子命令的 `--market` 都是 4 值域 `{ashare, crypto, futures, us}`，默认 `ashare`（`lift-null` 除外——它是纯模拟，不分市场）。

> ⚠️ `fz factor-library --help` 的帮助字符串**漏列了 `lift-null` 子命令**，但该命令真实存在且可用，见 [下文](#fz-factor-library-lift-null)。

### fz factor-library rebuild

从历史产物在统一默认窗口重算，并重建某个市场的因子库。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--start` | str | 无 | | 覆盖默认窗口起点；缺省 = 最近 6 年滚动 |
| `--end` | str | 无 | | 覆盖默认窗口终点；缺省 = 数据最新端 |
| `--universe` | str | 无 | | A 股票池名（如 `csi300`） |
| `--horizon` | int | `1` | | 前向收益持有期（交易日） |
| `--top-n` | int | `50` | | crypto/futures 池规模；us 为 S&P500 静态池截断 |
| `--symbols` | str | 无 | | 仅 crypto/futures/us：逗号分隔 symbols |
| `--decorr-threshold` | float | `0.7` | | 去相关 \|corr\| 门槛；超阈值仍收录但标记 `correlated` |
| `--holdout-ratio` | float | `0.2` | | holdout 比例 |
| `--only` | str… | 无 | | **定向重估**：只重估这些表达式（自动规范化，须已在库） |
| `--only-file` | str | 无 | | **定向重估**：从文件读表达式（一行一条，`#` 开头与空行跳过） |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |

```bash
pixi run -- fz factor-library rebuild --market ashare --universe csi500 --horizon 1
```

**产物**：重写 `workspace/factor_library/` 下该市场的登记簿与 `{market}.md`。

#### 定向重估（`--only` / `--only-file`）

不带 `--only`/`--only-file` 时是**全量重建**：清空该市场登记簿、重估全部历史源、
重跑全局贪心去相关、并对全部 lift 轨记录重跑 add-one lift。为几条记录付这个代价
（且会重排全库 `status`）通常不是想要的——算子实现变更后补估一小撮记录、或批量
补算存量 `admission_ic` / `lift_metric` 时，用定向重估：

```bash
# 少量目标：直接列在命令行
pixi run -- fz factor-library rebuild --market ashare --universe csi300 \
    --only "ts_decay_linear(close, 20)" "ts_decay_linear(vol, 10)"

# 批量补账：一行一条写进文件（可由登记簿 jsonl 生成）
pixi run -- fz factor-library rebuild --market ashare --universe csi300 \
    --only-file targets.txt
```

定向模式的语义：

- **绝不清库**，子集之外的记录一个字节都不动（含 `updated_at`）；
- 只评估子集；lift 轨复审也只覆盖子集；
- 去相关**只降不升**：目标可被下调为 `correlated`（与库内未重估的 active 超阈），
  但**绝不会**被上调回 `active`。想让 `correlated` / `no_lift` 升回 `active`，
  必须跑一次全量重建——只有全局重排才有权威口径；
- 已在库的目标**不再过单因子准入门**（准入门管「进库」，不管「留任」）。重估后不
  满足该门的记录仍会写入真实指标，但会在 stderr 大声列出、并记进 manifest 的
  `targeted_gate_failed`，由操作者决定是否跑全量重建淘汰；
- 给了定向旗标却解析出空目标集 → 直接报错退出（不静默降级成全量重建）。

manifest 增记 `targeted` / `n_targeted` / `targeted_missing`（不在库的目标）/
`targeted_python_skipped` / `targeted_gate_failed` / `fresh`。

### fz factor-library list

列出库内因子（rank / expression / holdout_ic / status）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |

```bash
pixi run -- fz factor-library list --market ashare
```

**产物**：无，仅打印。

### fz factor-library show

打印单个因子的全部字段。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--expression` | str | 无 | | 按表达式（规范形）查询 |
| `--rank` | int | 无 | | 按库内排名查询（1-based，`holdout_ic` 降序） |

> ⚠️ `--expression` 与 `--rank` 两者都默认为空，需要至少给一个来定位因子。

```bash
pixi run -- fz factor-library show --market ashare --rank 1
```

**产物**：无，仅打印。

### fz factor-library lift-test

对灰区候选 / registry python 因子做**组合增量 lift 实验**，通过者以 `status=probation` 入库。这是因子进库的第二通道（第一通道是挖掘收尾的自动 upsert）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session` | str，空格分隔多值 | 无 | | 挖掘 session 的 run 目录（含 `manifest.json`），可给多个 |
| `--factor` | str，空格分隔多值 | 无 | | registry 中的 python 型因子名；要求 `--market ashare` 且 `--universe` 必填 |
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--start` | str | — | ✅ | 评估窗口起点 `YYYYMMDD` |
| `--end` | str | — | ✅ | 评估窗口终点 `YYYYMMDD` |
| `--universe` | str | 无 | | A 股票池名 |
| `--top-m` | int | `20` | | 按 \|residual_ic_train\| 取 top-M 控成本；`--top-m 0` = 全测逃生口。截断会在 stderr 大声打印并记 `truncated_from` |
| `--queue-ic-floor` | float | 无（残差 `0.008` / 裸 IC `0.010`） | | 噪声地板：`\|train IC\|` 低于此值的候选默认剔出组门；无 IC 指标的候选（如 `--factor` 注入）不受影响 |
| `--include-sub-floor` | flag | 关 | | 逃生口：sub-floor 候选照旧进组门（复现旧行为）。组门是等权残差组合，噪声占多数时会连坐拒掉真信号 |
| `--threshold` | float | 无（= `0.001`） | | RankIC lift 阈值 |
| `--seed` | int | `0` | | 随机种子 |
| `--library-root` | str | 无（= `workspace/factor_library`） | | 因子库根目录 |
| `--apply` | flag | 关 | | **写库**：通过者入库，并把 lift 拒绝写回 experiment_index |
| `--dry-run` | flag | 关 | | 只打印不写库（已是默认行为，保留为兼容旗标）；与 `--apply` 互斥 |
| `--se-mult` | float | `1.0` | | lift 准入 SE 乘数：`lift ≥ max(threshold, se_mult × SE)` |
| `--allow-active` | flag | 关 | | 允许 lift 裁决直接写 `active`（默认封顶 `probation`） |
| `--admission-start` | str | 无 | | 覆盖由 session manifest holdout 推导的 lift 评分窗起点 |
| `--admission-end` | str | 无 | | 覆盖评分窗终点 |
| `--horizon` | int | 无 | | lift 前向持有期；缺省跟随 session manifest 的挖掘 horizon |
| `--lift-workers` | int | 无（自适应） | | 候选级 lift 线程并发，上限 4；`1` = 串行 |
| `--top-n` | int | `50` | | crypto/futures/us 池规模 |
| `--symbols` | str | 无 | | 仅 crypto/futures/us：逗号分隔 symbols |
| `--intraday-leaves` | flag | 关 | | 启用日内特征叶子 `i_*` 装帧（仅 `ashare`；库内已有 `i_*` 因子时会自动置位） |
| `--intraday-freq` | str | `5min` | | 日内特征面板频率 |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |

> ⚠️ **本命令默认 dry-run**：不加 `--apply` 时只打印裁决结果，既不写因子库也不写 experiment_index。下面第一条示例**不会落库**，第二条才会。养成先 dry-run 看结果、确认后再 `--apply` 的习惯。

> ⚠️ `--session` 与 `--factor` 在 argparse 层都不是 required，但**至少要给一个**，否则运行期报错。两者都是 `nargs="+"` 风格——多值用**空格分隔**（`--session a b c`），不是逗号，也不是重复旗标。

```bash
# 1) 先 dry-run 看裁决（不写库）
pixi run -- fz factor-library lift-test \
  --session workspace/mine_team/20260718_120000 \
  --market ashare --start 20200101 --end 20241231 --universe csi500

# 2) 确认后写库
pixi run -- fz factor-library lift-test \
  --session workspace/mine_team/20260718_120000 \
  --market ashare --start 20200101 --end 20241231 --universe csi500 --apply
```

**产物**：仅 `--apply` 时写盘——更新 `workspace/factor_library/`（新记录 `status=probation`），并把 lift 拒绝回灌 experiment_index。

### fz factor-library lift-null

lift 统计层的 null 校准：在 H0（无真实 lift）下扫 `se_mult × min_blocks` 网格的误准入率，用来给准入阈值定标。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--n-days` | int | `290` | | 配对评分日数（≈ holdout 量级） |
| `--daily-sigma` | float | `0.01` | | 日差分（`cand_ic − base_ic`）标准差量级 |
| `--ar1` | float | `0.3` | | 日差分的 AR(1) 自相关（由重叠前向收益导致） |
| `--se-mults` | str（逗号分隔） | `1.0,1.645,2.0` | | SE 乘数网格 |
| `--min-blocks` | str（逗号分隔） | `0,6,10` | | 最低块数网格（`0` = 不设，即现状） |
| `--n-sims` | int | `5000` | | 模拟次数 |
| `--seed` | int | `0` | | 随机种子 |

> ⚠️ 本命令是**纯蒙特卡洛模拟**，不读也不写真实因子库，因此也**没有 `--market`**。它只回答「按当前阈值，纯噪声候选会有多大概率被误准入」。

```bash
pixi run -- fz factor-library lift-null --n-days 290 \
  --se-mults 1.0,1.645,2.0 --min-blocks 0,6,10 --n-sims 5000
```

**产物**：无，仅打印误准入率网格。

### fz factor-library forward-track

记录 as_of 日库内因子的 paper forward RankIC。确认窗口随真实时间累积。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--date` | str | 无（= 数据最新交易日） | | 确认日 `YYYYMMDD` |
| `--root` | str | 无（= `workspace/factor_library`） | | 因子库根目录 |
| `--universe` | str | 无 | | forward 截面票池；缺省 = 库记录准入口径的众数 |
| `--allow-backfill` | flag | 关 | | 允许 as_of 超期的补录 / 初始播种（仍写真实 `recorded_at` 供审计） |
| `--max-backfill-days` | int | `10` | | as_of 相对 wall-clock 允许的最大日历滞后天数 |

> ⚠️ `--universe` 必须与因子准入时的口径一致，否则 forward 证据与准入证据不可比。缺省值已按库记录众数自动对齐，一般不需要手动指定。

> ⚠️ 默认**拒绝历史回灌**：as_of 距今超过 `--max-backfill-days` 时会被拒。初次播种需显式加 `--allow-backfill`。

```bash
pixi run -- fz factor-library forward-track --market ashare --universe csi500
```

**产物**：向 `workspace/factor_library/` 中各因子的 forward 记录追加一条观测。

### fz factor-library forward-review

裁决 `probation` 因子的 paper forward 证据：晋升为 `active`，或降级为 `no_lift`。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--min-days` | int | `60` | | 裁决所需的最低有效 forward 天数 |
| `--se-mult` | float | `1.645` | | 块 SE 乘数（≈ 单侧 95%） |
| `--block-days` | int | `20` | | 块 SE 的块长（交易日） |
| `--apply` | flag | 关 | | **写库**：promote → `active` / demote → `no_lift` |
| `--root` | str | 无（= `workspace/factor_library`） | | 因子库根目录 |

> ⚠️ **本命令默认 dry-run**，与 `lift-test` 同理：不加 `--apply` 只打印裁决建议，不改动任何记录。下面的第一条示例不落库。注意这里**没有** `--dry-run` 兼容旗标（`lift-test` 才有）。

```bash
# 先看裁决建议（不写库）
pixi run -- fz factor-library forward-review --market ashare --min-days 60

# 确认后落库
pixi run -- fz factor-library forward-review --market ashare --min-days 60 --apply
```

**产物**：仅 `--apply` 时更新 `workspace/factor_library/` 中的 `status`。

---

## fz research

端到端研究编排。

### fz research run

一条命令跑完：挖掘 → 取头部 `passed` 因子 → 按调仓日循环构建组合 → 模拟 → 出报告，全程贯穿同一个 `run_id`。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--universe` | str | 无（运行时解析为 `all_a`） | | 票池名；parser 默认 `None`，编排器内回落 `all_a` |
| `--method` | `random` \| `genetic` | `random` | | 挖掘搜索方法 |
| `--trials` | int | `200` | | 挖掘试验次数 |
| `--top-k` | int | `10` | | 挖掘保留的头部候选数 |
| `--seed` | int | `42` | | 随机种子 |
| `--rebalance-days` | int | `20` | | 调仓间隔（交易日，≈ 月频） |
| `--warmup` | int | `60` | | 起始跳过的交易日数，留给时序算子 lookback |
| `--lookback` | int | `60` | | 因子计算的 lookback 交易日数 |
| `--lam` | float | `1.0` | | 风险厌恶系数 |
| `--w-max` | float | `0.05` | | 单票权重上限 |
| `--turnover` | float | 无（无约束） | | 换手预算 |
| `--industry-neutral` | flag | 关 | | 行业中性到票池等权基准 |
| `--run-id` | str | 无（= `research_<seed>_<method>`） | | 贯穿全链路的 run_id |
| `--intraday-leaves` | flag | 关 | | 启用日内特征叶子 `i_*`（需先跑 `fz data intraday-features build`；仅 A 股） |
| `--intraday-freq` | str | `5min` | | 日内特征面板频率 |

> ⚠️ 本命令**没有 `--market`**，是 A 股专属链路。crypto 需分步走 `fz mine` → `fz portfolio build --market crypto` → `fz sim run --market crypto`。

> ⚠️ 行业中性是中性到票池的**等权基准**暴露，不是绝对中性到 0——后者与 long-only + Σw=1 联立必然无解。

```bash
pixi run -- fz research run --start 20200101 --end 20241231 --universe csi500 \
  --method genetic --trials 300 --rebalance-days 20 --w-max 0.05 --industry-neutral
```

**产物**：同一 `run_id` 贯穿多个阶段目录——挖掘 session、`workspace/portfolios/<run_id>/`、`workspace/sim/<run_id>/`，报告 HTML 落 `workspace/reports/portfolio_<run_id>.html`。

---

## fz validate

过拟合与稳健性检验。

### fz validate overfit

对单个因子做 Deflated Sharpe + bootstrap 置信区间检验。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<factor>` | str | 无 | | 已注册的因子名（A 股） |
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--expression` | str | 无 | | 待检验的因子表达式；`--market` 为 `crypto`/`futures`/`us` 时**必填** |
| `--top-n` | int | `50` | | crypto/futures/us 池规模 |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |

> ⚠️ 位置参数 `<factor>` 与 `--expression` 是两条不同入口：A 股走注册因子名，非 A 股市场没有注册表，必须走 `--expression` 传表达式。

```bash
# A 股：注册因子名
pixi run -- fz validate overfit momentum_20 --start 20200101 --end 20241231 --universe csi500

# crypto：必须给表达式
pixi run -- fz validate overfit --market crypto \
  --expression "rank(ts_std(close,20))" --start 20230101 --end 20241231 --freq 1h
```

**产物**：无，仅打印检验结果。

---

## fz risk

风险模型工作流。

### fz risk build

构建 Barra 风险模型：风格 + 行业暴露、Newey-West 协方差、特质风险。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--universe` | str | `all_a` | | 票池名 |
| `--cov-half-life` | int | `90` | | 因子协方差的指数加权半衰期（交易日） |
| `--nw-lags` | int | `2` | | Newey-West 自相关修正的滞后阶数 |
| `--spec-half-life` | int | `90` | | 特质风险的指数加权半衰期 |
| `--spec-shrinkage` | float | `0.3` | | 特质风险的收缩系数 |

> ⚠️ 本命令**没有 `--market`**，是 A 股专属。

```bash
pixi run -- fz risk build --start 20200101 --end 20241231 --universe all_a \
  --cov-half-life 90 --nw-lags 2
```

**产物**：`workspace/risk_models/` 下的因子暴露、协方差矩阵、特质风险与 `manifest.json`。

---

## fz portfolio

组合构建与归因。

### fz portfolio build

用 cvxpy 求解带约束的组合优化，并产出归因。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--universe` | str | `all_a` | | 票池名 |
| `--alpha-file` | str | — | ✅ | α 信号文件（parquet/csv，列 `ts_code` + `alpha`），通常来自 [`fz mine export-alpha`](#fz-mine-export-alpha) |
| `--lam` | float | `1.0` | | 风险厌恶系数 |
| `--w-max` | float | `0.05` | | 单票权重上限 |
| `--turnover` | float | 无（无约束） | | 换手预算 |
| `--industry-neutral` | flag | 关 | | 行业中性到票池等权基准 |
| `--market` | `ashare` \| `crypto` | `ashare` | | 市场剖面；`crypto` = 市场中性做空 |
| `--top-n` | int | `50` | | crypto 池规模 |
| `--gross-limit` | float | `1.0` | | crypto 毛敞口上限 Σ\|w\| |
| `--run-id` | str | 无（= `--end` 日期串） | | 产物子目录名 |
| `--out-dir` | str | `workspace/portfolios` | | 组合产物**根目录** |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |

> ⚠️ **`--run-id` 不传会用 `--end` 的日期串做目录名。** 做多期构建时若忘了区分，后一期会静默覆盖前一期的产物。多期循环务必显式传不同的 `--run-id`。

> ⚠️ 本命令的 `--market` 只有 `{ashare, crypto}` 两值，与 `fz mine` / `fz factor-library` 的四值域不同。

```bash
pixi run -- fz portfolio build --start 20200101 --end 20241231 --universe all_a \
  --alpha-file workspace/alpha/20241231.parquet \
  --lam 1.0 --w-max 0.05 --industry-neutral --run-id 20241231
```

**产物**：`workspace/portfolios/<run_id>/`（或 `--out-dir` 下），含 `weights.parquet`、`attribution.csv`、`risk_summary.csv`、`manifest.json`。

---

## fz sim

模拟交易。

### fz sim run

按组合权重表跑模拟交易回测（含交易约束与成本）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--portfolio-dir` | str | — | ✅ | 组合产物**根目录**，其下各 `{run_id}/` 含 `weights.parquet` + `manifest.json` |
| `--start` | str | — | ✅ | 起始日 `YYYYMMDD` |
| `--end` | str | — | ✅ | 终止日 `YYYYMMDD` |
| `--run-id` | str | 无 | | 可选的输出 run_id |
| `--market` | `ashare` \| `crypto` | `ashare` | | 市场剖面；`crypto` = 计资金费 + 做空的 NAV 回测 |
| `--top-n` | int | `50` | | crypto 池规模 |
| `--freq` | `1m` \| `5m` \| `15m` \| `1h` \| `daily` | `daily` | | crypto bar 粒度 |

> ⚠️ **`--portfolio-dir` 在这里是「根目录」**（`workspace/portfolios/`），命令会遍历其下的各 `{run_id}/` 子目录组成调仓日程。而 [`fz report portfolio`](#fz-report-portfolio) 的同名参数指的是**单个 run 目录**（`workspace/portfolios/20241231/`）。同名异义，是常见的传参错误来源。

```bash
pixi run -- fz sim run --portfolio-dir workspace/portfolios \
  --start 20200101 --end 20241231
```

**产物**：`workspace/sim/<run_id>/`，含 `metrics.json`、净值序列与 `manifest.json`。

### fz sim show

打印某次模拟的指标。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--sim-dir` | str | — | ✅ | 模拟输出目录（含 `metrics.json`） |

```bash
pixi run -- fz sim show --sim-dir workspace/sim/20260718_120000
```

**产物**：无，仅打印。

---

## fz live

向前执行工作流（纸面 / 实盘）。一个「会话」由 `--session-dir` 标识，`init` 建立、`step` 逐日推进、`status` 查看、`report` 出归因；`replay` 则在历史窗口上一次性重放。

> ⚠️ `--broker` 目前只有 `paper` 一个取值（纸面撮合），默认即 `paper`。

> ⚠️ **`--portfolio-run-dir` 是 append 型多值参数**：多个目录靠**重复旗标**给出（`--portfolio-run-dir A --portfolio-run-dir B`），既不是逗号分隔，也不是空格分隔。这与 `fz factor-library lift-test --session`（空格分隔）风格不同。本 CLI 中三种多值风格并存，逐命令看表即可。

### fz live replay

在历史窗口上 replay 出向前 NAV。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session-dir` | str | — | ✅ | 会话目录 |
| `--portfolio-run-dir` | str，**可重复** | — | ✅ | 组合单 run 目录，重复旗标可给多个 |
| `--start` | str | — | ✅ | 行情窗口起点 `YYYYMMDD` |
| `--end` | str | — | ✅ | 行情窗口终点 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |
| `--initial-cash` | float | `1000000.0` | | 初始资金 |
| `--broker` | `paper` | `paper` | | 撮合适配器 |
| `--from-date` | str | 无 | | 窗口内进一步裁剪的起点，格式 **`YYYY-MM-DD`** |
| `--to-date` | str | 无 | | 窗口内进一步裁剪的终点，格式 **`YYYY-MM-DD`** |
| `--seed` | int | `0` | | 随机种子 |

> ⚠️ **`--from-date` / `--to-date` 用带横杠的 `YYYY-MM-DD`**（如 `2024-06-01`），而同一条命令里的 `--start` / `--end` 以及本 CLI 其余所有日期参数都是紧凑的 `YYYYMMDD`。这是全 CLI 唯一的日期格式例外。

```bash
pixi run -- fz live replay --session-dir workspace/live/s2 \
  --portfolio-run-dir workspace/portfolios/20241231 \
  --start 20240101 --end 20241231 --from-date 2024-06-01 --to-date 2024-12-31
```

**产物**：`--session-dir` 下的向前 NAV 序列与会话状态文件。

### fz live init

初始化一个向前执行会话。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session-dir` | str | — | ✅ | 会话目录 |
| `--initial-cash` | float | `1000000.0` | | 初始资金 |
| `--slippage-bps` | float | `0.0` | | 滑点（基点） |
| `--broker` | `paper` | `paper` | | 撮合适配器 |

```bash
pixi run -- fz live init --session-dir workspace/live/s1 --initial-cash 1000000
```

**产物**：`--session-dir` 下的会话状态文件。

### fz live step

推进一个交易日（可续跑，重复执行同一日幂等）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session-dir` | str | — | ✅ | 会话目录 |
| `--date` | str | — | ✅ | 推进到的交易日 `YYYYMMDD` |
| `--portfolio-run-dir` | str，**可重复** | — | ✅ | 组合单 run 目录 |
| `--start` | str | — | ✅ | 行情窗口起点 `YYYYMMDD`（需含 ADV 回看） |
| `--end` | str | — | ✅ | 行情窗口终点 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |

> ⚠️ `--start` 要往前留足够的回看天数，成交量约束用的 ADV 需要历史窗口；只给 `--date` 当天会导致约束失效。

```bash
pixi run -- fz live step --session-dir workspace/live/s1 --date 20241231 \
  --portfolio-run-dir workspace/portfolios/20241231 --start 20240101 --end 20241231
```

**产物**：追加更新 `--session-dir` 下的持仓、成交与 NAV 记录。

### fz live status

打印会话当前状态。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session-dir` | str | — | ✅ | 会话目录 |

```bash
pixi run -- fz live status --session-dir workspace/live/s1
```

**产物**：无，仅打印。

### fz live report

生成向前执行与回测之间的分歧归因报告。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session-dir` | str | — | ✅ | 会话目录 |
| `--portfolio-run-dir` | str，**可重复** | — | ✅ | 组合单 run 目录 |
| `--start` | str | — | ✅ | 行情窗口起点 `YYYYMMDD` |
| `--end` | str | — | ✅ | 行情窗口终点 `YYYYMMDD` |
| `--universe` | str | 无 | | 票池名 |

```bash
pixi run -- fz live report --session-dir workspace/live/s1 \
  --portfolio-run-dir workspace/portfolios/20241231 --start 20240101 --end 20241231
```

**产物**：`--session-dir` 下的归因报告。

---

## fz combine

多因子组合的 OOS 对比实验，四种方法：`equal_weight` / `ic_weighted` / `max_ir` / `lgbm`。前三个子命令的区别只在**因子从哪来**：`run` 吃裸 parquet，`from-session` 吃挖掘 session，`from-library` 吃因子库登记簿。第四个 `backtest` 把 OOS 组合分数接进统一日环策略回测。

`run` / `from-session` / `from-library` 共享同一组切分与输出参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--train-days` | int | `120` | 训练窗长度（交易日） |
| `--test-days` | int | `20` | 测试窗长度（交易日） |
| `--purge-days` | int | `5` | 训练/测试之间的 purge 间隔 |
| `--embargo-days` | int | `0` | 测试窗之后的 embargo 间隔 |
| `--methods` | str | `all` | 逗号分隔的方法名，或 `all` |
| `--seed` | int | `0` | 随机种子 |
| `--run-id` | str | 无 | 输出子目录名 |
| `--out-dir` | str | `workspace/combinations` | 产物根目录 |

### fz combine run

从裸 parquet 文件直接跑四方法 OOS 对比。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--factor` | str，**可重复** | — | ✅ | 因子 parquet（列 `trade_date`, `ts_code`, `factor_value`），重复旗标可加多个因子 |
| `--ret` | str | — | ✅ | 前向收益 parquet（列 `trade_date`, `ts_code`, `ret`） |
| 共享参数 | | | | 见上表 |

> ⚠️ `--factor` 是 append 型：多个因子靠**重复旗标**（`--factor a.parquet --factor b.parquet`），不能写成逗号分隔。注意与 `from-session` / `lift-test` 的 `--session`（空格分隔多值）区分。

```bash
pixi run -- fz combine run \
  --factor workspace/f/a.parquet --factor workspace/f/b.parquet \
  --ret workspace/ret/h5.parquet --train-days 120 --test-days 20 --purge-days 5
```

**产物**：`workspace/combinations/<run_id>/`（或 `--out-dir` 下），含各方法的 OOS 指标、`oos_scores/<method>.parquet` 与 `manifest.json`。

### fz combine from-session

从挖掘 session 的因子库直接跑组合 OOS（因子物化与收益面板自动生成）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--session` | str，空格分隔多值 | — | ✅ | 挖掘 session 目录（含 `candidates.csv`），可传多个跨 run 合并去重 |
| `--start` | str | — | ✅ | 物化窗口起点 `YYYYMMDD` |
| `--end` | str | — | ✅ | 物化窗口终点 `YYYYMMDD` |
| `--universe` | str | 无（全 A） | | 票池名 |
| `--horizon` | int | `5` | | 前向收益持有期（交易日） |
| `--top-n` | int | 无（全取） | | 只取库前 N 个因子 |
| `--decorr-threshold` | float | `0.7` | | 贪心去相关阈值，\|corr\| > 阈值剔除近亲；`1.0` = 关闭 |
| `--all` | flag | 关 | | 含未过护栏的因子（默认只用 `passed`） |
| 共享参数 | | | | 见上表 |

```bash
pixi run -- fz combine from-session \
  --session workspace/mine_team/20260718_120000 \
  --start 20200101 --end 20241231 --universe csi500 --horizon 5
```

**产物**：`workspace/combinations/<run_id>/`。

### fz combine from-library

从因子库登记簿选品后跑组合 OOS。这是因子库的正式消费出口。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--market` | `ashare` \| `crypto` \| `futures` \| `us` | `ashare` | | 市场剖面 |
| `--statuses` | str（逗号分隔） | `active` | | status 过滤，仅允许 `active` / `probation` / `correlated` / `no_lift` |
| `--library-root` | str | 无（= `workspace/factor_library`） | | 因子库根目录 |
| `--start` | str | — | ✅ | 物化窗口起点 `YYYYMMDD` |
| `--end` | str | — | ✅ | 物化窗口终点 `YYYYMMDD` |
| `--universe` | str | 无（全 A） | | 票池名；**库内含 python 型因子时必填** |
| `--horizon` | int | `5` | | 前向收益持有期（交易日） |
| `--top-n` | int | 无（全取） | | 只取库前 N 个因子 |
| `--decorr-threshold` | float | `0.7` | | 贪心去相关阈值；`1.0` = 关闭 |
| 共享参数 | | | | 见上表 |

> ⚠️ 本子命令**没有 `--all`**（`from-session` 才有）。要放宽选品范围请用 `--statuses`，例如 `--statuses active,probation`。`--statuses` 有自定义校验：非法值或空串会被 argparse 直接拒绝。

```bash
pixi run -- fz combine from-library --market ashare --statuses active,probation \
  --start 20200101 --end 20241231 --universe csi500 --horizon 5
```

**产物**：`workspace/combinations/<run_id>/`。

### fz combine backtest

将组合 OOS 分数（或任意截面分数面板）接入统一日环策略回测，输出净值 / 换手 / 成本后指标。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--scores` | str | — | 与 `--run-dir` 二选一 | 分数 parquet（`trade_date`, `ts_code`, 分数列） |
| `--run-dir` | str | — | 与 `--scores` 二选一 | combine 产物目录，读 `oos_scores/<method>.parquet` |
| `--method` | str | `equal_weight` | | 配合 `--run-dir` 选方法 |
| `--score-col` | str | 无（自动） | | 分数列；缺省取除键列外唯一数值列，多列则必填 |
| `--strategy` | str | `quantile_ls_5` | | 与 `fz factor backtest` 默认一致；另支持 `topn_long_only` 等既有策略 |
| `--start` | str | — | ✅ | 回测起 `YYYYMMDD` |
| `--end` | str | — | ✅ | 回测止 `YYYYMMDD` |
| `--universe` | str | `all_a` | | PIT membership 票池（全 A 为标准资产口径） |
| `--market` | str | `ashare` | | 当前仅 `ashare` |
| `--cost-bps` | float | 无（= LinearCostModel 默认） | | 单边成本 bps；`0` = 零成本 |
| `--rebalance-days` | int | 无（= 逐日） | | 调仓间隔（交易日）。`1`/缺省=逐日；`k>1` 桥层降采样分数并前向填充，引擎仍日环、净值逐日更新 |
| `--run-id` | str | 时间戳 | | 输出子目录名 |
| `--out-dir` | str | `workspace/combine_backtests` | | 产物根目录 |

```bash
pixi run -- fz combine backtest \
  --run-dir workspace/combinations/exp1 --method equal_weight \
  --start 20200101 --end 20241231 --universe all_a
```

**产物**：`workspace/combine_backtests/<run_id>/` 下 `manifest.json` + `metrics.json` + `nav.parquet`。

---

## fz ops

无人值守运营的每日链路。

### fz ops daily

执行一个交易日的完整无人值守链路（按 `ops.yaml` 声明的阶段依次推进）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--config` | str | — | ✅ | `ops.yaml` 配置路径；模板见仓库内 `deploy/ops.example.yaml` |
| `--date` | str | 无（= 今天） | | 目标交易日 `YYYYMMDD` |

```bash
pixi run -- fz ops daily --config deploy/ops.example.yaml --date 20241231
```

**产物**：`workspace/ops/state/<YYYY-MM-DD>.json` 记录各阶段状态；各阶段自身的产物落在各自的目录下。

### fz ops status

打印某日各阶段的执行状态。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `--config` | str | — | ✅ | `ops.yaml` 配置路径 |
| `--date` | str | 无（= 今天） | | 目标交易日 `YYYYMMDD` |

```bash
pixi run -- fz ops status --config deploy/ops.example.yaml --date 20241231
```

**产物**：无，仅打印。

### fz ops validate-config

校验一份 YAML run config（原 `fz config validate`，handler 复用）。

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---|---|---|
| `<path>` | str | — | ✅ | YAML run config 路径 |

```bash
pixi run -- fz ops validate-config workspace/configs/daily/volume_return_corr_20d.yaml
```

**产物**：无，仅打印校验结果。

---

## 相关文档

- [架构](../concepts/architecture.md) — 各能力模块如何组织与衔接
- [因子库与增量准入](../concepts/factor-library.md) — 平台核心裁决机制
- [端到端教程](../getting-started/end-to-end-tutorial.md) — 从零跑通一条完整研究链路
- [因子编写](../guides/factor-authoring.md) — 自定义因子的写法
- [因子挖掘](../guides/mining.md) — 表达式搜索、LLM 挖掘、日内叶子
- [配置参考](configuration.md) · [产物布局](artifacts.md) · [环境变量](environment.md) · [数据源与口径](data-sources.md)
