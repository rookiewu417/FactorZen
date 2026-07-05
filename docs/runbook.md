# 运行手册

> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · **运行手册** · [端到端教程](end-to-end-tutorial.md) · [路线图](evolution-plan-2026.md)

所有命令从仓库根目录执行，通过 `pixi run fz` 进入项目环境。

---

## 命令速查

| 场景 | 命令 |
|------|------|
| 环境自检 | `pixi run smoke` |
| 数据 smoke | `pixi run smoke-data --start … --end …` |
| 拉日行情 | `pixi run fz data fetch daily --start … --end …` |
| 拉日基础数据 | `pixi run fz data fetch daily-basic --start … --end …` |
| 列出因子 | `pixi run fz factor list` |
| 新建因子 | `pixi run fz factor new <name>` |
| 运行因子 | `pixi run fz factor run <name> --start … --end …` |
| 参数网格扫描 | `pixi run fz factor sweep <name> --grid K=V1,V2` |
| 表达式搜索 | `pixi run fz mine search --start … --end …` |
| 导出候选 alpha | `pixi run fz mine export-alpha --session … --rank 1 --date … --out …` |
| LLM 单 Agent 挖掘 | `pixi run fz mine agent --start … --end …` |
| 多 Agent 团队挖掘 | `pixi run fz mine team --start … --end …` |
| 挖掘排行榜 | `pixi run fz mine leaderboard <session_dir>` |
| 防过拟合验收 | `pixi run fz validate overfit <factor> --start … --end …` |
| 构建风险模型 | `pixi run fz risk build --start … --end …` |
| 组合优化建仓 | `pixi run fz portfolio build --start … --end … --alpha-file …` |
| 模拟交易 | `pixi run fz sim run --portfolio-dir … --start … --end …` |
| 打印模拟绩效 | `pixi run fz sim show --sim-dir …` |
| 重建单因子报告 | `pixi run fz report build <name>` |
| 组合 Dashboard | `pixi run fz report portfolio --sim-dir … --portfolio-dir …` |
| 报告路径 | `pixi run fz report path <run_id>` |
| 历史运行列表 | `pixi run fz runs list` |
| 查看 manifest | `pixi run fz runs show <run_id>` |
| 校验配置 | `pixi run fz config validate <yaml>` |
| 质量门 | `pixi run lint && pixi run typecheck && pixi run test && pixi run coverage` |

---

## 环境自检

```bash
pixi install
cp .env.example .env          # 填入 TUSHARE_TOKEN
pixi run smoke
```

真实数据拉取需要在 `.env` 配置 `TUSHARE_TOKEN`。LLM 研究解读默认关闭；缺少 `FACTORZEN_LLM_*` 配置时自动跳过，不影响核心链路。

```bash
# 验证 Tushare 连通性 + 本地原始数据分区完整性（不进 CI）
pixi run smoke-data --start 20230101 --end 20231231
pixi run smoke-data --skip-tushare   # 仅离线审计本地分区
```

退出码：0 = 全部正常，1 = 出现 error，2 = 仅 warning。

---

## factor — 单因子研究

### `fz factor new`

**用途**：在 `workspace/factors/{frequency}/` 下生成新因子模板文件。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<name>` | 因子名（字母+下划线） |
| `--frequency` / `--freq` | 频率，`daily`（默认）/`weekly`/`monthly`/`intraday` |
| `--force` | 目标文件已存在时强制覆盖 |

```bash
pixi run fz factor new momentum_20d --frequency daily
```

**产物**：`workspace/factors/daily/momentum_20d.py`（模板，需手动填 `compute` 逻辑）

---

### `fz factor list`

**用途**：列出所有已注册因子。

```bash
pixi run fz factor list
```

**产物**：终端打印因子名称列表。

---

### `fz factor run`

**用途**：对指定因子执行完整单因子评估——IC/分层回测/walk-forward（可选）/Tear Sheet。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<name>` | 因子名 |
| `--start` | 开始日期，格式 `YYYYMMDD` |
| `--end` | 结束日期，格式 `YYYYMMDD` |
| `--universe` | 股票池，如 `csi500`（默认） |
| `--config <yaml>` | 指定 YAML 配置文件 |
| `--set K=V` | 临时覆盖配置字段，可重复 |
| `--dry-run` | 打印生效配置，不跑回测 |

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 --set preprocessing.normalizer=rank_normal
```

**产物**：`workspace/factor_evaluations/{run_id}/`（`report.html` + `manifest.json` + 因子值 parquet）

> 无 `--config` 时使用内置研究级默认配置：`csi500`、行业+市值中性化、IC/IR/分层四套策略。walk-forward 默认关闭，按需用 `--set walk_forward.enabled=true` 开启。

---

### `fz factor sweep`

**用途**：参数网格扫描——笛卡尔积跑多组配置，按指标排序出对比表。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<name>` | 因子名（或 `--config` 指定配置） |
| `--grid K=V1,V2` | 网格维度，可多个（笛卡尔积） |
| `--set K=V` | 每组固定覆盖 |
| `--sort-by` | 排序指标：`ir` / `ic_mean` / `ic_pos` / `t` |

```bash
pixi run fz factor sweep momentum_20d \
  --grid backtest.top_n=30,50,100 \
  --grid preprocessing.normalizer=zscore,rank_normal \
  --sort-by ir
```

**产物**：`workspace/factor_evaluations/sweep_{ts}/sweep_results.csv`（单组失败不中断全局，表中标注 error）

---

## mine — 因子挖掘

### `fz mine search`

**用途**：用随机/遗传搜索在算子库上自动生成并评估因子表达式。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 训练段开始日期 |
| `--end` | 训练段结束日期 |
| `--universe` | 股票池（可选，默认全 A） |
| `--method` | `random`（默认）或 `genetic` |
| `--trials` | 搜索次数，默认 200 |
| `--top-k` | 保留前 k 个表达式，默认 10 |
| `--seed` | 随机种子，默认 42，保证可复现 |

```bash
pixi run fz mine search --start 20200101 --end 20231231 \
  --method genetic --trials 200 --top-k 10 --seed 42
```

**产物**：`workspace/mining_sessions/session_{seed}_{method}/`
- `candidates.csv`：候选排行榜（列 `rank,n_trials,expression,ic_train,...`）
- `manifest.json`：参数 / seed / 复现说明
- `exported/*.py`：top 候选渲染成的因子文件，复制到 `workspace/factors/daily/` 即可被 registry 发现

> `candidates.csv` 里的 IC 为挖掘内估计（plain zscore，无中性化）。用 `fz factor run` 复跑时默认带中性化，若要对齐 IC 需加 `--set preprocessing.neutralize=false`。

---

### `fz mine export-alpha`

**用途**：把某个候选表达式在指定日期当天的截面 α 落成 `(ts_code, alpha)` 两列 parquet，直接喂给 `fz portfolio build --alpha-file`。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--session` | 挖掘 session 目录（含 `candidates.csv`） |
| `--rank` | 候选排名（1-based），默认 1 |
| `--date` | 截面日期 `YYYYMMDD`（与组合建仓 `--end` 对齐） |
| `--universe` | 股票池，默认 `all_a` |
| `--lookback` | 时序算子回看交易日数，默认 60 |
| `--out` | 输出 parquet 路径（列：`ts_code, alpha`） |

```bash
pixi run fz mine export-alpha \
  --session workspace/mining_sessions/session_42_genetic \
  --rank 1 --date 20231231 --universe all_a --out alpha.parquet
```

**产物**：`--out` 指定的 parquet（两列 `ts_code, alpha` 的单截面长表）。

---

### `fz mine leaderboard`

**用途**：读取一次挖掘 session 的 `candidates.csv`，打印排行榜。

```bash
pixi run fz mine leaderboard workspace/mining_sessions/session_42_genetic
```

**产物**：终端打印 `candidates.csv` 内容（rank / 表达式 / IC 等）。

---

### `fz mine agent`

**用途**：LLM 单 Agent 挖掘闭环——假设→生成→护栏→critic→反思。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 训练段开始 |
| `--end` | 训练段结束 |
| `--universe` | 股票池（可选） |
| `--iterations` | 反思迭代轮数，默认 5 |
| `--top-k` | 保留候选数，默认 5 |
| `--seed` | 随机种子，默认 42 |
| `--human-review` | 开启人工复核挂起 |

```bash
pixi run fz mine agent --start 20200101 --end 20231231 --iterations 10
```

**产物**：`workspace/mine_agent/{run_id}/`（`candidates.csv` + `manifest.json` + `exported/`）

> 需配置 `FACTORZEN_LLM_*` 环境变量；缺失时命令退出并提示。

---

### `fz mine team`

**用途**：多 Agent 团队挖掘——4 个角色 Agent（Hypothesis / Coder / Critic / Librarian）加 Evaluator 评估环节，支持跨轮否决与长期记忆。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` / `--end` | 训练段起止 |
| `--universe` | 股票池（可选） |
| `--iterations` | 团队迭代轮数，默认 5 |
| `--top-k` | 保留候选数，默认 5 |
| `--seed` | 随机种子，默认 42 |
| `--index-path` | 实验索引 jsonl 路径，默认 `workspace/mine_team/experiment_index.jsonl` |

```bash
pixi run fz mine team --start 20200101 --end 20231231
```

**产物**：`workspace/mine_team/{run_id}/`（`candidates.csv` + `manifest.json` + `exported/`，并更新 `experiment_index.jsonl`）

---

## validate — 防过拟合验收

### `fz validate overfit`

**用途**：对指定因子执行 Deflated Sharpe（DSR）+ block bootstrap IC 置信区间评估，只在终端打印，不落盘。单因子样本数 N=1，**不计算 PBO**（PBO/CSCV 适用于候选因子池的多重检验场景）。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<factor>` | 已注册因子名（挖掘出的表达式需先把 session `exported/*.py` 复制到 `workspace/factors/daily/`） |
| `--start` | 评估段开始 |
| `--end` | 评估段结束 |
| `--universe` | 股票池（可选） |

```bash
pixi run fz validate overfit momentum_20d --start 20200101 --end 20241231
```

**产物**：终端打印一行 `IC / IR / DSR p 值 / bootstrap IC 95% CI`；不写任何文件。

**解读**：DSR p 值越小越显著；bootstrap IC 95% CI 下界 > 0 表示 IC 显著异于零。

---

## risk — 风险模型

### `fz risk build`

**用途**：构建 Barra 风格风险模型——8 个风格因子暴露（`size / value / momentum / volatility / liquidity / quality / growth / leverage`）+ 行业因子 + Newey-West 协方差估计 + 特质风险收缩。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 估计窗口开始 |
| `--end` | 估计窗口结束 |
| `--universe` | 股票池，默认 `all_a` |
| `--cov-half-life` | 因子协方差半衰期（交易日），默认 90 |
| `--nw-lags` | Newey-West 滞后阶数，默认 2 |
| `--spec-half-life` | 特质风险半衰期（交易日），默认 90 |
| `--spec-shrinkage` | 特质风险收缩系数，默认 0.3 |

```bash
pixi run fz risk build --start 20200101 --end 20241231 \
  --cov-half-life 90 --nw-lags 2 --spec-half-life 90 --spec-shrinkage 0.3
```

**产物**：`workspace/risk_models/{run_id}/`（`exposures.parquet` + `factor_covariance.parquet` + `specific_risk.parquet` + `factor_returns.parquet` + `risk_summary.csv` + `manifest.json`）

---

## portfolio — 组合优化与归因

### `fz portfolio build`

**用途**：在 `--end` 当日做**单截面**建仓——用 cvxpy（CLARABEL solver）求解 mean-variance 二次规划，生成一组目标权重，并输出归因与风险摘要。风险模型由命令内部从同段数据现算（**不需要、也不消费** `fz risk build` 的产物）。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 数据/估计起始日 |
| `--end` | 建仓信号日（`signal_date = end`，只解一次 QP） |
| `--universe` | 股票池，默认 `all_a` |
| `--alpha-file` | α 信号文件（parquet 或 csv，两列 `ts_code, alpha` 的单截面长表） |
| `--lam` | 风险厌恶系数，默认 1.0 |
| `--w-max` | 单票上限权重，默认 0.05 |
| `--turnover` | 双边换手上限（小数），默认无（不约束换手） |
| `--industry-neutral` | 启用行业中性约束（相对股票池等权基准） |

```bash
pixi run fz portfolio build \
  --start 20200101 --end 20231231 \
  --alpha-file alpha.parquet \
  --lam 1.0 --w-max 0.05 --turnover 0.3 \
  --industry-neutral
```

**产物**：`workspace/portfolios/{run_id}/`（`weights.parquet`（列 `ts_code/target_weight/prev_weight`）+ `attribution.csv` + `risk_summary.csv` + `manifest.json`，无 HTML）

> **MVP 限制**：`--industry-neutral` 行业中性约束目前以股票池行业等权为基准（不是市值加权基准）。

---

## sim — 模拟交易

### `fz sim run`

**用途**：把 `--portfolio-dir` 根目录下各 `{run_id}/weights.parquet`（按 manifest 的 `signal_date` 串成持仓序列）执行多周期净值回测，扣除换手成本后输出净值曲线、年化收益、夏普比率、最大回撤。换手成本由内部 `CostModel` 计算，无 CLI 入口。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--portfolio-dir` | 组合产物**根目录**，其下各 `{run_id}/` 含 `weights.parquet` + `manifest.json` |
| `--start` | 模拟开始日 |
| `--end` | 模拟结束日 |
| `--run-id` | 可选，自定义输出 run_id |

```bash
pixi run fz sim run \
  --portfolio-dir workspace/portfolios \
  --start 20200101 --end 20241231
```

**产物**：`workspace/sim/{sim_id}/`（`nav.parquet` + `metrics.json` + `manifest.json`）

---

### `fz sim show`

**用途**：终端打印已有模拟结果的绩效摘要。

```bash
pixi run fz sim show --sim-dir workspace/sim/20240601_130000
```

**产物**：终端打印年化收益 / 夏普 / 最大回撤 / 年化换手等核心指标。

---

## report — 报告生成

### `fz report build`

**用途**：为指定因子重建单因子 HTML 报告。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<name>` | 因子名 |
| `--start` | 回测开始日 |
| `--end` | 回测结束日 |
| `--universe` | 股票池 |
| `--reuse` | 复用已有产物（跳过重算） |

```bash
pixi run fz report build momentum_20d \
  --start 20230101 --end 20241231 --universe csi500

# 已有产物时用 --reuse
pixi run fz report build momentum_20d \
  --start 20230101 --end 20241231 --universe csi500 --reuse
```

**产物**：因子报告 HTML（run 目录内标准化命名为 `report.html`，可经 `fz report path <run_id>` 定位）。

---

### `fz report portfolio`

**用途**：生成组合绩效 HTML Dashboard——指标卡 + 净值曲线 + 月度收益热图 + 归因 + 风险摘要。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--sim-dir` | 模拟结果目录（含 `metrics.json` / `nav.parquet`） |
| `--portfolio-dir` | 组合产物目录（含 `attribution.csv` / `risk_summary.csv` / `manifest.json`） |
| `--out` | 输出 HTML 路径（可选，默认 `workspace/reports/portfolio_<run_id>.html`） |

```bash
pixi run fz report portfolio \
  --sim-dir workspace/sim/20240601_130000 \
  --portfolio-dir workspace/portfolios/20240601_120000
```

**产物**：`workspace/reports/portfolio_<run_id>.html`（`<run_id>` 取自 `--sim-dir` 目录名；或 `--out` 指定路径）

---

### `fz report path`

**用途**：按 run_id 打印报告文件路径。

```bash
pixi run fz report path <run_id>
```

**产物**：终端打印绝对路径。

---

## data — 数据拉取

### `fz data fetch daily`

**用途**：从 Tushare 拉取日行情数据，缓存到本地 parquet 分区。

```bash
pixi run fz data fetch daily --start 20200101 --end 20241231
```

**产物**：`data/raw/daily/year=YYYY/month=MM/data.parquet`（按年/月分区缓存，增量更新）

---

### `fz data fetch daily-basic`

**用途**：从 Tushare 拉取日基础数据（市值、换手率、PE/PB 等），缓存到本地。

```bash
pixi run fz data fetch daily-basic --start 20200101 --end 20241231
```

**产物**：`data/raw/daily_basic/year=YYYY/month=MM/data.parquet`

---

## runs — 运行历史

### `fz runs list`

**用途**：列出历史运行记录（从 `workspace/factor_evaluations/experiment_index.jsonl` 读取）。

```bash
pixi run fz runs list
pixi run fz runs list --limit 50
```

---

### `fz runs show`

**用途**：打印指定 run 的完整 manifest（参数/seed/git_sha/产物路径）。

```bash
pixi run fz runs show <run_id>
```

**产物**：终端打印 manifest.json 内容。

---

## config — 配置校验

### `fz config validate`

**用途**：校验 YAML 配置文件，打印生效后的完整配置与标准输出目录；不启动回测。

```bash
pixi run fz config validate workspace/configs/daily/daily_factor_template.yaml
```

---

## YAML 配置参考

```yaml
# walk_forward 默认关闭，需要时显式开启
walk_forward:
  enabled: true
  train_days: 504
  test_days: 252
  step_days: 252
  embargo_days: 5

preprocessing:
  normalizer: rank_normal   # zscore | rank_uniform | rank_normal | quantile_normal
  neutralize: true

backtest:
  top_n: 50
  universe: csi500
```

**`--set` 临时覆盖**（写入 manifest 保证可复现，值类型从 YAML 同源推断）：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 \
  --set backtest.top_n=30 \
  --set preprocessing.neutralize=true \
  --set walk_forward.enabled=true \
  --set walk_forward.train_days=252
```

先用 `--dry-run` 确认生效配置再执行：

```bash
pixi run fz factor run momentum_20d --start 20230101 --end 20241231 --dry-run
```

---

## 质量门

```bash
pixi run lint         # ruff 代码风格
pixi run typecheck    # mypy 类型检查
pixi run test         # pytest（1185 测试）
pixi run coverage     # 覆盖率报告
```

CI 在 push / PR 到 `main` 或 `master` 时运行同一套检查。

---

## 常见故障

| 现象 | 原因 | 处理 |
|------|------|------|
| `TUSHARE_TOKEN` 未配置 / 认证失败 | `.env` 缺少或 token 过期 | 在 `.env` 填写 `TUSHARE_TOKEN=<token>`，Tushare 官网查看个人 token；离线测试不受影响 |
| `smoke-data` 报 warning：分区缺失 | 本地缓存不完整 | 运行 `fz data fetch daily --start … --end …` 补齐缺失分区 |
| 数据缺失 / NaN 过多导致 IC 全 NaN | 股票池日期超出缓存范围 | 确认 `--start`/`--end` 在已拉取数据范围内；重新执行 `fz data fetch` |
| `fz portfolio build` 报 `infeasible` | 约束冲突（换手上限 + 行业中性 + 权重上限无解） | 放宽 `--turnover` 或 `--w-max`；`--industry-neutral` 时确保股票池覆盖所有行业 |
| 报告 HTML 中文字体显示为方块 | 系统缺少中文字体 | 安装 `fonts-wqy-microhei`（Ubuntu：`sudo apt install fonts-wqy-microhei`）或在配置中指定 `matplotlib` 字体路径 |
| `report path` 找不到报告 | run_id 输错或产物目录被移动 | 先 `fz runs list` 确认 run_id，再 `fz runs show <run_id>` 查产物路径 |
| `manifest.json` 中 `git_dirty=true` | 工作树有未提交改动 | 提交或 stash 改动后重跑，确保 git_sha 可复现 |
| `qlib` 因子运行失败 | `QLIB_PROVIDER_URI` 数据包不覆盖运行日期 | 参见 [`src/factorzen/builtin_factors/qlib/README.md`](../src/factorzen/builtin_factors/qlib/README.md) |
| LLM Agent/Team 命令无响应 | `FACTORZEN_LLM_*` 环境变量未配置 | 在 `.env` 补全 LLM 配置；非 LLM 链路不受影响 |
| `fz mine search` 遗传搜索结果不稳定 | 未固定随机种子 | 加 `--seed 42`，seed 写入 manifest 可复现 |

---

## 无人值守运营（ops）

把「数据补齐 → 质量门 → 信号 → 纸面执行 → 摘要 → 发布」编排为幂等的每日链路，由外部调度触发、失败可重入续跑。

### 配置

复制 `deploy/ops.example.yaml` 为 `workspace/configs/ops.yaml` 并按需修改。关键字段：

| 字段 | 说明 |
|------|------|
| `session_dir` | 纸面执行会话目录（首跑自动 init）|
| `portfolio_run_dirs_glob` | 目标组合产物 glob（如 `workspace/portfolios/prod-*`）|
| `signal_command` | 可选外部命令每日重建组合；省略则直接消费已有产物 |
| `audit_fail_on` | `error`（默认）或 `warning`：质量门拦截级别 |
| `notify_kind` | `stdout` 或 `webhook`（从 `notify_url_env` 环境变量读 URL）|
| `publish_enabled` | `true` 则渲染 track record 净值页 |

字段拼错会被直接拒绝（extra=forbid），不会静默忽略。

### 命令

```bash
# 执行某交易日链路（缺省今天）；同日重跑自动跳过已完成阶段、从失败处续跑
pixi run fz ops daily --config workspace/configs/ops.yaml [--date 20260704]

# 查看某日各阶段状态
pixi run fz ops status --config workspace/configs/ops.yaml --date 20260704
```

非交易日由 guard 阶段自动短路（成功退出，不执行后续）。

### 调度（三选一，按可靠性递增）

1. **WSL2 systemd timer**（推荐起步）
   ```bash
   # 先确保 WSL 启用 systemd：/etc/wsl.conf 加 [boot]\nsystemd=true，然后 wsl --shutdown 重启
   cp deploy/systemd/factorzen-ops.{service,timer} ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now factorzen-ops.timer
   systemctl --user list-timers | grep factorzen   # 确认下次触发
   ```
2. **Windows 任务计划兜底**（WSL 未常驻时）：见 `deploy/windows-task.md`。
3. **VPS + Docker + cron**（真无人值守）：见下「Docker / VPS 部署」。

### 通知接入

`notify_kind: webhook` 时，把渠道 webhook URL 放进环境变量（默认名 `FACTORZEN_NOTIFY_WEBHOOK`）：

```bash
echo 'FACTORZEN_NOTIFY_WEBHOOK=https://qyapi.weixin.qq.com/...' >> .env   # 企业微信机器人
```

每日成功推送日报（NAV / 成交 / 期间收益），任一阶段失败推送错误告警；通知失败不影响主链路。

### 发布 track record 页

配置 `publish_enabled: true` 后，每日渲染净值页到 `workspace/ops/site/index.html`。发布到 GitHub Pages：

```bash
tools/publish_track_record.sh    # 推到 gh-pages 分支 live/ 路径（用临时 worktree，不碰主工作区）
```

GitHub 仓库 Settings → Pages 选 gh-pages 分支，生效后访问 `https://<user>.github.io/<repo>/live/`。

### Docker / VPS 部署

```bash
docker build -t factorzen:latest -f deploy/docker/Dockerfile .
docker compose -f deploy/docker/compose.yaml run --rm ops    # 挂载 data/workspace 卷持久化
```

VPS 上把 compose 命令挂 cron（工作日 18:30）：

```cron
30 18 * * 1-5 cd /path/to/FactorZen && docker compose -f deploy/docker/compose.yaml run --rm ops >> /var/log/factorzen-ops.log 2>&1
```

### 故障处置

| 现象 | 处理 |
|------|------|
| 某日中途失败 | 看 `workspace/ops/state/<日期>.json` 的 failed 阶段与 detail；修因后同日重跑 `fz ops daily` 自动从失败处续跑 |
| 数据质量门拦截 | detail 含缺口/空值信息；补数据（`fz data fetch`）后重跑，或临时放宽 `audit_fail_on` |
| 通知没收到 | 确认 `FACTORZEN_NOTIFY_WEBHOOK` 已设且 URL 有效；stdout 模式看日志 `workspace/runs/logs/` |
| WSL systemd timer 不触发 | 确认 `/etc/wsl.conf` 启用 systemd 且 `systemctl --user` 可用；否则用 Windows 任务计划兜底 |

诚实边界：纸面撮合是模型（开盘价 + 滑点，不建模盘口深度/排队）；WSL 方案非真无人值守，常态运营需 VPS。

---

## 兼容入口

`pixi run daily`、`pixi run report`、`fz factor test`、`fz report open` 仍保留为兼容入口。新增脚本与文档优先使用 `fz factor run`、`fz report build`、`fz report path`。
