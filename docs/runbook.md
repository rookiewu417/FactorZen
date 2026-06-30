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
| LLM 单 Agent 挖掘 | `pixi run fz mine agent --start … --end …` |
| 多 Agent 团队挖掘 | `pixi run fz mine team --start … --end …` |
| 挖掘排行榜 | `pixi run fz mine leaderboard <session_dir>` |
| 防过拟合验收 | `pixi run fz validate overfit <factor> --start … --end …` |
| 构建风险模型 | `pixi run fz risk build --start … --end …` |
| 组合优化建仓 | `pixi run fz portfolio build --start … --end … --alpha-file …` |
| 模拟交易 | `pixi run fz sim run --portfolio-dir … --start … --end …` |
| 打印模拟绩效 | `pixi run fz sim show --sim-dir …` |
| 重建 Tear Sheet | `pixi run fz report build <name>` |
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

**用途**：在 `src/factorzen/factors/` 下生成新因子模板文件。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<name>` | 因子名（字母+下划线） |
| `--frequency daily` | 频率，默认 daily |

```bash
pixi run fz factor new momentum_20d --frequency daily
```

**产物**：`src/factorzen/factors/momentum_20d.py`（模板，需手动填 `compute` 逻辑）

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

**产物**：`workspace/factor_evaluations/{run_id}/`（Tear Sheet HTML + manifest.json + 因子值 parquet）

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

## mine — 因子挖掘（M1/M5/M6）

### `fz mine search`

**用途**：用随机/遗传搜索在算子库上自动生成并评估因子表达式（M1）。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 训练段开始日期 |
| `--end` | 训练段结束日期（holdout 内置隔离） |
| `--method` | `random`（默认）或 `genetic` |
| `--trials` | 搜索次数，默认 100 |
| `--top-k` | 保留前 k 个表达式，默认 10 |
| `--seed` | 随机种子，保证可复现 |

```bash
pixi run fz mine search --start 20200101 --end 20231231 \
  --method genetic --trials 200 --top-k 10 --seed 42
```

**产物**：`workspace/discovery/{session_id}/`（top-k 表达式 + IC 统计 + 去相关矩阵）

---

### `fz mine leaderboard`

**用途**：读取一次挖掘 session 的结果，打印排行榜。

```bash
pixi run fz mine leaderboard workspace/discovery/20240101_120000
```

**产物**：终端打印表达式 / IC / IR 排行榜。

---

### `fz mine agent`

**用途**：LLM 单 Agent 挖掘闭环——假设→生成→护栏→critic→反思（M5）。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 训练段开始 |
| `--end` | 训练段结束 |
| `--iterations` | 反思迭代轮数，默认 5 |

```bash
pixi run fz mine agent --start 20200101 --end 20231231 --iterations 10
```

**产物**：`workspace/discovery/agent_{session_id}/`（同 `mine search`，含 LLM 思考日志）

> 需配置 `FACTORZEN_LLM_*` 环境变量；缺失时命令退出并提示。

---

### `fz mine team`

**用途**：5 角色多 Agent 团队挖掘（Hypothesis/Coder/Critic/Librarian/Evaluator），支持跨轮否决与长期记忆（M6）。

```bash
pixi run fz mine team --start 20200101 --end 20231231
```

**产物**：`workspace/discovery/team_{session_id}/`（角色日志 + 最终候选因子集 + experiment_index.jsonl 更新）

---

## validate — 防过拟合验收（M2）

### `fz validate overfit`

**用途**：对指定因子执行 Deflated Sharpe（DSR）+ block bootstrap IC 置信区间 + PBO/CSCV 过拟合概率评估；holdout 段永久隔离，不参与训练。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `<factor>` | 因子名或表达式字符串 |
| `--start` | 全段开始（内部自动切分 holdout） |
| `--end` | 全段结束 |

```bash
pixi run fz validate overfit momentum_20d --start 20200101 --end 20241231
```

**产物**：`workspace/validation/{run_id}/`（DSR 报告 JSON + bootstrap CI 图 + PBO 矩阵）

**解读**：DSR > 0 且 bootstrap IC CI 不含零为通过；PBO < 0.5 为可接受。

---

## risk — 风险模型（M3）

### `fz risk build`

**用途**：构建 Barra 风格风险模型——8 个风格因子暴露 + 行业因子 + Newey-West 协方差估计 + 特质风险收缩（M3）。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 估计窗口开始 |
| `--end` | 估计窗口结束 |
| `--cov-half-life` | 协方差半衰期（交易日），默认 63 |
| `--nw-lags` | Newey-West 滞后阶数，默认 5 |
| `--spec-shrinkage` | 特质风险收缩系数，默认 0.1 |

```bash
pixi run fz risk build --start 20200101 --end 20241231 \
  --cov-half-life 63 --nw-lags 5 --spec-shrinkage 0.1
```

**产物**：`workspace/risk_models/{run_id}/`（因子暴露矩阵 parquet + 因子协方差矩阵 + 特质风险向量 + manifest.json）

---

## portfolio — 组合优化与归因（M4）

### `fz portfolio build`

**用途**：以 alpha 信号 + 风险模型为输入，用 cvxpy（CLARABEL solver）求解 mean-variance 二次规划，生成目标权重，并输出 Brinson + 风险因子归因报告。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--start` | 优化起始日 |
| `--end` | 优化结束日 |
| `--alpha-file` | alpha 信号文件（parquet 或 csv，列为股票代码，行为日期） |
| `--lam` | 风险厌恶系数，默认 1.0 |
| `--w-max` | 单票上限权重，默认 0.05 |
| `--turnover` | 双边换手上限（小数），默认 0.3 |
| `--industry-neutral` | 启用行业中性约束（相对等权基准） |

```bash
pixi run fz portfolio build \
  --start 20200101 --end 20241231 \
  --alpha-file workspace/discovery/session_001/alpha.parquet \
  --lam 1.0 --w-max 0.05 --turnover 0.3 \
  --industry-neutral
```

**产物**：`workspace/portfolios/{run_id}/`（每日权重 parquet + 归因报告 HTML + manifest.json）

> **MVP 限制**：`--industry-neutral` 行业中性约束目前以行业等权为基准（不是市值加权基准）。

---

## sim — 模拟交易（M7）

### `fz sim run`

**用途**：对每日目标权重执行多周期净值回测，扣除换手成本后输出净值曲线、年化收益、夏普比率、最大回撤。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--portfolio-dir` | 权重文件目录（`fz portfolio build` 产物） |
| `--start` | 模拟开始日 |
| `--end` | 模拟结束日 |
| `--cost-rate` | 单边换手成本率（bps），默认 15 |

```bash
pixi run fz sim run \
  --portfolio-dir workspace/portfolios/20240601_120000 \
  --start 20200101 --end 20241231
```

**产物**：`workspace/sim/{sim_id}/`（净值序列 parquet + 绩效摘要 JSON + manifest.json）

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

**用途**：为指定因子重建 Tear Sheet（HTML）。

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

**产物**：`workspace/factor_evaluations/{run_id}/tearsheet.html`

---

### `fz report portfolio`

**用途**：生成组合绩效 HTML Dashboard——指标卡 + 净值曲线 + 月度收益热图 + 归因 + 风险摘要。

**关键参数**：

| 参数 | 说明 |
|------|------|
| `--sim-dir` | 模拟结果目录 |
| `--portfolio-dir` | 权重/归因目录 |
| `--out` | 输出 HTML 路径（可选，默认写入 sim-dir） |

```bash
pixi run fz report portfolio \
  --sim-dir workspace/sim/20240601_130000 \
  --portfolio-dir workspace/portfolios/20240601_120000
```

**产物**：`workspace/sim/{sim_id}/portfolio_report.html`（或 `--out` 指定路径）

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

**产物**：`data/tushare/daily/YYYYMMDD.parquet`（分区缓存，增量更新）

---

### `fz data fetch daily-basic`

**用途**：从 Tushare 拉取日基础数据（市值、换手率、PE/PB 等），缓存到本地。

```bash
pixi run fz data fetch daily-basic --start 20200101 --end 20241231
```

**产物**：`data/tushare/daily_basic/YYYYMMDD.parquet`

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
  normalizer: rank_normal   # zscore | rank_normal
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
pixi run test         # pytest（1109 测试）
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

## 兼容入口

`pixi run daily`、`pixi run report`、`fz factor test`、`fz report open` 仍保留为兼容入口。新增脚本与文档优先使用 `fz factor run`、`fz report build`、`fz report path`。
