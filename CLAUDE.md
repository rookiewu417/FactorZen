# CLAUDE.md — FactorZen 项目指令

> 给在本仓库工作的 Claude 的 onboarding。全局规则见 `~/.claude/CLAUDE.md`（git 身份 `rookiewu417 <1007372080@qq.com>`、pixi 环境、中文回复等），本文件只补 **FactorZen 特定** 的内容。

## 这是什么

FactorZen — 端到端、可复现的 A 股量化研究平台。完整链路:**因子挖掘 → 防过拟合 → Barra 风险模型 → AI Agent 挖掘 → 组合优化与归因 → 模拟交易 → 成果展示**。研究可信度优先(PIT 无未来函数 / 样本外+防过拟合护栏 / 一切产物落 manifest 可复现)。

## 环境与命令

全部经 pixi(无全局 python):

```bash
pixi run fz <command>      # CLI 入口 = python -m factorzen.cli.main
pixi run test              # pytest tests/ -v (约 1116 测试)
pixi run lint              # ruff check .      ← 全仓库(含 tests/)!
pixi run typecheck         # mypy(范围 = src/factorzen 整包)
pixi run format            # ruff format .
pixi run coverage          # tools/run_coverage.py
```

跑单个测试:`pixi run -- pytest tests/test_xxx.py::test_name -v`。

**⚠️ 提交前必跑 `pixi run lint` 和 `pixi run typecheck`,它们扫的都不止你改的那个模块:**
- `lint` = `ruff check .`,扫**整个仓库**(连 `tests/` 都算)。
- `typecheck` = `mypy`,范围由 `pyproject.toml` 的 `[tool.mypy] files = ["src/factorzen"]` 固定,扫**整个 `src/factorzen`**(不含 tests)。

本项目 CI(`.github/workflows/ci.yml`)依次跑 Lint→Type check→Test→Coverage,任一红则 fail。历史教训:只 `ruff check src/factorzen/<改动模块>/` 会漏掉 `tests/` 的 lint、漏掉 `src/` 别处的 mypy 类型错,push 后 CI 才炸。**改完一定全仓库 lint + 全 src typecheck。**

## 架构地图(模块 → 能力)

> 平台按内部里程碑 M0-M7 迭代而成;**这套代号只用于内部定位,不要写进对外 README/文档**(外部读者不懂 M3/M4,对外按能力命名)。

| 内部 | 能力 | 模块路径 |
|---|---|---|
| M0 | 微观结构/交易约束(停牌/涨跌停/ST/次新/T+1)+ universe 快照 + 性能基准 | `core/`(universe/benchmark)、`daily/evaluation/backtest.py` |
| M1 | 因子挖掘(算子库 + 表达式 AST↔字符串 + 随机/遗传搜索 + 去相关 + export-alpha) | `discovery/` |
| M2 | 防过拟合(bootstrap IC CI + Deflated Sharpe + PBO/CSCV + holdout 隔离) | `validation/` |
| M3 | Barra 风险模型(风格+行业暴露 + Newey-West 协方差 + 特质风险 + MCR) | `risk/` |
| M4 | 组合优化(cvxpy 因子形式 mean-variance QP + 约束)+ 归因(Brinson + 风险因子) | `portfolio/`、`attribution/` |
| M5 | 单 Agent 挖掘(LLM 闭环,零依赖自建 loop) | `agents/` |
| M6 | 多 Agent 团队(4 角色 Agent + Evaluator 评估环节 + 跨轮否决 + 跨 session 记忆) | `agents/roles/`、`team_orchestrator.py`、`experiment_index.py` |
| M7 | 模拟交易(多周期净值回测)+ 成果展示页(HTML Dashboard) | `sim/`、`reports/portfolio_report.py` |
| 基线 | 单因子研究链路(IC/分层回测/walk-forward/Tear Sheet) | `daily/`、`reports/` |

**命名空间分离(易混,务必区分):** `portfolio/` 是「组合构建流」(用因子风险模型,因子形式 QP);`daily/optimization/` 是「单因子研究流」(全 Σ 矩阵,Tear Sheet 用)。两者接口/风险形式/归因方法都不同,**不要互相复用或合并**。注:`fz portfolio build` 内部自建 RiskModel,`fz risk build` 是独立诊断步骤、不被组合建仓消费。

## 数据与产物

- 数据源 Tushare(token 在 `.env` 的 `TUSHARE_TOKEN`);拉取缓存为本地 parquet。LLM 挖掘(M5/M6)需 `FACTORZEN_LLM_*`,缺失直接报错退出(非自动跳过;仅报告 LLM 解读这一可选功能缺失配置时才自动跳过)。
- 产物落 `workspace/{factor_evaluations,mining_sessions,risk_models,portfolios,sim,reports}/{run_id}/`,每个含 `manifest.json`(配置/命令/git_sha/seed)。
- 测试以 mock 离线为主(CI 无 token 也能跑);真实数据端到端 smoke 需 Tushare。

## 提交卫生

- 工作区常有**未跟踪的** `data/`、`docs/*升级计划*`、`docs/*讲解*` 等草稿——**提交时精确 `git add` 你改的文件,绝不 `git add -A`/`git add .`**,否则会误带草稿。
- 一个逻辑改动一个 commit,conventional commits,中文 message 可。
- **subagent 在同一工作区操作时,绝不让它们跑 `git clean`/`git checkout .`/`git stash`** —— 会误删别处未跟踪的工作(本项目曾因此丢过未提交的 CLAUDE.md)。

## 本项目反复踩的陷阱(改相关代码时警惕)

- **守恒/恒真断言**:`C` 由 `A`、`B` 构造时断言 `C=f(A,B)` 零判别力(如 specific_return = port_ret−Σfactor,再断言 Σfactor+specific=port_ret)。要用**独立公式/ground-truth/跨函数**验证。
- **同一逻辑双路径,修一处漏一处**:`backtest.py` 涨跌停判断有慢路径 `_apply_trade_constraints` + 快路径 `_run_precomputed_weights_backtest_fast`(模拟交易只走快路径)。改一处必同步另一处 + 两条路径都要有测试。
- **跨组件/跨 milestone 集成 gap**:单元测试各自绿 ≠ 拼起来能跑(如 M4 manifest 漏 `signal_date` → M7 sim 崩;mine search 产 candidates 不是 alpha 截面 → 需 export-alpha)。多组件工作收尾**必跑真实数据端到端 smoke**。
- **文档与代码漂移**:写文档/示例命令必须对照真实 `cli/main.py` parser 与 pipeline 落盘路径,别凭印象写(参数默认值/产物文件名/模块路径最容易失真)。
- **N/多重检验记账三角和**:迭代循环里对累积状态计数会 over-count,按轮过滤。
- **行业中性**:对真实 one-hot 行业列,绝对中性到 0 + long-only + Σw=1 必 infeasible;须中性到**基准暴露**(当前 MVP 用等权基准)。
- **polars 1.41**:`join(how="outer")` 已废,用 `how="full", coalesce=True`;`min_periods`→`min_samples`。

## 文档约定

- 对外文档(README/docs/)按**能力**组织,**不暴露** M0-M7 里程碑代号、SDD/subagent/spec-plan 等内部开发过程。`docs/superpowers/`(内部 spec/plan)已 gitignore、不入库、不对外链接。
- 内部记录(本文件、`.superpowers/sdd/progress.md` ledger、记忆库)可自由用 M0-M7 代号。
- 诚实标 MVP 限制(不接实盘 OMS、行业中性=等权基准、收益归因需持仓期收益),不夸大(如「4 角色 + Evaluator 环节」别写成「5 角色」)。
