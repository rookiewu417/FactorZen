# Changelog

本文件记录值得注意的变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed

- **文档对齐：** 修正 README / runbook 把 walk-forward 误述为「无 YAML 默认开启」的过时描述；消除 runbook 中 LLM 默认行为自相矛盾的两处表述，统一为与当前代码一致的口径。
- **报告引擎：** 事件研究 `ci_95=None` 时模板对 `None` 下标取值导致整份报告崩溃，补齐 `avg_cumret` / `ci_95` 取值守卫。
- **报告语义：** 统一多空判定为单一 `_resolve_is_long_short`，修复 `factor_weighted + long_only=True` 在概览（判多头）与策略分页（判多空）之间自相矛盾。
- **类型安全：** 修复 `tear_sheet.py` / `backtest.py` / `generate_report.py` 共 10 处 mypy 错误，恢复 CI `typecheck` 步骤（此前失败导致测试被跳过）。
- **健壮性：** 图表辅助函数单列输入的 `StopIteration` 守卫；分位收益除零防护；事件研究窗口序列校验由 `>=` 收紧为 `==`。
- **文档编码：** 修复 README 及 docs 在合并中被 GBK 双重编码损坏的中文（重写 README / architecture / runbook，恢复 evolution-plan / project-explanation）。
- **文档刷新：** `project-explanation.md` 由 1374 行陈旧版（合并前布局）改写为 ~130 行准确版，反映 `src/factorzen` 当前布局并补入可复现 / 可观测 / 数据契约等新能力。
- **文档死链：** 修复 `runbook.md` 与 `docs/README.md` 指向 qlib README 的死链（`workspace/factors/qlib/` → `src/factorzen/builtin_factors/qlib/`）；澄清 `runbook.md` 中 LLM 解读默认行为（默认关闭，无 YAML 默认配置与 `--all` 模式自动启用）。

### Changed

- **Walk-forward：** 策略 walk-forward 样本外评估改为**默认关闭**（`WalkForwardConfig.enabled` 默认 `false`），按需通过 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启。
- **报告模块解耦：** `tear_sheet.py` 2986 → 1054 行（-65%），按职责拆为 `_formatting` / `_scoring` / `_charts` / `_strategy` / `_summaries` 五个模块；经 re-export 保持对外导入接口不变。
- **工程化：** `.pre-commit-config.yaml` 改为通过 `pixi run` 的 local hooks，保证 pre-commit / CI / 本地三者版本一致（修复 mypy hook 指向已删除旧路径的问题）。
- **CI：** 增加 `permissions: contents: read` 最小权限与 `concurrency` 取消重复运行；`tools/run_coverage.py` 增加 `--fail-under=73` 覆盖率门槛（防回退）。
- **可复现性：** `run_experiment` 在工作树 dirty 时 `logger.warning` 提示；manifest 增记 `duration_seconds`。
- **锁文件 / 覆盖率：** `pixi.lock` 升级 v6 → v7（改善可复现性）；`tools/run_coverage.py` 基线说明 76% → 82%（实测总覆盖率，门槛仍为 74%）。

### Added

- **示例报告：** 新增示例因子 `volume_return_corr_20d` 的真实 tear sheet（`https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html`）与分区导读 README。
- **示例因子：** 新增 `workspace/factors/daily/volume_return_corr_20d.py`（20 日量价滚动相关）及其配置，并在 factor-authoring 中作为进阶 worked example。
- **因子模板：** 各频率目录新增 `TEMPLATE.md` 手写模板，并在 factor-authoring 中引用。
- **测试加固：** 新增 `test_charts_helpers` / `test_summaries_helpers` 共 29 个边界单测，覆盖报告模块的 None / 空输入防御分支（`_charts` 78%→85%，`_summaries` 78%→81%）；覆盖率门槛 73%→74%。
- **开源：** 以 MIT License 开源（`LICENSE`）；pyproject 增加 `license`、`readme` 与分类器元数据；README 增加许可说明。仓库当前文件与全部 git 历史经扫描确认无凭据泄露。
- **数据契约：** `core/validation.py::require_columns` 列契约校验；`compute_fwd_returns`、`compute_turnover` 入口及 `backtest._prepare_factor_df` / `_prepare_price_df` 对必需列做 fail-fast 校验，畸形输入给出清晰错误（列出缺失列与实际列）。
- **可观测性：** `core/timing.py::StageTimer` 按阶段计时（INFO 日志 + 累计）；`generate_report` 与 `daily_single` 两条日频主管线均对 IC / 回测 / 换手 / 报告四阶段计时并把 `stage_timings` 写入 manifest；新增 `record_experiment_metadata` 并修复 `run_experiment` finally 丢失运行期元数据的问题。
- **企业治理文件：** `CONTRIBUTING.md`、`SECURITY.md`、`CHANGELOG.md`、`.github/PULL_REQUEST_TEMPLATE.md`。
- **升级计划：** `docs/evolution-plan-2026.md`。

## [0.3.0]

详见 [docs/release-notes/v0.3.0.md](docs/release-notes/v0.3.0.md)。自 v0.2.0 起，项目从「A 股低频单因子研究框架」扩展为端到端、可复现的买方研究平台。测试套件扩展到 1109 个离线可重复用例。

### Added

- **因子挖掘引擎：** `discovery/` 算子库（30+ 时序/截面/算术算子）+ 表达式 AST 双向序列化 + 随机/遗传搜索（`fz mine search`，`--trials` 默认 200）+ IC 打分去相关；新增 `fz mine export-alpha` 把单个候选算成 `(ts_code, alpha)` 单截面 parquet，衔接组合优化。
- **防过拟合护栏：** `validation/` block bootstrap IC 置信区间、Deflated Sharpe Ratio、PBO/CSCV（候选池）、holdout 段永久隔离；`fz validate overfit` 打印单因子 IC/IR/DSR/bootstrap CI（不落盘，N=1 不计 PBO）。
- **Barra 风险模型：** `risk/` 8 个风格因子（size/value/momentum/volatility/liquidity/quality/growth/leverage）+ 中信一级行业因子 + Newey-West 协方差 + James-Stein 特质风险收缩 + 边际风险贡献；`fz risk build`（默认 `--cov-half-life 90 --nw-lags 2 --spec-shrinkage 0.3 --spec-half-life 90`）。
- **组合优化与归因：** `portfolio/` cvxpy mean-variance QP（CLARABEL solver，box/预算/换手/行业中性约束，单截面建仓）+ `attribution/` Brinson 多期归因与风险因子归因；`fz portfolio build`。
- **单/多 Agent 挖掘：** `agents/` 零外部依赖自建 LLM 闭环（假设→生成→护栏→IC 验证→反思）+ Negative RAG 失败注入（`fz mine agent`）；4 个角色 Agent（Hypothesis/Coder/Critic/Librarian）+ Evaluator 评估环节 + 跨 session 长期记忆（`fz mine team`）。
- **模拟交易 + 成果展示：** `sim/engine.py` 多周期权重回测（对齐行情、扣换手成本、净值与绩效指标）+ `reports/portfolio_report.py` 组合绩效 HTML Dashboard（指标卡 + 净值曲线 + 月度热图 + 归因 + 风险摘要）；`fz sim run` / `fz sim show` / `fz report portfolio`。
- **微观结构与交易约束：** `core/universe.py` universe 快照（停牌/涨跌停/ST/次新股/流通市值过滤）+ `core/benchmark.py` 基准管理（HS300/ZZ500/ZZ1000 + 行业等权替代基准）+ 回测引擎 GEM 双路径容差、T+1、`signal_date` 解耦、ADV 零值 fallback。
- **端到端教程：** `docs/end-to-end-tutorial.md` 从拉数据到组合展示的完整链路逐步教程。

### Changed

- **定位升级：** README / project-explanation / architecture / evolution-plan 由单因子框架口径改写为端到端买方研究平台口径。
- **依赖：** 新增 cvxpy（CLARABEL solver）强依赖，组合优化功能依赖此包。

## [0.2.0]

见 [docs/release-notes/v0.2.0.md](docs/release-notes/v0.2.0.md)。

## [0.1.0]

见 [docs/release-notes/v0.1.0.md](docs/release-notes/v0.1.0.md)。
