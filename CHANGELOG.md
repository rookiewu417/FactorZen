# Changelog

本文件记录值得注意的变更,遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed
- **报告引擎:** 事件研究 `ci_95=None` 时模板对 `None` 下标取值导致整份报告崩溃,补齐 `avg_cumret`/`ci_95` 取值守卫。
- **报告语义:** 统一多空判定为单一 `_resolve_is_long_short`,修复 `factor_weighted + long_only=True` 在概览(判多头)与策略分页(判多空)之间自相矛盾。
- **类型安全:** 修复 `tear_sheet.py` / `backtest.py` / `generate_report.py` 共 10 处 mypy 错误,恢复 CI `typecheck` 步骤(此前失败导致测试被跳过)。
- **健壮性:** 图表辅助函数单列输入的 `StopIteration` 守卫;分位收益除零防护;事件研究窗口序列校验由 `>=` 收紧为 `==`。
- **文档编码:** 修复 README 及 docs 在合并中被 GBK 双重编码损坏的中文(重写 README/architecture/runbook,恢复 evolution-plan/project-explanation)。
- **文档刷新:** `project-explanation.md` 由 1374 行陈旧版(合并前布局)改写为 ~130 行准确版,反映 `src/factorzen` 当前布局并补入可复现/可观测/数据契约等新能力。

### Changed
- **报告模块解耦:** `tear_sheet.py` 2986 → 1054 行(-65%),按职责拆为 `_formatting`/`_scoring`/`_charts`/`_strategy`/`_summaries` 五个模块;经 re-export 保持对外导入接口不变。
- **工程化:** `.pre-commit-config.yaml` 改为通过 `pixi run` 的 local hooks,保证 pre-commit / CI / 本地三者版本一致(修复 mypy hook 指向已删除旧路径的问题)。
- **CI:** 增加 `permissions: contents: read` 最小权限与 `concurrency` 取消重复运行;`tools/run_coverage.py` 增加 `--fail-under=73` 覆盖率门槛(防回退)。
- **可复现性:** `run_experiment` 在工作树 dirty 时 `logger.warning` 提示;manifest 增记 `duration_seconds`。

### Added
- **测试加固:** 新增 `test_charts_helpers` / `test_summaries_helpers` 共 29 个边界单测,覆盖报告模块的 None/空输入防御分支(`_charts` 78%→85%,`_summaries` 78%→81%);覆盖率门槛 73%→74%。
- **开源:** 以 MIT License 开源(`LICENSE`);pyproject 增加 `license`、`readme` 与分类器元数据;README 增加许可说明。仓库当前文件与全部 git 历史经扫描确认无凭据泄露。
- **数据契约:** `core/validation.py::require_columns` 列契约校验;`compute_fwd_returns`、`compute_turnover` 入口及 `backtest._prepare_factor_df`/`_prepare_price_df` 对必需列做 fail-fast 校验,畸形输入给出清晰错误(列出缺失列与实际列)。
- **可观测性:** `core/timing.py::StageTimer` 按阶段计时(INFO 日志 + 累计);`generate_report` 与 `daily_single` 两条日频主管线均对 IC/回测/换手/报告四阶段计时并把 `stage_timings` 写入 manifest;新增 `record_experiment_metadata` 并修复 `run_experiment` finally 丢失运行期元数据的问题。
- 企业治理文件:`CONTRIBUTING.md`、`SECURITY.md`、`CHANGELOG.md`、`.github/PULL_REQUEST_TEMPLATE.md`。
- 升级计划:`docs/superpowers/plans/2026-06-03-enterprise-grade-daily-platform.md`。

## [0.2.0]
见 [docs/release-notes/v0.2.0.md](docs/release-notes/v0.2.0.md)。

## [0.1.0]
见 [docs/release-notes/v0.1.0.md](docs/release-notes/v0.1.0.md)。
