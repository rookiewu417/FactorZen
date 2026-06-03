# 企业级日频因子评估平台 升级计划

> **For agentic workers:** 本计划以 TDD / 频繁提交方式逐任务执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 把 FactorZen 日频评估与报告链路升级为可信、可复现、CI 守护的企业级平台:先让 master CI 转绿,再补齐工程化与治理短板。

**Architecture:** 分阶段推进。Phase 0 先消除 CI 红灯(类型错误 + 冻结模块的测试),Phase 1 加固工程化(pre-commit/CI/治理文件/编码),Phase 2+ 为路线图(可观测性、报告模块解耦、复现性强化)。

**Tech Stack:** Python 3.10–3.12 · pixi · polars · ruff · mypy · pytest · GitHub Actions · Jinja2 报告。

---

## 审计结论(2026-06-03)

事实采集结果(本地 + GitHub REST API,仓库为 private):

| 维度 | 状态 | 证据 |
|------|------|------|
| Lint (ruff) | ✅ 通过 | `ruff check .` All checks passed |
| **类型检查 (mypy)** | ❌ **10 错 / 3 文件** | `tear_sheet.py`、`backtest.py`、`generate_report.py` |
| 测试 (pytest) | 658 passed / **1 failed** / 2 skipped | 失败为 `test_intraday_template_explains_missing_daily_ic_chart_with_table`(intraday WIP) |
| **GitHub CI (master c98e1c3)** | ❌ **failure** | `Type check` 步骤失败 → `Test`/`Coverage` 被跳过 |
| pre-commit mypy hook | ⚠️ 失效 | `files: ^(common\|daily/evaluation\|...)` 指向 reorg 前旧路径,现匹配为空 |
| CI 矩阵 | ⚠️ 仅 3.11 | pyproject 声明支持 3.10–3.12,未矩阵化;无 coverage 门槛/并发控制 |
| 编码 | ⚠️ 乱码 | README.md 等 6 文件 GBK→UTF-8 双重编码乱码 |
| 治理文件 | ⚠️ 缺失 | LICENSE / CONTRIBUTING / CHANGELOG / SECURITY / PR 模板 全缺 |
| 安全 | ⚠️ | git remote URL 内嵌 PAT;PAT 在 CLAUDE.md 明文(建议轮换) |
| 报告模块 | ⚠️ 巨石 | `tear_sheet.py` 2986 行单文件(后续解耦) |

**根因链:** 最近的 "consolidate package layout" 合并把代码搬到 `src/factorzen/`,但 (a) 引入/暴露了 10 处类型不安全点,(b) pre-commit 的 mypy 路径未同步,(c) 部分中文文档在搬运中被双重编码。CI 的 typecheck 因此变红,连带测试都没跑。

---

## Phase 0 — 让 master CI 转绿(最高优先,日频/报告范围内)

### Task 0.1: 修复 `tear_sheet.py` 4 处类型错误

**Files:** Modify `src/factorzen/reports/tear_sheet.py`(行 132 / 483 / 1949 / 1953)

- [ ] **132** `label.set_ha("center")` → `label.set_horizontalalignment("center")`(matplotlib 别名,stub 不识别 `set_ha`)
- [ ] **483** `_make_attribution_chart` 调 `_prepare_brinson_plot_frame(sector_df)` 前 `sector_df` 为 `Any|None`;`has_brinson_plot` 已保证非空,加 `assert sector_df is not None`
- [ ] **1949/1953** `top_n.group(1) if hasattr(top_n, "group")`、`quantiles.group(1) if hasattr(quantiles, "group")` → 用 `isinstance(x, re.Match)` 收窄
- [ ] 验证:`pixi run mypy ... src/factorzen/reports/tear_sheet.py` 该文件 0 错

### Task 0.2: 修复 `backtest.py` 2 处类型错误

**Files:** Modify `src/factorzen/daily/evaluation/backtest.py:498-505`

- [ ] `has_signal` bool 无法让 mypy 收窄 `signal_date`;改为在 `if has_signal:` 块内 `assert signal_date is not None`(语义不变,块内 `has_signal` 已蕴含非空)
- [ ] 验证:该文件 mypy 0 错

### Task 0.3: 修复 `generate_report.py` 3 处类型错误

**Files:** Modify `src/factorzen/pipelines/generate_report.py`(行 568 / 582 / 709 + `_apply_backtest_direction` 签名)

- [ ] `compute_ic(method="both")` 返回 `BothIcResult`(TypedDict),`["pearson"]` 取 `IcStats`;用 `isinstance(both, dict)` 收窄后取值,使 `pearson_ic_result` 静态为 `IcStats`(具 `ic_mean`/`ir`)
- [ ] `_apply_backtest_direction(clean_df, decision)` 签名 `decision: dict[str, Any]` → `dict[str, Any] | None`,首行 `if not decision or decision.get("direction") != "reversed": return clean_df`
- [ ] 验证:`pixi run typecheck` 全仓 0 错

### Task 0.4: 处理 intraday WIP 测试(尊重"intraday 冻结 + 日频聚焦")

**Files:** Modify `tests/test_reporting.py`(`test_intraday_template_explains_missing_daily_ic_chart_with_table`)

- [ ] 该测试断言 intraday 报告附录应有"研究边界"模块状态行(尚未实现的冻结模块 UI)。按 evolution-plan "intraday 冻结" 决策,加
  `@pytest.mark.xfail(reason="intraday 报告 UI 冻结(见 docs/evolution-plan-2026);研究边界模块状态行未实现", strict=False)`
- [ ] 验证:`pixi run test` → 0 failed(该用例计为 xfailed)

### Task 0.5: 本地全绿 + 提交

- [ ] `pixi run lint && pixi run typecheck && pixi run test` 全绿
- [ ] commit

---

## Phase 1 — 工程化与治理加固(企业级)

### Task 1.1: 修复 pre-commit mypy 路径
- [ ] `.pre-commit-config.yaml`:`files: ^(common|daily/evaluation|daily/preprocessing)/` → `^src/factorzen/`;`additional_dependencies: [types-all]`(已废弃易坏)→ 移除或换具体 stub;ruff rev 升到与 pyproject 一致

### Task 1.2: CI 加固
- [ ] `.github/workflows/ci.yml`:Python 矩阵 `["3.10","3.11","3.12"]`;加 `concurrency`(取消同分支旧跑);coverage 设最低门槛(`--fail-under`)并上传产物;保留 typecheck 门槛

### Task 1.3: 企业治理文件
- [ ] 新增 `LICENSE`(与用户确认许可;缺省 MIT 占位并标注)、`CONTRIBUTING.md`、`CHANGELOG.md`(Keep a Changelog)、`SECURITY.md`、`.github/PULL_REQUEST_TEMPLATE.md`

### Task 1.4: 修复中文文档编码
- [ ] 对 README.md / docs/architecture.md / docs/evolution-plan-2026.md / docs/project-explanation.md / docs/runbook.md / clean-legacy-directories.md 执行 `bytes.decode('utf-8').encode('latin1').decode('gbk')` 反演,逐个核验后重写为正确 UTF-8

### Task 1.5: 安全
- [ ] git remote 去除内嵌 PAT(改用凭据助手/env);在 SECURITY.md 与交付说明中提示**轮换该 PAT**(已多处明文暴露)

---

## Phase 2 — 报告模块解耦(进行中)

`tear_sheet.py`(2986 行巨石)按职责逐步拆分,全程保持测试绿:

- [x] `reports/_formatting.py` —— 格式化/数值/安全取值工具(已抽取)
- [x] `reports/_scoring.py` —— 评级评分卡 `FactorRating` + `_score_*` + `_compute_factor_rating`(已抽取)
- [x] `reports/_charts.py` —— 全部 `_make_*_chart` + matplotlib 设置 + 图表辅助(已抽取,705 行)
- [ ] `reports/_summaries.py` —— `_build_*_summary` / `_build_*_notice` / `_display_*`(后续;函数分散,需多段抽取)
- [ ] `reports/_strategy.py` —— 策略命名/口径/约束/交易摘要(后续)

**当前:** `tear_sheet.py` 2986 → 2050 行(-31%);4 个聚焦模块;658 passed / 2 skipped / 1 xfailed,mypy/ruff 全绿。`tear_sheet.py` 经 re-export 保持对外导入接口不变。

## Phase 3+ — 路线图(后续)

- **报告模块解耦收尾:** 抽取 summaries / strategy 两个模块。
- **可复现性强化:** manifest 记录 git SHA / pixi.lock hash / dirty 状态;`workspace/factor_evaluations/index.jsonl` run 索引(对齐 evolution-plan Phase 2)。
- **可观测性:** 结构化日志、运行耗时/数据覆盖率指标落盘。
- **数据契约:** 用 pydantic/参数化校验固化评估输入 schema,异常早失败。

---

## 验收命令

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
```

## 进度记录
- 2026-06-03:完成审计,产出本计划。开始执行 Phase 0。
- 2026-06-03:**Phase 0 完成** —— 修复 10 处 mypy 错误(tear_sheet/backtest/generate_report),xfail 冻结的 intraday WIP 测试。本地 `lint ✅ / typecheck ✅(87 文件 0 错)/ test ✅(658 passed, 2 skipped, 1 xfailed)`。提交 5baa4dd、352394e。
- 2026-06-03:**Phase 1 完成** —— 修复 README 及 docs 的 GBK 乱码(6d1d195);pre-commit 改为 pixi local hooks 并修死路径,CI 加最小权限+并发取消,新增 CONTRIBUTING/SECURITY/CHANGELOG/PR 模板(4fbe69a)。
- 待办:推送分支 + 开 PR 验证 CI 转绿;Phase 2+(报告模块解耦、可复现性强化)列入后续。
