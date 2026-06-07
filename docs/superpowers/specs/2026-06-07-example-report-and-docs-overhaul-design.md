# 示例报告接入 + 文档整体优化 设计

> 日期：2026-06-07 · 状态：已确认设计，待写实现计划

## 目标

1. 把已生成的 tear sheet 作为「示例报告」提交进 `docs/`，使 clone 仓库的人无需自行跑评估即可直接打开查看。
2. 对所有面向用户的文档做一次**纠错 + 重构润色**，使其与当前代码现实一致，并把示例因子 / 报告织入文档作为教学材料。

## 已确认决策

| 决策点 | 选择 |
|--------|------|
| 示例托管 | 复制进 `docs/examples/` 并提交（git 增加约 1.1MB） |
| 优化深度 | 深度：纠错 + 重构润色 + 把示例写进作者手册 |
| LLM 口径 | 据实描述当前行为，**不改代码** |

## 背景事实（核实于当前工作树）

- `WalkForwardConfig.enabled` 默认 `False`（`src/factorzen/core/config_loader.py:131`）；`daily_factor_template.yaml` 与 `volume_return_corr_20d.yaml` 的 `walk_forward.enabled` 均为 `false`。→ **walk-forward 现已默认关闭。**
- LLM streamline 计划（`docs/superpowers/plans/2026-06-07-streamline-llm-report-evaluation.md`）**尚未落地**：`schema.py` 仍是 7 字段判断结构（`rating`/`risk_flags`/`usage_suggestion` 等），`PROMPT_VERSION = "v1"`，`daily_single.py` 与 `_report_config.py` 仍对 `--all` / 无 YAML 强制 `llm_explain = True`。→ **当前现实：无 YAML / `--all` 默认会尝试 LLM 解读，缺配置自动跳过。**
- 示例报告产物：`workspace/factor_evaluations/volume_return_corr_20d_20260607_161021/`，由当前代码生成（`git_dirty=true`），report.html 内仍呈现旧版 LLM 判断（「LLM 综合结论（weak/low）」「LLM 使用建议」等），与当前代码一致。
- 示例因子 `volume_return_corr_20d`：20 日「1 日收益 × 对数成交量」滚动 Pearson 相关。关键指标 IC≈-0.016、t≈-11.5、IR≈-0.23、换手≈48%，统计显著但经济意义弱——是一个诚实的「弱因子」案例。
- `.gitignore` 忽略 `workspace/factor_evaluations/*`，但**不**忽略 `docs/examples/`（已 `git check-ignore` 确认）。
- 新建但未被文档引用的模板：`workspace/factors/{daily,weekly,monthly,intraday}/TEMPLATE.md`。

## A. 示例报告接入

新增文件：

```text
docs/examples/
├── README.md                                示例索引 + 报告走查页
└── volume_return_corr_20d-tear-sheet.html   report.html 副本（约 1.1MB，自包含）
```

`docs/examples/README.md` 内容：

- **这是什么**：FactorZen 对示例因子 `volume_return_corr_20d` 的真实 tear sheet 快照。
- **示例因子**：20 日「1 日收益 × 对数成交量」滚动 Pearson 相关，实现见 `workspace/factors/daily/volume_return_corr_20d.py`。
- **运行配置**：csi500、20160606–20260606、benchmark `000905.SH`、`seed=42`、industry+size 中性化、zscore、内置 4 策略套件（topn_50 / quantile_ls_5 / factor_weighted_ls / optimizer_mv_long_only）、walk-forward 关闭、IC both、neutralized IC、event study、LLM 解读。
- **诚实结论**：IC≈-0.016、t≈-11.5、IR≈-0.23、换手≈48%、回测夏普≈0.08 → 统计显著但经济意义弱，典型「弱因子」。借此呼应设计原则 #3「报告必须暴露问题」。
- **报告走查**：逐段解读 tear sheet（评分卡 / 概览 / IC / 分层回测 / 多空 NAV / 月度收益 / 换手成本 / 限制说明 / 复现摘要与模块状态 / 大模型补充解读），告诉读者每块该看什么。
- **如何复现**：

  ```bash
  pixi run fz factor run --config workspace/configs/daily/volume_return_corr_20d.yaml
  pixi run fz report path <run_id>
  ```

- **查看提示**：GitHub 不内联渲染 HTML，需下载后用浏览器打开，或用 htmlpreview 类工具。该 HTML 为快照；本地实跑产物落在 `workspace/factor_evaluations/{run_id}/`（默认不入库）。

## B. 文档纠错（与现实对齐）

### B1. Walk-forward 现已默认关闭

- `README.md`：「无 YAML 默认」特性清单移除 walk-forward；补一句「walk-forward 默认关闭，按需用 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启」。
- `docs/runbook.md`：同上修正「因子工作流」中的默认清单；YAML 段补 `walk_forward.enabled` 字段说明。
- `docs/project-explanation.md`：§5 配置字段补 `walk_forward.enabled`；§8 评估清单把 walk-forward 标为「按需开启」。
- `docs/architecture.md` 数据流、`docs/evolution-plan-2026.md` 链路图：把 walk-forward 标为可选 / 默认关闭（不删除其作为能力的描述）。

### B2. LLM 口径统一（据实）

- `docs/runbook.md`：第 31 行「LLM 默认关闭」与第 50 行「默认含 LLM」自相矛盾 → 统一为现实：**无 YAML / `--all` 默认会尝试 LLM 解读，缺 `FACTORZEN_LLM_*` 配置自动跳过；显式 `--llm-explain` 对自定义配置也启用。**
- `README.md` 第 64/82 行、`docs/project-explanation.md` llm 描述：措辞与 runbook 对齐，避免发散。

## C. 文档重构润色（深度）

1. `docs/factor-authoring.md`：在现有 `reversal_5d` 最小示例之后，新增 `volume_return_corr_20d` 作为**进阶 worked example**（多中间列、滚动相关、`min_samples`、方差非正守卫），展示实现 + YAML + 运行命令，并链到 `docs/examples/` 示例报告，形成「写因子 → 跑评估 → 读报告」闭环。
2. `docs/factor-authoring.md`：在「创建模板」一节引用新建的 `workspace/factors/{daily,weekly,monthly,intraday}/TEMPLATE.md`（说明 `fz factor new` 与手写模板的关系）。
3. 交叉链接 / 面包屑 / 术语一致性 pass：
   - `docs/README.md` 文档地图、`README.md` 文档表新增「示例报告」行，指向 `docs/examples/`。
   - 各文档顶部面包屑统一加入「示例报告」入口；统一术语（tear sheet / 报告、walk-forward 中文措辞）。

## D. CHANGELOG

`CHANGELOG.md` 的 `## [Unreleased]` 增补：

- **Changed**：walk-forward 评估改为默认关闭（opt-in，YAML `walk_forward.enabled` 或 `--set`）。
- **Added**：示例报告 `docs/examples/` + 示例因子 `volume_return_corr_20d` + factor-authoring 进阶示例。
- **Fixed**：文档与现状对齐（walk-forward 过时描述、runbook LLM 自相矛盾）。

## 不做（YAGNI）

- **不**实现 streamline-llm 计划（schema 收窄 / PROMPT v2 / `--llm-explain` opt-in）——属代码变更，本次仅文档。
- 不回写 `docs/release-notes/`（已发布历史不回写）。
- 不整体重写 `architecture.md` / `project-explanation.md` 结构，仅局部纠错 + 一致性。

## 验收标准

- `docs/examples/` 含可直接打开的示例报告 HTML 与走查 README，且被 `README.md` / `docs/README.md` 链接。
- 全部文档对 walk-forward 默认状态描述一致（默认关闭、opt-in）。
- runbook 内部不再自相矛盾，LLM 默认行为描述与当前代码一致。
- factor-authoring 含 `volume_return_corr_20d` 进阶示例并引用 TEMPLATE.md。
- CHANGELOG `[Unreleased]` 记录上述用户可见变更。
- 不产生任何代码 / 测试改动；提交身份为 `rookiewu417 <1007372080@qq.com>`。
