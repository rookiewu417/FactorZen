# 示例报告接入 + 文档整体优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已生成的 tear sheet 作为示例报告提交进 `docs/examples/`，并对所有面向用户的文档做纠错 + 重构润色，使其与当前代码现实一致。

**Architecture:** 纯文档变更，不动任何 `src/` 代码或 `tests/`。新增 `docs/examples/`（报告 HTML 副本 + 走查 README），逐文件修正 walk-forward 默认状态与 LLM 口径，把示例因子织入 factor-authoring，最后做一次交叉链接 / 面包屑一致性 pass 并更新 CHANGELOG。

**Tech Stack:** Markdown、HTML（静态副本）、git。验证用 `grep` / `rg` 与人工核对，无单元测试。

**Spec:** `docs/superpowers/specs/2026-06-07-example-report-and-docs-overhaul-design.md`

---

## 前置：分支与身份

- [ ] **Step 0: 从 master 开分支**

当前在默认分支 `master`，开一个文档分支再改：

```bash
git checkout -b docs/example-report-and-overhaul
```

所有提交身份固定为 `rookiewu417 <1007372080@qq.com>`（每次 commit 用 `-c user.name=... -c user.email=...` 或仓库已配置该身份）。

## 文件结构

```text
docs/examples/                                    新增目录
├── README.md                                     新增：示例索引 + 报告走查
└── volume_return_corr_20d-tear-sheet.html        新增：report.html 副本（约 1.1MB）
README.md                                         修改：walk-forward/LLM 口径 + 文档表加示例报告
docs/README.md                                    修改：文档地图加示例报告
docs/runbook.md                                   修改：walk-forward 默认 + LLM 自相矛盾 + YAML 段
docs/project-explanation.md                       修改：§5 配置字段 + §8 walk-forward + llm 描述
docs/architecture.md                              修改：数据流 walk-forward 标可选
docs/evolution-plan-2026.md                       修改：链路图 walk-forward 标可选
docs/factor-authoring.md                          修改：进阶示例 + TEMPLATE.md 引用 + 面包屑
CHANGELOG.md                                       修改：[Unreleased] 增补
```

---

### Task 1: 接入示例报告文件

**Files:**
- Create: `docs/examples/volume_return_corr_20d-tear-sheet.html`
- Create: `docs/examples/README.md`

- [ ] **Step 1: 复制报告 HTML 到 docs/examples/**

```bash
mkdir -p docs/examples
cp "workspace/factor_evaluations/volume_return_corr_20d_20260607_161021/report.html" \
   docs/examples/volume_return_corr_20d-tear-sheet.html
```

- [ ] **Step 2: 写示例走查 README**

写入 `docs/examples/README.md`，完整内容：

```markdown
# 示例报告

> [FactorZen](../../README.md) · [文档](../README.md) · **示例报告**

本目录收录一份由 FactorZen 真实评估流程生成的 Tear Sheet，供你在不自己跑数据的情况下先看产出长什么样、怎么读。

## 文件

| 文件 | 说明 |
|------|------|
| [`volume_return_corr_20d-tear-sheet.html`](volume_return_corr_20d-tear-sheet.html) | 示例因子的完整 HTML 报告（自包含，约 1.1MB）|

> GitHub 不会内联渲染 HTML。请点开后下载用浏览器打开，或借助 htmlpreview 类工具在线预览。

## 示例因子

`volume_return_corr_20d` —— 20 日「1 日收益 × 对数成交量」滚动 Pearson 相关，衡量量价同步 / 背离程度。实现见 [`workspace/factors/daily/volume_return_corr_20d.py`](../../workspace/factors/daily/volume_return_corr_20d.py)。

## 运行配置

| 项 | 值 |
|----|----|
| universe | csi500 |
| 区间 | 2016-06-06 ~ 2026-06-06 |
| benchmark | 000905.SH |
| seed | 42 |
| 预处理 | MAD 去极值 · zscore 标准化 · industry+size 中性化 |
| 策略套件 | topn_50 · quantile_ls_5 · factor_weighted_ls · optimizer_mv_long_only |
| IC | both（Rank + Pearson）· 中性化 IC · event study |
| walk-forward | 关闭（默认）|
| LLM 解读 | 开启 |

复现命令：

\`\`\`bash
pixi run fz factor run --config workspace/configs/daily/volume_return_corr_20d.yaml
pixi run fz report path <run_id>
\`\`\`

实跑产物落在 `workspace/factor_evaluations/{run_id}/`（默认不入库）；本页 HTML 是其中一次运行的快照。

## 这是一个「弱因子」示例

示例特意选了一个预测能力不强的因子，用来展示 FactorZen 如何**诚实暴露问题**而非用漂亮图表掩盖（设计原则 #3）：

| 指标 | 值 | 解读 |
|------|----|----|
| IC 均值 | ≈ -0.016 | 方向稳定但绝对值很小 |
| IC t 统计 | ≈ -11.5 | 统计显著（长样本放大了显著性）|
| IR | ≈ -0.23 | 信息比率低 |
| 样本外 IC | ≈ -0.014 | 略有衰减 |
| 多空年化 | ≈ 0.3% | 经济意义弱 |
| 夏普 | ≈ 0.08 | 收益不稳定 |
| 换手率 | ≈ 48% | 偏高，成本侵蚀收益 |

结论：统计上存在、经济上微弱、交易上不划算——不建议单独使用。这正是报告该说清楚的事。

## 报告分区导读

报告按「结论 → 证据 → 限制 → 复现」组织，自上而下：

| 分区 | 看什么 |
|------|--------|
| 研究仪表盘 | 第一屏速览：评分、星级、关键指标与一句话结论 |
| 综合评估 | 规则引擎给出的结论、评级、证据强度、风险与下一步（确定性，不依赖 LLM）|
| 大模型研究解读 | 可选的 LLM 解读（本快照为当前版本，呈现评级 / 风险 / 建议等字段）|
| 跨策略对比 | 4 个策略并排对比收益、夏普、回撤、换手 |
| 收益表现 | 主策略 NAV、多空 NAV、月度收益与分位价差 |
| 预测能力 | Rank / Pearson IC、中性化 IC、多持有期一致性、HAC t 统计 |
| 结构检验 | 单调性、Rank 自相关、因子相关性、分组分层 |
| 交易可行性 | 换手、成本、容量约束 |
| 稳健性验证 | walk-forward 等样本外检验（本例 walk-forward 关闭，会标注为未运行）|
| 风险归因 | 市值 / 行业 / 市场状态分层归因 |
| 附录 | 复现摘要：配置、命令、git SHA、lockfile hash、阶段耗时与模块状态 |

读报告的顺序建议：先看仪表盘与综合评估拿结论，再用预测能力 / 交易可行性 / 稳健性核对证据，最后看附录确认可复现。
```

> 注：上面代码块里的 `\`\`\`bash ... \`\`\`` 在落盘时写成真实的三反引号围栏。

- [ ] **Step 3: 验证文件就位且不被 gitignore**

```bash
ls -la docs/examples/
git check-ignore docs/examples/volume_return_corr_20d-tear-sheet.html && echo "被忽略(异常)" || echo "可提交(OK)"
```

Expected: 两个文件存在；输出 `可提交(OK)`。

- [ ] **Step 4: 提交**

```bash
git add docs/examples/
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: 接入示例报告 volume_return_corr_20d tear sheet"
```

---

### Task 2: README.md 纠错 + 文档表加示例报告

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 修正无 YAML 默认描述（移除 walk-forward）**

`README.md` 第 82 行当前为：

```markdown
无 `--config` 时会使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、内置 4 策略套件、walk-forward、both IC、neutralized IC、event study 与 LLM 解读。
```

改为：

```markdown
无 `--config` 时会使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、内置 4 策略套件、both IC、neutralized IC、event study 与 LLM 解读（缺 `FACTORZEN_LLM_*` 配置时自动跳过）。walk-forward 默认关闭，按需用 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启。
```

- [ ] **Step 2: 统一安装段 LLM 措辞**

`README.md` 第 64 行当前为：

```markdown
`.env` 不入库。真实数据拉取需要配置 `TUSHARE_TOKEN`；无 YAML 默认运行会启用 LLM 研究解读，缺少 `FACTORZEN_LLM_*` 配置时自动跳过。
```

保持语义（与现实一致），仅微调为：

```markdown
`.env` 不入库。真实数据拉取需要配置 `TUSHARE_TOKEN`；无 YAML 默认运行会尝试 LLM 研究解读，缺少 `FACTORZEN_LLM_*` 配置时自动跳过。
```

- [ ] **Step 3: 文档表新增示例报告行**

在 `README.md` 文档表（约第 183–189 行）`[演进计划]` 行之后追加一行：

```markdown
| [示例报告](docs/examples/) | 真实 tear sheet 示例与分区导读 |
```

- [ ] **Step 4: 验证**

```bash
grep -n "walk-forward 默认关闭" README.md
grep -n "示例报告" README.md
grep -n "内置 4 策略套件、walk-forward、both" README.md && echo "旧描述残留(异常)" || echo "旧描述已清除(OK)"
```

Expected: 前两条有命中；第三条输出 `旧描述已清除(OK)`。

- [ ] **Step 5: 提交**

```bash
git add README.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: README 纠正 walk-forward 默认状态并接入示例报告链接"
```

---

### Task 3: docs/README.md 文档地图加示例报告

**Files:**
- Modify: `docs/README.md`

- [ ] **Step 1: 文档地图新增示例报告行**

在 `docs/README.md` 文档地图表（约第 9–17 行）`evolution-plan-2026` 行之后、`release-notes/` 行之前插入：

```markdown
| [examples/](examples/) | 新用户、复核者 | 真实 tear sheet 示例与分区导读 |
```

- [ ] **Step 2: 验证**

```bash
grep -n "examples/" docs/README.md
```

Expected: 命中新增行。

- [ ] **Step 3: 提交**

```bash
git add docs/README.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: 文档地图新增示例报告入口"
```

---

### Task 4: docs/runbook.md 纠错（walk-forward + LLM 自相矛盾 + YAML 段）

**Files:**
- Modify: `docs/runbook.md`

- [ ] **Step 1: 修正因子工作流默认清单（第 50 行）**

`docs/runbook.md` 第 50 行当前为：

```markdown
无 `--config` 时会使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、内置 4 策略套件、walk-forward、both IC、neutralized IC、event study 与 LLM 解读。缺少 `FACTORZEN_LLM_*` 配置时，LLM 解读会自动跳过。
```

改为：

```markdown
无 `--config` 时会使用内置研究级默认配置：`csi500`、匹配 benchmark、`seed=42`、行业+市值中性化、内置 4 策略套件、both IC、neutralized IC、event study 与 LLM 解读。缺少 `FACTORZEN_LLM_*` 配置时，LLM 解读会自动跳过。walk-forward 默认关闭，按需通过 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启。
```

- [ ] **Step 2: 修正环境自检段 LLM 自相矛盾（第 31 行）**

`docs/runbook.md` 第 31 行当前为：

```markdown
真实数据拉取需要在 `.env` 配置 `TUSHARE_TOKEN`。LLM 研究解读默认关闭，只有命令显式传入 `--llm-explain` 时才会尝试读取相关配置。
```

改为（与第 50 行及代码现实一致）：

```markdown
真实数据拉取需要在 `.env` 配置 `TUSHARE_TOKEN`。无 YAML 默认运行与 `--all` 会尝试 LLM 研究解读、缺少 `FACTORZEN_LLM_*` 配置时自动跳过；自定义配置下需显式传入 `--llm-explain` 才启用。
```

- [ ] **Step 3: YAML 配置段补 walk_forward.enabled 说明**

在 `docs/runbook.md`「YAML 配置」一节（约第 59–66 行）`config validate` 说明之后追加一段：

```markdown
walk-forward 样本外评估默认关闭，需要时在 YAML 打开：

\`\`\`yaml
walk_forward:
  enabled: true
  train_days: 504
  test_days: 252
  step_days: 252
  embargo_days: 5
\`\`\`

也可用 `--set walk_forward.enabled=true` 临时开启。
```

> 注：落盘时写成真实三反引号围栏。

- [ ] **Step 4: 验证**

```bash
grep -n "walk-forward 默认关闭" docs/runbook.md
grep -n "无 YAML 默认运行与 .--all. 会尝试 LLM" docs/runbook.md
grep -n "LLM 研究解读默认关闭" docs/runbook.md && echo "矛盾描述残留(异常)" || echo "矛盾已消除(OK)"
grep -n "内置 4 策略套件、walk-forward、both" docs/runbook.md && echo "旧清单残留(异常)" || echo "旧清单已清除(OK)"
```

Expected: 前两条命中；后两条分别输出 `矛盾已消除(OK)` 与 `旧清单已清除(OK)`。

- [ ] **Step 5: 提交**

```bash
git add docs/runbook.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: runbook 纠正 walk-forward 默认与 LLM 自相矛盾描述"
```

---

### Task 5: docs/project-explanation.md 纠错

**Files:**
- Modify: `docs/project-explanation.md`

- [ ] **Step 1: §5 配置字段补 walk_forward.enabled（第 91 行）**

`docs/project-explanation.md` 第 91 行当前为：

```markdown
配置样例在 `workspace/configs/daily/daily_factor_template.yaml`。常用字段包括 `factor`、`universe`、`start`、`end`、`benchmark`、`seed`、`preprocessing`、`backtest`、`walk_forward`、`ic_method`、`event_study` 与 `neutralized_ic`。
```

改为：

```markdown
配置样例在 `workspace/configs/daily/daily_factor_template.yaml`。常用字段包括 `factor`、`universe`、`start`、`end`、`benchmark`、`seed`、`preprocessing`、`backtest`、`walk_forward`、`ic_method`、`event_study` 与 `neutralized_ic`。其中 `walk_forward.enabled` 默认 `false`（样本外 walk-forward 按需开启）。
```

- [ ] **Step 2: §8 评估清单标注 walk-forward 默认关闭（第 125 行）**

`docs/project-explanation.md` 第 125 行当前为：

```markdown
- 单调性、Rank 自相关、因子相关性、市值/行业/市场状态分层、事件研究、walk-forward。
```

改为：

```markdown
- 单调性、Rank 自相关、因子相关性、市值/行业/市场状态分层、事件研究、walk-forward（默认关闭，按需开启）。
```

- [ ] **Step 3: 验证**

```bash
grep -n "walk_forward.enabled. 默认 .false" docs/project-explanation.md
grep -n "walk-forward（默认关闭" docs/project-explanation.md
```

Expected: 两条均命中。

- [ ] **Step 4: 提交**

```bash
git add docs/project-explanation.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: project-explanation 标注 walk-forward 默认关闭"
```

---

### Task 6: architecture.md + evolution-plan-2026.md walk-forward 标可选

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/evolution-plan-2026.md`

- [ ] **Step 1: architecture 数据流标注（第 49 行）**

`docs/architecture.md` 第 49 行当前为：

```text
前向收益 + IC / 回测 / 换手 / walk-forward / 归因 / 基准
```

改为：

```text
前向收益 + IC / 回测 / 换手 / walk-forward（默认关闭）/ 归因 / 基准
```

- [ ] **Step 2: evolution-plan 链路图标注（约第 27 行）**

`docs/evolution-plan-2026.md` 「当前定位」链路图中的：

```text
  → IC / 分层回测 / walk-forward
```

改为：

```text
  → IC / 分层回测 / walk-forward（默认关闭）
```

> 该链路图在 `evolution-plan-2026.md` 内仅「当前定位」一处出现，改它即可。

- [ ] **Step 3: 验证**

```bash
grep -n "walk-forward（默认关闭）" docs/architecture.md docs/evolution-plan-2026.md
```

Expected: 两个文件各至少一处命中。

- [ ] **Step 4: 提交**

```bash
git add docs/architecture.md docs/evolution-plan-2026.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: architecture/evolution 标注 walk-forward 默认关闭"
```

---

### Task 7: factor-authoring.md 进阶示例 + TEMPLATE.md 引用 + 面包屑

**Files:**
- Modify: `docs/factor-authoring.md`

- [ ] **Step 1: 「创建模板」一节引用 TEMPLATE.md**

在 `docs/factor-authoring.md` 第 1 节「创建模板」生成文件说明（约第 13–17 行）之后追加：

```markdown
每个频率目录下还放了一份手写模板，列出编写约定、可复制代码与检查点，可直接照着改：

- [`workspace/factors/daily/TEMPLATE.md`](../workspace/factors/daily/TEMPLATE.md)
- [`workspace/factors/weekly/TEMPLATE.md`](../workspace/factors/weekly/TEMPLATE.md)
- [`workspace/factors/monthly/TEMPLATE.md`](../workspace/factors/monthly/TEMPLATE.md)
- [`workspace/factors/intraday/TEMPLATE.md`](../workspace/factors/intraday/TEMPLATE.md)

`fz factor new` 生成最小骨架，TEMPLATE.md 提供更完整的约定与示例；两者择一起步即可。
```

- [ ] **Step 2: 新增进阶 worked example 小节**

在 `docs/factor-authoring.md` 第 3 节「最小端到端示例」之后（第 4 节「注册与发现」之前）插入新小节：

```markdown
## 3.1 进阶示例：量价相关因子

`volume_return_corr_20d` 是一个更接近真实研究的示例：20 日「1 日收益 × 对数成交量」滚动 Pearson 相关。它展示了多中间列、`rolling_*` 的 `min_samples`、以及方差非正时的守卫写法。完整实现见 [`workspace/factors/daily/volume_return_corr_20d.py`](../workspace/factors/daily/volume_return_corr_20d.py)，配置见 [`workspace/configs/daily/volume_return_corr_20d.yaml`](../workspace/configs/daily/volume_return_corr_20d.yaml)。

运行并查看报告：

\`\`\`bash
pixi run fz factor run --config workspace/configs/daily/volume_return_corr_20d.yaml
pixi run fz report path <run_id>
\`\`\`

它产出的真实 tear sheet 已收录为示例报告：[docs/examples/](examples/)。该因子预测能力偏弱，正好演示报告如何诚实暴露「统计显著但经济意义弱」的结论——这是 FactorZen 的设计取向，而非缺陷。

要点（相对最小示例新增的）：

- 多个 `rolling_mean(...).over("ts_code")` 中间列拼出协方差与方差，再算相关系数。
- `min_samples` 控制窗口内最少有效样本，避免早期窗口噪声。
- 用 `pl.when(...).then(...).otherwise(None)` 守卫方差非正的退化情形，并 `clip(-1.0, 1.0)` 约束到合法相关系数区间。
```

> 注：落盘时写成真实三反引号围栏。

- [ ] **Step 3: 面包屑加入示例报告入口（第 3 行）**

`docs/factor-authoring.md` 第 3 行当前为：

```markdown
> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · **因子编写** · [运行手册](runbook.md)
```

改为：

```markdown
> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · **因子编写** · [运行手册](runbook.md) · [示例报告](examples/)
```

- [ ] **Step 4: 验证**

```bash
grep -n "TEMPLATE.md" docs/factor-authoring.md
grep -n "3.1 进阶示例" docs/factor-authoring.md
grep -n "示例报告.*examples" docs/factor-authoring.md
```

Expected: 三条均命中。

- [ ] **Step 5: 提交**

```bash
git add docs/factor-authoring.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: factor-authoring 增加进阶示例与 TEMPLATE.md 引用"
```

---

### Task 8: CHANGELOG.md [Unreleased] 增补

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Changed 段增补 walk-forward 默认关闭**

在 `CHANGELOG.md` `## [Unreleased]` → `### Changed` 段追加一条：

```markdown
- **Walk-forward：** 策略 walk-forward 样本外评估改为**默认关闭**（`WalkForwardConfig.enabled` 默认 `false`），按需通过 YAML `walk_forward.enabled: true` 或 `--set walk_forward.enabled=true` 开启。
```

- [ ] **Step 2: Added 段增补示例报告与示例因子**

在 `### Added` 段追加：

```markdown
- **示例报告：** 新增 `docs/examples/`，收录示例因子 `volume_return_corr_20d` 的真实 tear sheet（`docs/examples/volume_return_corr_20d-tear-sheet.html`）与分区导读 README。
- **示例因子：** 新增 `workspace/factors/daily/volume_return_corr_20d.py`（20 日量价滚动相关）及其配置，并在 factor-authoring 中作为进阶 worked example。
- **因子模板：** 各频率目录新增 `TEMPLATE.md` 手写模板，并在 factor-authoring 中引用。
```

- [ ] **Step 3: Fixed 段增补文档对齐**

在 `### Fixed` 段追加：

```markdown
- **文档对齐：** 修正 README / runbook 把 walk-forward 误述为「无 YAML 默认开启」的过时描述；消除 runbook 中 LLM 默认行为自相矛盾的两处表述，统一为与当前代码一致的口径。
```

- [ ] **Step 4: 验证**

```bash
grep -n "Walk-forward：" CHANGELOG.md
grep -n "示例报告：" CHANGELOG.md
grep -n "文档对齐：" CHANGELOG.md
```

Expected: 三条均命中。

- [ ] **Step 5: 提交**

```bash
git add CHANGELOG.md
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: CHANGELOG 记录 walk-forward 默认关闭与示例报告接入"
```

---

### Task 9: 一致性 pass 与最终验证

**Files:**
- 只读核对，必要时微调 Task 2–8 已改文件。

- [ ] **Step 1: 全局核对 walk-forward 默认描述一致**

```bash
rg -n "walk.forward" README.md docs/*.md
```

人工确认：凡是描述「无 YAML 默认」特性清单的地方都已不含 walk-forward；凡是提 walk-forward 的地方都附「默认关闭 / 按需开启」语义或属能力罗列。

- [ ] **Step 2: 全局核对 LLM 口径一致**

```bash
rg -n "LLM|llm_explain|--llm-explain" README.md docs/runbook.md docs/project-explanation.md
```

人工确认：无 YAML / `--all` 默认尝试 LLM、缺配置跳过；自定义配置需 `--llm-explain`；无「默认关闭」类与之矛盾的表述。

- [ ] **Step 3: 核对示例报告链接可达**

```bash
rg -n "examples/" README.md docs/README.md docs/factor-authoring.md
ls docs/examples/README.md docs/examples/volume_return_corr_20d-tear-sheet.html
```

Expected: README、docs/README、factor-authoring 均有指向 `examples/` 的链接；两个文件都存在。

- [ ] **Step 4: 确认无代码 / 测试改动**

```bash
git diff --name-only master...HEAD
```

Expected: 仅出现 `README.md`、`CHANGELOG.md`、`docs/**`；不含任何 `src/**` 或 `tests/**`。

- [ ] **Step 5: 如有微调则提交**

```bash
git add -A
git -c user.name='rookiewu417' -c user.email='1007372080@qq.com' commit -m "docs: 文档一致性收尾" || echo "无需收尾提交"
```

---

## 验收标准（对照 spec）

- `docs/examples/` 含可直接打开的示例报告 HTML 与走查 README，且被 `README.md` / `docs/README.md` / `docs/factor-authoring.md` 链接。（Task 1/2/3/7）
- 全部文档对 walk-forward 默认状态描述一致：默认关闭、opt-in。（Task 2/4/5/6/9）
- runbook 内部不再自相矛盾，LLM 默认行为描述与当前代码一致。（Task 4/9）
- factor-authoring 含 `volume_return_corr_20d` 进阶示例并引用 TEMPLATE.md。（Task 7）
- CHANGELOG `[Unreleased]` 记录上述用户可见变更。（Task 8）
- 不产生任何代码 / 测试改动；提交身份均为 `rookiewu417 <1007372080@qq.com>`。（Task 9 / 全程）
