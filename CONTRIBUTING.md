> [FactorZen](README.md) · **贡献指南**

# 贡献指南

FactorZen 是以**因子库准入**为核心的量化研究平台。优先接受能提升**研究可信度、可复现性、测试判别力与文档准确性**的改动。

在动手之前，先读[三条设计铁律](README.md#三条设计铁律)——PIT 无未来函数、护栏咬合、可复现。冲突时它们是裁决依据，评审也按它们来。

---

## 环境

**本项目没有全局 Python。** 所有命令都经 [pixi](https://pixi.sh/) 执行，不要用系统 `python` / `pip`，也不要装全局包。

```bash
pixi install
cp .env.example .env    # 填入 TUSHARE_TOKEN（仅真实数据拉取需要）
pixi run smoke
```

任务定义在 `pixi.toml` 的 `[tasks]`，常用的几个：

| 任务 | 展开 | 用途 |
|---|---|---|
| `pixi run fz` | `python -m factorzen.cli.main` | CLI 入口 |
| `pixi run lint` | `ruff check .` | 全仓 lint |
| `pixi run typecheck` | `mypy` | 全量类型检查 |
| `pixi run typecheck-fast` | `dmypy run -- src/factorzen` | 增量类型检查（daemon，首跑后秒级） |
| `pixi run test` | `pytest tests/ -n auto` | 全量测试 |
| `pixi run coverage` | `python tools/run_coverage.py` | 全量测试 + 覆盖率门槛 |

任意包内的命令都可以用 `pixi run -- <cmd>` 直接跑，例如 `pixi run -- pytest tests/test_x.py -v`。

---

## 三层验证

不同阶段跑不同强度的检查。**每层都不能省，但可以降频。**

### 第一层 · 内环：定向测试

改代码的循环里只跑与改动直接相关的测试，用 `-x -q` 快速失败：

```bash
pixi run -- pytest tests/test_lift_test.py::test_lift_admission_rejects_low_se -v
pixi run -- pytest tests/ -k "pit or neutral or membership" -x -q
```

> ⚠️ **绝不在前台跑无界的全量 pytest**——套件有 2,561 个用例、314 个测试文件，历史上出现过挂死。全量只在合并前跑，且必须带超时。

### 第二层 · 提交前：lint + typecheck

```bash
pixi run lint
pixi run typecheck
```

两条都必须**扫全仓**：`lint` 包含 `tests/`，`typecheck` 覆盖整个 `src/factorzen`。

> ⚠️ 只 lint 自己改的模块会漏掉别处，push 之后在 CI 上炸。迭代频繁时可以用 `pixi run typecheck-fast`（mypy daemon）顶替，但**合并前必须跑一次冷 `typecheck`** 兜住 daemon 的 stale 状态。

### 第三层 · 合并前：一次全量

```bash
timeout 900 pixi run test
```

> ℹ️ 全量测试不能被定向测试替代。本仓库最大的一类缺陷是**双路径漂移**——同一份语义有两处实现，改一侧忘了另一侧。这类问题定向测试抓不到，只有全量能抓。所以策略是「降频，不是省略」：一批改动合并前跑一次，而不是每次编辑都跑。

---

## 绝对不要做的两件事

### 1. 不要跑 `pixi run format`

`pixi.toml` 里有 `format` 任务，但它是 `ruff format .`，会**一次性改动数百个文件**、把真实改动淹没在格式 diff 里。

> ✅ 格式问题按 `pixi run lint` 的报错**逐处修**。
>
> ℹ️ 例外：`.pre-commit-config.yaml` 里的 `ruff-format` hook 只作用于**你已暂存的 Python 文件**，与全仓 format 是两回事，启用 pre-commit 是安全的。

如本机装了 `pre-commit`，可启用本地钩子（版本与 CI 一致，都经 `pixi run`）：

```bash
pre-commit install
```

### 2. 不要 `git add -A` / `git add .`

工作区常年存在未跟踪的本地文件——行情数据、缓存、研究产出、草稿。批量 add 会把它们卷进提交。

> ✅ **精确 `git add` 你改的那几个文件。** 提交前先 `git status --short` 过一眼。

---

## Git 工作流

### 走 PR 还是直接 master

| 改动性质 | 流程 |
|---|---|
| 新模块 / 新能力 | 分支 + PR + CI 绿后 merge |
| 跨模块重构 | 分支 + PR |
| 改承重文件（`daily/evaluation/backtest.py`、`cli/main.py`、`discovery/lift_test.py` 等） | 分支 + PR |
| 单文件小 bug 修复 | 可直接 master |
| 文档 / 单测补充 | 可直接 master |

> ⚠️ **判不准就当「大改」走 PR。** 误开 PR 的成本是几分钟，误直推 master 的成本是回滚一条已被别人拉走的历史。

PR 合并后删除 head 分支。栈式 PR 要注意：下层合并并删除分支时，GitHub 会直接关闭依赖它的上层 PR——先把上层 rebase 到 master 再重建。

### 提交信息

遵循 [Conventional Commits](https://www.conventionalcommits.org/)，**正文可以用中文**：

```text
fix(discovery): lift 裁决 SE 非有限时改为 reject——修静默放行
feat(intraday): 日内特征叶子接入挖掘全链（零回归）
perf(risk): 风格因子一次物化 + numpy OLS 替换逐日回归
docs: CLI 参考对齐 parser 真实参数
test(backtest): 补空权重表 carry 语义的回归断言
```

常用类型：`feat` / `fix` / `perf` / `refactor` / `test` / `docs` / `chore`。

**一个逻辑一个 commit。** 顺手的格式调整、无关的重命名不要塞进功能 commit。

### 并行工作

同时开多条线时用 `git worktree` 隔离，不要在同一个 checkout 里来回切分支——工作区里的未跟踪产物会互相污染。

---

## 写代码的约定

### 测试先行

修 bug 先写一个**能复现问题的失败测试**，再动实现。这不只是流程洁癖：没有先失败过的测试，无法证明它真的在测那个 bug。

### 测试必须有判别力

本仓库反复踩过的坑是**恒真断言**：用 `A` 和 `B` 构造出 `C`，然后断言 `C == f(A, B)`——这永远成立，零判别力。

> ✅ 用独立公式、外部 ground truth、或反例来验证。等价性重构要拿**旧实现的拷贝**当 golden 逐值比对。
>
> ⚠️ 「测试全绿」不等于「测试有效」。把契约字段改成 `Optional` 之后，原本有判别力的断言可能被静默恒真化——改契约时要 grep 全部直接引用逐个审。

### PIT 自查

任何触及收益、价格、成交约束、样本切分的改动，都必须在 PR 里说明**是否可能引入未来函数**，并补相应回归测试。自查清单：

- universe 是逐日快照，不是期末快照。
- 财务数据按公告日对齐，不按报告期。
- 执行定价用 `pre_close`，不用当日收盘。
- 滚动因子要扩窗预热（`expanded_start`）。
- 预处理的统计量只用 ≤t 的样本。
- t 日算出的信号在 t+1 执行。

### 双路径登记簿

平台里有若干处「同一语义、两套实现」的配对（如模拟交易与向前执行的信号执行时点、挖掘护栏与 Agent 护栏的 `passed` 判定、单因子链路与报告链路的回测参数）。**改任一侧必须检查配对侧，两侧都要有测试。**新增第二条路径时，必须同时加一致性测试。

### 异常契约

解析外部输入（LLM 输出、因子表达式、配置文件）时只抛 `ValueError` 一类异常。`except ValueError` 接不住 `IndexError`——一条畸形输入不许崩掉整个 session。

### 退化截面守卫

单股票、全 NaN、样本数小于分组数、近常数序列都要显式守卫：秩相关在 n=2 时恒为 ±1，分组回测的空腿会变成裸头寸，`E[x²]−E[x]²` 的微负值开方会让 NaN 穿透下游。

### 代码放哪

- 框架代码 → `src/factorzen/`
- 用户可扩展因子 → `workspace/factors/{daily,weekly,monthly,intraday}/`

各频率目录下有 `TEMPLATE.md` 手写模板，写法见[因子编写](docs/guides/factor-authoring.md)。

### 不要提交的东西

行情数据、运行产物、日志、notebook checkpoint、`.env`、任何 token。数据一律落 `data/`，研究产出一律落 `workspace/`，两者都已 gitignore。

---

## CI

CI 在向 `main` / `master` 的 push 与 PR 上触发，跑 **3 个实质步骤**：

| 步骤 | 命令 |
|---|---|
| Lint | `pixi run lint` |
| Type check | `pixi run typecheck` |
| Test + Coverage | `pixi run coverage` |

`coverage` 任务本身就是全量 pytest（`-n auto` + `pytest-cov`），测试失败即红，所以不需要额外的 Test 步骤。覆盖率门槛为 **74%**，低于门槛直接失败。

任一步骤红即 fail。本地按[三层验证](#三层验证)跑过，CI 基本不会有意外。

PR 模板 `.github/PULL_REQUEST_TEMPLATE.md` 里的自查项请如实勾选——特别是「无未来函数」与「无凭据入库」两条。

---

## 文档

- 对外文档（`README.md` / `docs/`）按**能力**组织，不暴露内部里程碑代号与过程术语。
- 写文档或示例前，先跑 `pixi run -- fz <cmd> --help` 对照 `src/factorzen/cli/` 的真实 parser，**不要凭印象写命令**。help 里承诺的行为、报错里引用的旗标，必须真实存在。
- 已知限制**就地标注**在对应能力旁，不要用模糊措辞掩盖。MVP 阶段的东西如实写成 MVP。
- 每份文档职责单一，互相链接而非互相复述。

---

## 报告问题

普通问题走 GitHub Issues。

**安全问题不要公开披露**，处理方式见 [SECURITY.md](SECURITY.md)。
