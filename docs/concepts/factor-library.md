# 因子库与增量准入

> [FactorZen](../../README.md) · [文档](../README.md) · **因子库与增量准入**

这是 FactorZen 与常规因子平台差别最大的一层，也是整个平台的裁决中枢。读懂这一份，就读懂了平台的研究哲学。

---

## 为什么不用单因子指标决定入库

一个候选因子 RankIC 0.03、IR 0.4、DSR 显著、holdout 也没垮——看起来该收。但如果它和库里已有的某个动量因子相关性 0.85，那么把它加进组合，除了增加维护成本和拥挤风险之外，**对组合的预测能力几乎没有贡献**。

单因子指标衡量的是「这个因子相对于零有没有信号」。而研究真正要回答的是「这个因子相对于**我已经拥有的东西**有没有信号」。这两个问题在因子库积累到一定规模后会给出完全相反的答案，且随着库变大，分歧只会更严重。

FactorZen 的选择是把后者作为最终裁决：

| | 传统做法 | FactorZen |
|---|---|---|
| 入库判据 | 单因子指标过门槛 | **相对现有库的增量（lift）显著** |
| 单因子门槛的角色 | 硬门，一票否决 | **降级为排序信号**，用于挑选先测谁 |
| 硬门还剩什么 | 一整套阈值 | **只剩数据质量** |
| 库的性质 | 越堆越长的候选表 | 持续收敛、互相不冗余的资产 |

代价是准入变贵了——每评一个候选都要重跑一遍「基线 vs 基线+候选」的对比。平台为此做了基线复用、库池子进程隔离等一系列工程优化，见[性能与资源](../guides/performance.md)。

---

## lift 是什么

**lift = 把候选因子加进基线组合后，样本外表现的增量。**

具体做法：在交叉验证的每个测试折上，分别评估「仅基线因子」和「基线 + 候选因子」两个组合，取二者表现之差；对所有折汇总得到 `lift` 及其标准误 `lift_se`。

关键在于它是**配对**统计——同一折、同一 universe、同一窗口下的两个组合直接相减，共同的市场噪声被抵消掉，剩下的才是候选因子的净贡献。

裁决实现在 `discovery/lift_test.py:532` 的 `lift_admission()`，是全平台唯一的准入裁决函数。

---

## 裁决规则

```text
门槛  bar = max(threshold, se_mult × lift_se)

lift 为 None / 非有限（NaN, ±inf）      → reject
lift_se 为 None / 非有限                → reject
lift < bar                              → reject
lift ≥ bar 且 lift_second_half > 0      → active
lift ≥ bar 且 second_half ≤ 0 或缺失    → probation
```

三处设计值得留意：

**门槛取 `max(绝对阈值, SE 倍数)`。** 绝对阈值挡住「统计显著但幅度小到没有经济意义」的候选；SE 倍数挡住「幅度看着大但噪声更大」的候选。两道都要过。

**SE 不可用即 reject，不退化为裸阈值。** 这是刻意的保守：区间证据不完整时，平台选择拒绝而不是假装 SE 为 0 从而让门槛塌缩成 `threshold`。同样的约定也用在组门（`lift_test.py:107-122`），代码注释明确写了「不再把『无 SE』当『零方差』」。

**后半段决定是 `active` 还是 `probation`。** 把评估窗口切两半，若后半段的增量为正，说明这个因子的贡献不是集中在样本前期的一次性效应，直接给 `active`；否则先收进 `probation`，等真实时间累积更多证据再定。

### 实际阈值

| 常量 | 值 | 定义处 |
|---|---|---|
| `DEFAULT_LIFT_THRESHOLD` | `0.001` | `discovery/guardrails.py:39` |
| `se_mult` 默认 | `1.0` | `lift_test.py:536`，CLI `--lift-se-mult` |
| `DEFAULT_HOLDOUT_MIN_DAYS` | `60` | `discovery/guardrails.py:28` |
| 交叉验证默认 | `train_days=250, test_days=40, purge_days=5, embargo_days=0` | `lift_test.py:60-66` |
| `DEFAULT_TOP_M` / `DEFAULT_HORIZON` / `DEFAULT_BLOCK_DAYS` | `10` / `5` / `20` | `lift_test.py:51-53` |

> ⚠️ **两套 `se_mult` 别混淆。** lift 准入（`fz factor-library lift-test --lift-se-mult`）默认 **1.0**；向前复审（`fz factor-library forward-review --se-mult`）默认 **1.645**（单侧 95%）。两者作用于不同证据、不同时间尺度，阈值不通用。

---

## 四态状态机

每条因子记录（`FactorRecord`，`discovery/factor_library.py:137`）带一个 `status`：

| 状态 | 含义 | 参与组合 |
|---|---|---|
| `active` | 增量显著且后半段确认 | ✅ |
| `probation` | 增量过门槛但后半段未确认，观察中 | ❌ 待确认 |
| `correlated` | 与在库因子高度相关 | 打标收录，记 `correlated_with` |
| `no_lift` | 无增量 | ❌ |

`correlated` 这一态体现了平台的去相关策略：**高相关因子仍然收录并打标，而不是直接丢弃**。理由是相关性是相对当前库的快照结论——库演化后，今天被判定冗余的因子可能明天就有位置了。丢弃会丢失信息，打标则保留了复审的可能。

---

## probation 的完整生命周期

这是因子库最容易被忽略、却最能体现「诚实」的机制。一个拿到 `probation` 的因子要转正，必须靠**真实时间**积累证据：

```text
lift_admission 判 probation
   （增量过门槛，但后半段未确认）
        │
        ▼
fz factor-library forward-track       逐日记录 paper forward RankIC
   --max-backfill-days=10             ⚠️ 默认拒绝历史回灌
        │
        │  累积 ≥ --min-days（默认 60）个有效交易日
        ▼
fz factor-library forward-review --apply
   --se-mult 1.645（单侧 95%）
        │
        ├──→ promote → active
        └──→ demote  → no_lift
```

`--max-backfill-days` 默认只允许极少量回灌，本质上是拒绝用历史数据把观察期「补」出来——如果允许无限回灌，向前确认就退化成了又一次样本内检验，失去全部意义。

> ⚠️ **当前接线缺口**：`forward-track` 尚未接进 `fz ops daily` 的 8 阶段无人值守链路，probation 因子的每日记录目前需要人工执行。这是已知待办，命令的 help 文本里也标注了。

---

## 登记簿的数据模型

`FactorRecord`（`discovery/factor_library.py:137`）是唯一登记簿记录，40+ 个字段按语义分组：

| 组 | 内容 |
|---|---|
| 身份 | `expression`、`market`、`hypothesis` |
| 训练期指标 | `ic_train`、`ir_train`、`n_train` |
| holdout | `holdout_ic`、`holdout_ir`、`holdout_n_days` |
| 统计护栏 | `dsr`、`dsr_pvalue`、`ic_ci_low/high`、`pbo`、`turnover` |
| 状态机 | `status`、`max_corr_in_lib`、`correlated_with` |
| 准入轨道 | `admission_track`、`admission_ic`、`admission_decision`、`evidence_tier` |
| **lift 证据** | `lift`、`lift_baseline`、`lift_se`、`lift_first_half`、`lift_second_half`、`lift_threshold`、`lift_se_mult`、`baseline_hash` |
| 可复现窗口 | `eval_start/end`、`universe`、`horizon`、`admission_start/end`、`scored_start/end`、`block_days`、`cv_train_days/test_days`、`profile_name`、`frequency` |
| 溯源 | `source_run_id`、`source_session_dir`、`git_sha`、`added_at`、`updated_at`、`forward_confirmed_at`、`forward_n_days` |
| 形态 | `kind`、`name`、`impl` |
| 评估链接 | `last_eval_run_id`、`last_eval_at` |

「可复现窗口」那一组是[可复现铁律](design-principles.md)在库层的落实——记录不只存结论，还存**得出这个结论的全部条件**：哪段窗口、哪个 universe、什么 CV 参数、什么阈值、基线是哪个版本（`baseline_hash`）。任何一条准入结论事后都能被重新验算。

> ℹ️ `last_eval_run_id` / `last_eval_at` 是**指向评估产物的链接，不是裁决指标**。它们不会覆盖 `ic` / `lift` / `status`——评估可以反复跑，裁决只认准入时那一次。

---

## 表达式因子与 python 因子共存

因子库同时容纳两种形态：

- **表达式因子** —— 挖掘产出的字符串表达式，如 `rank(ts_std(close, 20))`
- **python 因子** —— 手写的 `DailyFactor` 子类

问题在于登记簿的一切逻辑（去重、池键、台账键）都以 `expression` 为主键。为 python 因子引入第二套主键会让所有消费方分叉。

平台的解法是 **`py::` 身份哨兵**（`factor_library.py:254`）：python 因子的 `expression` 字段填成 `"py::{name}"` 这样一个**故意不合法**的表达式串。它不可解析，因此绝不会被误当成表达式求值；但它是唯一的字符串，因此去重、池键、台账全部零改动继续工作。

真正的实现由 `kind` / `name` / `impl` 三个显式字段承载，且**优先级高于** `py::` 推断（`factor_library.py:1083`）。消费方按 `kind` 分派到不同的物化路径。

写手写因子并入库的操作步骤见[因子编写指南](../guides/factor-authoring.md)。

---

## 常用命令

```bash
# 查看库现状
pixi run fz factor-library list --market ashare
pixi run fz factor-library show --market ashare --rank 1
pixi run fz factor-library show --market ashare --expression "rank(ts_std(close,20))"

# 增量准入（默认 dry-run，只打印裁决）
pixi run fz factor-library lift-test --market ashare \
  --session workspace/mining_sessions/session_42_genetic \
  --start 20200101 --end 20231231

# 确认后写库
pixi run fz factor-library lift-test --market ashare \
  --session workspace/mining_sessions/session_42_genetic \
  --start 20200101 --end 20231231 --apply

# 向前确认（probation → active/no_lift）
pixi run fz factor-library forward-track --market ashare
pixi run fz factor-library forward-review --market ashare --apply

# 重建登记簿（含复审）
pixi run fz factor-library rebuild --market ashare

# 空假设校准：蒙特卡洛跑随机序列，看准入规则本身的误报率
pixi run fz factor-library lift-null --n-sims 1000 --seed 42
```

> ⚠️ `lift-test` 与 `forward-review` **默认 dry-run**，必须显式 `--apply` 才写库。准入是不可逆的库变更，这道确认是有意的。

全部参数见 [CLI 参考](../reference/cli.md)。

---

## 相关阅读

- [设计铁律](design-principles.md) —— 准入规则背后的 PIT / 护栏 / 可复现三原则
- [防过拟合护栏](guardrails.md) —— DSR、PBO、holdout 与空假设校准如何与准入咬合
- [架构](architecture.md) —— 因子库在整体数据流中的位置
- [多因子组合](../guides/combination.md) —— 如何从库里消费因子做组合
