# 多因子组合

> [FactorZen](../../README.md) · [文档](../README.md) · **多因子组合**

因子库积累起来之后，下一个问题是：**这些因子合起来能做出什么样的策略？**

`fz combine` 回答的就是这个。它在同一套 purged & embargoed walk-forward 协议下跑四种合成方法，给出可横向对比的样本外指标。它不是组合优化（那是 [`fz portfolio build`](risk-and-portfolio.md) 的事）——它产出的是**合成后的单个因子序列**，回答「怎么把 N 个因子揉成 1 个」，不涉及权重求解与约束。

---

## 入口的区别

前三个子命令跑的是**完全相同的 OOS 组合实验**，区别只在**因子从哪来**；`backtest` 是把 OOS 分数接进真回测的桥：

| 子命令 | 因子来源 | 选品方式 | 用途 |
|---|---|---|---|
| `fz combine run` | 裸 parquet 文件 | 你自己给文件列表 | 自定义因子、外部因子、快速验证 |
| `fz combine from-session` | 挖掘 session 的 `candidates.csv` | 默认只取 `passed` | 一轮挖掘刚跑完，看这批候选合起来行不行 |
| **`fz combine from-library`** | **因子库登记簿** | 按 `status` 过滤 + `\|ic_train\|` 降序 | **因子库的正式消费出口** |
| `fz combine backtest` | `oos_scores/*.parquet` 或任意分数面板 | 指定 `--method` / `--scores` | **OOS 组合分数 → 真回测（日环引擎）** |

选哪个：

- **日常研究选 `from-library`。** 库是经过 lift 准入裁决沉淀下来的资产，`from-library` 是把这份资产变成策略的正式路径。
- **`from-session` 是挖掘的即时验收。** 一个 session 刚跑完，还没走完准入流程，想先看看这批候选组合起来的样子。注意它吃的是 session 快照，不带准入结论。
- **`run` 是逃生口。** 因子来自平台之外，或者你想手工控制精确的输入面板。

> ℹ️ `run` / `from-session` / `from-library` 共享同一组切分与输出参数（`--train-days` / `--test-days` / `--purge-days` / `--embargo-days` / `--methods` / `--seed` / `--run-id` / `--out-dir`），产物格式完全一致，可以互相对比。

---

## 四种合成方法

所有方法都在**截面 z-score 标准化后**的因子值上操作。

| 方法 | 做法 | 特点 |
|---|---|---|
| `equal_weight` | 各因子截面 z-score 后取均值 | 无参数、无估计误差，最稳健的基线 |
| `ic_weighted` | 权重 = `max(0, IC 均值)` 归一化，**负 IC 因子权重归零** | 让历史上更有效的因子占更大比重 |
| `max_ir` | 闭式解 `w = Σ⁻¹·μ`，协方差用 **Ledoit-Wolf 收缩** | 理论最优 IR，但对协方差估计误差敏感 |
| `lgbm` | LightGBM 学非线性与因子交互，标签用**截面 rank 归一** | 唯一能捕捉非线性的方法 |

`--methods` 默认 `all`（四种全跑），也可以给逗号分隔的子集：`--methods equal_weight,lgbm`。

> ℹ️ `max_ir` 的 Ledoit-Wolf 收缩不是可选项而是必需的——因子数一多，样本协方差矩阵的逆会被估计噪声放大到不可用。
>
> **两处静默退化要知道**：`ic_weighted` 在所有因子 IC 都非正时退化成等权；`max_ir` 在数据不足以估协方差时也退化成等权。这两种情况下对比表里的三行会长得几乎一样——**看到 `equal_weight` 与另外两个方法指标高度接近时，先怀疑是退化而不是巧合**。
>
> `lgbm` 固定 `deterministic + num_threads=1 + 固定 seed`，同 seed 下结果可复现。

---

## 样本外协议：为什么不能直接比 IC

这是整份文档最重要的一节。

四种方法里有三种要**估权**（`ic_weighted` 用历史 IC、`max_ir` 用历史均值与协方差、`lgbm` 要训练）。如果在全样本上估权、再在全样本上评估，估权用到的信息和评估用的信息重叠，结果必然是「方法越复杂看着越好」——这是纯粹的样本内偏差，不是真实能力差异。

平台的做法是 **purged & embargoed walk-forward CV**（`research/combination/cv.py`，López de Prado, AFML）：逐折 **train 段估权 / test 段应用**，只统计 test 段的表现。

```text
|<-- train_days -->|purge|embargo|<- test ->|
                    ↑     ↑
                    │     └─ 额外隔离带，压序列自相关导致的泄漏
                    └─ 剔掉与 test 标签窗口重叠的 train 末尾样本
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--train-days` | `120` | 训练窗长度（交易日） |
| `--test-days` | `20` | 测试窗长度 |
| `--purge-days` | `5` | 剔除 train 末尾与 test 标签窗重叠的样本 |
| `--embargo-days` | `0` | test 前的额外隔离带 |

**purge 的天数应该 ≥ 前向收益的 horizon。** 用 `--horizon 5` 的 5 日前向收益时，train 末尾 5 天的标签会伸进 test 区间——不 purge 掉就是直接的标签泄漏。默认 `--purge-days 5` 正好配 `--horizon 5`，改了一个记得改另一个。

> ⚠️ **CLI 层的 CV 是 expanding（训练集展开）**，不是定长滚动（`PurgedWalkForwardCV.expanding` 默认 `True`，CLI 不透传这个参数）。意味着越靠后的折训练样本越多。
>
> 另外：日期数不足以构成任何一折时会直接 `ValueError`。至少要 `train_days + test_days + purge_days` 天的可用数据。

> ⚠️ 库内 Python API `methods.equal_weight` / `ic_weighted` / `max_ir` 的**直接调用是样本内口径**，只适合做方法对比与候选筛选。无偏的样本外结论必须走 `oos.combine_oos` / `models.combine_lgbm` / `experiment.run_combination_experiment`——也就是 `fz combine` 这三个子命令走的路径。

---

## 从因子库消费因子

`from-library` 是因子库登记簿的正式消费出口，完整链路是：

```text
因子库登记簿
   │  按 --statuses 过滤
   ▼
按 |ic_train| 降序排序（与库池同序）
   │  --top-n 截断（必记 truncated_from，禁止静默 cap）
   ▼
逐因子物化到 [start, end] × universe 的面板
   │  表达式因子走 AST 求值；python 因子走 registry 物化
   ▼
按 |holdout_ic| 降序贪心去相关（--decorr-threshold）
   ▼
四方法 OOS 对比
```

```bash
pixi run -- fz combine from-library --market ashare --statuses active \
  --start 20200101 --end 20241231 --universe csi500 --horizon 5
```

### `--statuses` 怎么选

| 取值 | 含义 |
|---|---|
| `active`（默认） | 只用增量显著且后半段确认的因子 —— **生产口径** |
| `active,probation` | 把观察期因子也纳进来 —— 看「如果这批 probation 全部转正，组合会好多少」 |
| `correlated` | 被判定与在库因子高相关的那些 |
| `no_lift` | 无增量的 —— 一般只用于对照实验 |

四个状态的语义见[因子库与增量准入](../concepts/factor-library.md#四态状态机)。

> ⚠️ `from-library` **没有 `--all`**（那是 `from-session` 独有）。要放宽选品范围只能用 `--statuses`。`--statuses` 有自定义校验，非法值或空串会被 argparse 直接拒绝。

### 库里有 python 因子时，`--universe` 必填

```text
combine from-library：选品含 python 型因子时 universe 必填
（组合路径 fail-loudly，不同于库池的跳过+告警）
```

这是刻意的差别：库池构建时遇到无法物化的 python 因子会「跳过 + 告警」继续跑；但**组合路径直接报错退出**。理由是组合少一条腿会直接改变实验结论——你会拿到一个看似正常、实则少了几个因子的对比表，而且不会注意到。

> ℹ️ 手写 python 因子如何入库见[因子编写](factor-authoring.md#让手写因子进因子库)。python 因子在组合里和表达式因子完全平权，报告里显示的是可读的 `name` 而不是 `factor_0` 这类占位符。

### 去相关

`--decorr-threshold`（默认 `0.7`）在物化之后、喂进组合之前执行：按 `|holdout_ic|` 降序贪心纳入，与已纳入池 `max|corr|` 超阈值的剔除。`1.0` 关闭（逃生口）。

这一步是必需的。跨 run 合并时 `ts_rank(turnover_rate, 20)` 和 `ts_rank(turnover_rate, 21)` 这类近亲会同时入选，把组合的有效 breadth 塌缩掉——名义上 20 个因子，实际只有 5 个独立方向。

被剔的因子会在终端逐条打印：

```text
[combine]   ✗ ts_rank(turnover_rate,21) → 与 ts_rank(turnover_rate,20) 相关 0.94
```

> ⚠️ **组合至少需要 2 个因子。** 库因子 < 2、可物化 < 2、或去相关后 < 2 都会直接报错。看到「因子库不足 2 个」时的办法是放宽 `--statuses` 或者先去挖矿 / 跑 lift-test 入库。

---

## 从挖掘 session 消费

```bash
pixi run -- fz combine from-session \
  --session workspace/mine_team/20260718_120000 \
  --start 20200101 --end 20241231 --universe csi500 --horizon 5
```

- `--session` 是**空格分隔多值**：`--session a b c`。多个 session 的 `candidates.csv` 会合并 + 按**规范形**去重（跨 run 的同一表达式只留 `|holdout_ic|` 更大的那条）。
- 默认只取 `passed=True` 的候选，`--all` 放开到全部。
- 其余流程（物化 → 去相关 → 四方法 OOS）与 `from-library` 完全一致。

> ℹ️ 因子在**含预热前缀的完整帧**上求值，然后裁剪到 `>= start`——和挖掘路径同一套扩窗预热逻辑，避免首段因窗口不满而失真。

---

## 从裸 parquet 消费

```bash
pixi run -- fz combine run \
  --factor workspace/f/a.parquet --factor workspace/f/b.parquet \
  --ret workspace/ret/h5.parquet --train-days 120 --test-days 20 --purge-days 5
```

文件契约：

| 文件 | 必需列 |
|---|---|
| `--factor` | `trade_date` · `ts_code` · `factor_value` |
| `--ret` | `trade_date` · `ts_code` · `ret` |

**因子名取自文件名（`Path(f).stem`）**，会出现在 `comparison.csv` 和报告里——文件叫 `tmp1.parquet` 报告里就是 `tmp1`，起名时留意。

> ⚠️ `--factor` 是 **append 型多值**：靠重复旗标给多个（`--factor a.parquet --factor b.parquet`），**不能写成逗号分隔**。这和 `from-session` / `from-library` 的 `--session`（空格分隔）风格不同——本 CLI 里三种多值风格并存，逐命令看[参数表](../reference/cli.md#fz-combine)。

> ⚠️ `--ret` 里的 `ret` 必须是**前向**收益，且与因子日对齐。用错成同期收益，四种方法的 IC 都会好得离谱——这类错误没有任何护栏能替你发现。

---

## 产物与怎么读

产物落 `workspace/combinations/<run_id>/`（或 `--out-dir` 下）：

| 文件 | 内容 |
|---|---|
| `combined_<method>.parquet` | 每种方法的**合成因子序列**（`trade_date` · `ts_code` · `factor_value` · `fold_id`），逐折 test 段拼起来的样本外值 |
| `oos_scores/<method>.parquet` | 同上 OOS 面板的分数形态（`trade_date` · `ts_code` · `score`），供 `fz combine backtest` 消费；折间日期零重叠（重叠 = bug，直接 raise） |
| `comparison.csv` | 方法 × 指标对比表 |
| `importance.csv` | LightGBM 因子重要性（跑了 `lgbm` 才有） |
| `report.md` | Markdown 报告，含对比表 + 重要性表 + 最高 RankIC 方法结论 |
| `manifest.json` | `git_sha` / `run_id` / `seed` / `methods` / 因子清单 / `oos_scores` 路径 / 完整 CV 参数 |

### `comparison.csv` 的指标

| 指标 | 含义 | 怎么看 |
|---|---|---|
| `rank_ic_mean` | 逐日截面 RankIC 的均值 | 预测方向的平均强度 |
| `icir` | `mean(IC) / std(IC)` | 信噪比，比 RankIC 更值得看。IC 高但抖得厉害的方法未必好用 |
| `ic_positive_ratio` | IC > 0 的天数占比 | 稳定性。0.5 附近说明方向靠掷硬币 |
| `top_bottom_spread` | 分 5 层的多空日均收益差（**毛**） | 可交易性的粗略代理 |
| `turnover` | 相邻期 top 分位成分变动率 `\|Tₜ\\Tₜ₋₁\|/\|Tₜ\|` | 桶内次序变动不计（不产生交易） |
| **`net_spread_10bp`** | **扣掉交易成本后的 spread** | **要下部署结论就看这个**，不是 RankIC |
| **`net_sharpe_10bp`** | 净 spread 的年化 Sharpe | 净收益的信噪比 |
| `max_drawdown` | 多空 spread 累计序列的最大回撤 | 这条曲线会不会长期趴着 |
| `n_periods` | 参与统计的有效交易日数 | **先看这个**。太小的话其余指标都不可信 |

RankIC 用 average-rank Spearman（`core/stats.spearman_avg_rank`），与 `lift_test` / `ic_analysis` 同口径。单日截面 `n < 10` 的日子跳过。

**净收益口径**：`净 spread = spread − 4 × 换手 × 10bp`，**逐期扣费再平均**（先平均换手再乘成本不等价——会丢掉「换手高的日子收益也高/低」的协同）。spread 是 `top均值 − bottom均值`，即 1 份多头 + 1 份空头；每腿换手 `x` ⇒ 卖 `x` 买 `x`、成交 `2x`，两腿合计 `4x`。**假设空头腿换手与多头腿相同**（只量了 top 桶）。10bp/边是 A 股成本的**乐观**侧（佣金 2.5bp + 印花税摊双边 5bp + 冲击 5–10bp ⇒ 实际 10–15bp），该档已为负则更贵只会更差；融券受限时实盘应看 long-only 形态，成本约减半。

> ⚠️ **`rank_ic_mean` 最高的方法不等于最该上线的方法**，而且这不是空话——2026-07-19 实测：库 120 上 lgbm RankIC 最高（0.0793），但它换手也最高（55.3%/日），10bp 下成本年化 **27.9%**，**吃掉毛 alpha 的 92%**，净仅 +2.44%；四方法在现实成本下净收益全部为负或贴零。**只看 IC 会系统性高估可部署性。** 报告会同时给出「RankIC 最优」与「净 SR 最优」两个方法，两者不同时明确提示。
>
> 若 ML 未显著胜出，线性方法因更稳健、更可解释而更可运营。**「ML 没赢」本身就是一条有价值的实验记录**，不是失败的实验。

### `importance.csv`

三列：`factor` · `importance` · `method`。

`method` 列标注实际用的是 **`shap`** 还是 **`gain`**——SHAP 更忠实，但 `shap` 是 **dev extras 依赖**，只装运行时依赖的环境会自动降级到 LightGBM 内置的 gain。**看这一列**，别假设拿到的就是 SHAP 值。

> ⚠️ LightGBM 重要性是**在全样本上重新 fit 一次**算出来的（`experiment.py` 的 `run_combination_experiment()`），与 OOS 逐折的模型不是同一批。它是解释性辅助，不是样本外证据。

---

## 真回测：`fz combine backtest`

`comparison.csv` 里的 IC / 净 spread@10bp 是**零持仓、按分位桶换手的近似**，不是统一日环引擎的真回测。要把 OOS 组合分数接进策略引擎：

```bash
# 读 combine 产物的 oos_scores
pixi run -- fz combine backtest \
  --run-dir workspace/combinations/<run_id> --method equal_weight \
  --start 20200101 --end 20241231 --universe all_a

# 或任意分数面板
pixi run -- fz combine backtest \
  --scores panel.parquet --score-col score \
  --start 20200101 --end 20241231 --strategy topn_long_only --cost-bps 0
```

| 要点 | 行为 |
|---|---|
| 输入 | `--scores` 与 `--run-dir` **二选一**；后者需 `--method`（默认 `equal_weight`） |
| 策略 | `--strategy` 默认 `quantile_ls_5`（与 `fz factor backtest`/daily_single 无 YAML 默认一致）；另支持 `topn_long_only` 等既有 registry 类 |
| 成本 | `--cost-bps` 缺省 = daily_single 的 `LinearCostModel`；`0` = 零成本；显式 bps = 单边 commission |
| 调仓 | `--rebalance-days` 缺省/1=逐日；`k>1` 时桥层把分数降采样到每 k 日并前向填充（非调仓日权重不变），引擎仍日环、净值逐日更新 |
| 产物 | `workspace/combine_backtests/<run_id>/`：`manifest.json` + `metrics.json` + `nav.parquet` |

数据装配（PIT membership、复权日线、`is_st_by_date`）与 `daily_single` 同口径，避免双路径漂移。

---

## 常见做法

**基线对照。** 先跑一次 `--methods equal_weight` 拿到基线，再跑全四种。如果复杂方法赢不过等权，说明估权引入的噪声超过了它捕捉到的信号——这在因子数少、样本短的时候很常见。

**选品范围的敏感性。** 同一个窗口跑两次，`--statuses active` 和 `--statuses active,probation`，对比 ICIR 变化。这直接回答「观察期那批因子值不值得等」。

**去相关阈值的敏感性。** `--decorr-threshold 0.7` 与 `1.0`（关闭）各跑一次。差距大说明库里的冗余严重，该回头看看准入环节。

**换 `--horizon` 要同步换 `--purge-days`。** 前向收益持有期变了，purge 窗口必须跟着变，否则泄漏。

---

## 相关阅读

- [因子库与增量准入](../concepts/factor-library.md) —— 因子怎么进的库、四态状态机
- [因子挖掘](mining.md) —— `from-session` 消费的 session 从哪来
- [因子编写](factor-authoring.md) —— 手写 python 因子如何进库并参与组合
- [风险与组合优化](risk-and-portfolio.md) —— 拿到合成因子之后的建仓与约束求解
- [防过拟合护栏](../concepts/guardrails.md) —— 组合层的显著性把关
- [`research/combination/` 模块说明](../../src/factorzen/research/combination/README.md) —— Python API 层的用法
- [CLI 参考](../reference/cli.md#fz-combine) —— `fz combine` 全参数
