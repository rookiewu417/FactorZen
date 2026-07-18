# 防过拟合护栏

> [FactorZen](../../README.md) · [文档](../README.md) · **防过拟合护栏**

因子研究的核心风险不是「找不到信号」，而是「找到了不存在的信号」。搜索空间越大、迭代越多，这个风险越高。本文说明平台用哪些统计方法对抗它，以及它们如何真正咬合进筛选而非只出现在报告里。

---

## 代码分工（容易搞错的一点）

| 位置 | 行数 | 角色 |
|---|---:|---|
| `validation/` | 202 | **纯统计原语**：DSR、PBO、bootstrap、holdout 切分的教科书实现 |
| `discovery/guardrails.py` | 422 | **阈值单一真源 + 判定逻辑** |
| `discovery/scoring.py` | 675 | **打分与多重检验记账** |

`validation/` 薄，是因为它只提供无状态的纯函数。**真正决定「过不过」的逻辑不在这里**。想调阈值或改判定，去 `discovery/guardrails.py`。

---

## 五道护栏

### 1. Block bootstrap IC 置信区间

`validation/bootstrap.py:7` `block_bootstrap_ic_ci()`

IC 序列有强自相关，直接按天重采样会严重低估方差、把置信区间做窄，从而高估显著性。平台用 **moving block bootstrap**：按块重采样，保留块内的时序结构。

产出 `ic_ci_low` / `ic_ci_high` 进因子库记录。

### 2. Deflated Sharpe Ratio

`validation/deflated_sharpe.py:15` `expected_max_sharpe()` / `:25` `deflated_sharpe()`

实现 Bailey & López de Prado (2014)。核心思想：如果你试了 N 个策略取最好的那个，即使全部无效，最大 Sharpe 的期望也显著大于 0。DSR 把这个「选择偏差」扣掉。

> ⚠️ **N 的取法决定 DSR 的成败。** N = 真实评估过的**唯一表达式数**（取自求值缓存），不是最终存活集的大小。在迭代循环中对累积状态计数会产生三角和 over-count，必须按轮过滤。取错 N 会让 DSR 系统性偏松，是历史上出现过的真实缺陷。

### 3. PBO（via CSCV）

`validation/pbo.py:13` `compute_pbo(perf_matrix, n_splits=10)`

Probability of Backtest Overfitting，用组合对称交叉验证（López de Prado 2016）估计：「样本内最优的配置，在样本外落到中位数以下」的概率。PBO 接近 0.5 意味着你的选优过程与掷硬币无异。

### 4. Holdout 隔离

`validation/holdout.py` `holdout_boundary` / `split_holdout` / `holdout_ic_result`

holdout 段与训练/验证期严格分离，用于最终验收。

> ⚠️ holdout 滚动因子必须**扩窗预热**——不预热会让 holdout 段起点处的因子值用到不足的窗口，实测可导致 IC 偏差约 40%。
>
> ⚠️ holdout 覆盖不足时必须显式守卫。`DEFAULT_HOLDOUT_MIN_DAYS = 60`（`discovery/guardrails.py:28`）。样本不够就该说不够，而不是用一个天数极少的 holdout 假装做过样本外。

### 5. 空假设校准

`discovery/lift_null.py`（464 行）

前四道回答「这个因子显著吗」。第五道回答一个元问题：**「我这套准入规则本身的误报率是多少？」**

做法是把明确无效的随机因子灌进完整的准入流程，看有多少能混进来。如果 100 个随机因子里过了 15 个，那么准入规则本身就是坏的，前面所有的统计严谨性都是徒劳。

> ℹ️ 这个模块有一条硬性架构守卫：**它必须调用生产环境的 `paired_lift_stats` 与 `lift_admission`，禁止自己重写一遍规则**（docstring 明确要求）。否则校准的是一套影子实现，与真正在跑的规则漂移，校准结果毫无意义。唯一允许的差异是校准层多一个 `min_blocks` 前置参数，代码里已标注。

运行：`pixi run fz factor-library lift-null --n-sims 1000 --seed 42`

> ℹ️ `lift-null` 是纯蒙特卡洛模拟，**没有 `--market` 参数**——它检验的是准入规则本身，与具体市场数据无关。参数是模拟规格（`--n-days`、`--daily-sigma`、`--ar1`、`--se-mults`、`--min-blocks`、`--n-sims`、`--seed`）。

---

## 阈值一览

全部定义在 `discovery/guardrails.py` 与 `discovery/lift_test.py`，是单一真源。

| 常量 | 值 | 用途 |
|---|---|---|
| `DEFAULT_LIFT_THRESHOLD` | `0.001` | lift 绝对门槛下界 |
| `DEFAULT_HOLDOUT_MIN_DAYS` | `60` | holdout 最少天数 |
| lift `se_mult` | `1.0` | 准入的 SE 倍数 |
| forward `se_mult` | `1.645` | 向前复审的 SE 倍数（单侧 95%） |
| forward `min_days` | `60` | 向前确认最少天数 |
| CV 默认 | `train=250, test=40, purge=5, embargo=0` | lift 交叉验证 |
| `DEFAULT_TOP_M` / `HORIZON` / `BLOCK_DAYS` | `10` / `5` / `20` | 组合规模 / 预测期 / 分块 |

---

## 护栏与准入的关系

护栏**不再是入库的硬门**。当前结构：

```text
硬门      仅数据质量（覆盖率、退化截面等）
排序信号  IC / IR / DSR / PBO / holdout  → 决定先花算力测谁
最终裁决  lift 增量检验                   → 决定收不收
```

这不是放松，而是把严格性挪到了更有判别力的位置。一个 DSR 极显著但与在库因子相关 0.9 的候选，在旧结构下会入库并稀释因子库，在新结构下会被 lift 拒掉。反过来，一个单因子指标平庸但方向独特的候选，在旧结构下会被误杀。

详见[因子库与增量准入](factor-library.md)。

---

## 退化情形的守卫

统计方法在退化输入上会给出**看似正常实则无意义**的数字，这类失效尤其危险，因为它不报错。平台对以下情形有显式守卫：

| 退化情形 | 后果 | 守卫 |
|---|---|---|
| 单股票 / 全 NaN 截面 | 秩相关无定义 | 显式跳过 |
| n = 2 的秩相关 | 恒为 ±1 | 最小样本量守卫 |
| n < 分组数 | 分组空腿变裸空头 | 分组数守卫 |
| 近常数序列 | `E[x²] − E[x]²` 微负，开方 NaN 穿透 | 数值下限钳制 |
| 稀薄截面 | IC 噪声极大但不报错 | 覆盖率告警 |

> ⚠️ **polars 语义陷阱**：NaN ≠ null。聚合会跳过 null 但被 NaN 传染，`rank` 把 NaN 排最大，`NaN > x` 为 True。截面计算前默认 `fill_nan(None)`。

---

## 命令

```bash
# 对已注册因子跑防过拟合验收（DSR + bootstrap IC CI，仅打印）
pixi run fz validate overfit <factor> --start 20200101 --end 20231231

# 准入规则的空假设校准（蒙特卡洛，无 --market）
pixi run fz factor-library lift-null --n-sims 1000 --seed 42
```

参数见 [CLI 参考](../reference/cli.md)。

---

## 相关阅读

- [设计铁律](design-principles.md) —— 护栏咬合作为平台原则
- [因子库与增量准入](factor-library.md) —— 最终裁决如何工作
- [因子挖掘](../guides/mining.md) —— 护栏在挖掘流程中的位置
