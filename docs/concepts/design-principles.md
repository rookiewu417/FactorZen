# 设计铁律

> [FactorZen](../../README.md) · [文档](../README.md) · **设计铁律**

三条原则贯穿全平台。设计冲突时以它们为裁决依据——包括「为了性能牺牲一点点 PIT」这类看起来划算的交易，也一律不做。默认语境是 A 股。

---

## 一、PIT 无未来函数

**t 日的信号，只能使用 ≤t 日收盘时刻真实可得的信息。**

未来函数是量化研究中最贵的错误：它不会让回测报错，只会让回测变好看，然后在实盘里原形毕露。平台把这条约束下沉到数据层，而不是依赖研究员自觉。

具体落点：

| 环节 | 做法 | 代码位置 |
|---|---|---|
| universe 成分 | 逐日快照，不用今天的成分股回看历史 | `core/universe.py` |
| 交易约束 | 停牌 / 涨跌停 / ST / 次新 / T+1 内嵌进 universe 快照；ST 的涨跌停阈值按 PIT 收窄 | `core/universe.py` `_get_board_limit` |
| 财务数据 | 按**公告日**对齐，不用报告期末数据 | `daily/data/pit.py` |
| 执行定价 | 用 `pre_close` 而非当日收盘 | `daily/evaluation/backtest.py` |
| 滚动因子 | 扩窗预热（`expanded_start`），避免窗口起点处用到未来样本 | `daily/data/context.py` |
| 预处理统计量 | 中性化、标准化的统计量只用 ≤t 样本 | `daily/preprocessing/` |
| 时序 | t 日计算 → t+1 日执行 | 全链 |

> ⚠️ **已知例外：美股 universe 不是 PIT**（静态成分快照，存在幸存者偏差）。A 股侧的 PIT 是严格的。细节见[多市场](multi-market.md)。

---

## 二、护栏咬合

**护栏必须参与筛选，而不是「只算不判」。**

算出 DSR 却不用它卡人，等于没算——报告上多一行漂亮数字，决策上零影响。平台的护栏默认咬合进筛选流程，`passed` 参与实际过滤，`--all` 是显式的逃生口而不是默认行为。

### 准入判据的演化

平台早期用一组单因子阈值做硬门。现在的结构是：

```text
硬门（一票否决）      仅剩数据质量
      ↓
排序信号（决定先测谁） 单因子指标：IC、IR、DSR、holdout……
      ↓
最终裁决              lift 增量检验（相对现有因子库）
```

单因子门槛之所以降级，是因为它回答错了问题——详见[因子库与增量准入](factor-library.md)。它仍然有用，但用途从「决定收不收」变成了「决定先花算力测谁」。

### 多重检验记账

搜索得越多，最好的那个越可能只是运气。平台从挖掘起就记账：

- **DSR 的 N = 真实评估过的唯一表达式数**（取自求值缓存），不是最终存活集的大小。
- 迭代循环中对累积状态计数会产生三角和 over-count，必须**按轮过滤**。
- 记账结果进 Deflated Sharpe 做矫正，见[防过拟合护栏](guardrails.md)。

---

## 三、可复现

**每一条结论都要能被重新验算。**

落 `manifest.json` 只是最低要求。真正的可复现要求记录的不是结论，而是**得出结论的全部条件**。

### run 级

每次运行写 `manifest.json`：配置、命令行、`git_sha`、依赖 lock hash、工作树 dirty 状态、seed、窗口、universe、各阶段耗时、输出路径、成功/失败状态。字段全集见[产物参考](../reference/artifacts.md)。

### 因子库级

因子库记录（`FactorRecord`）除了指标，还存一整组复现条件：

| 存什么 | 为什么 |
|---|---|
| `eval_start/end`、`admission_start/end`、`scored_start/end` | 三段窗口各不相同，混淆会得出不同结论 |
| `universe`、`horizon`、`frequency`、`profile_name` | 评估口径 |
| `cv_train_days`、`cv_test_days`、`block_days` | CV 参数变了 lift 就变了 |
| `lift_threshold`、`lift_se_mult` | 当时用的是哪套阈值 |
| `baseline_hash` | **基线是哪个版本的库**——同一因子对不同基线的 lift 完全不同 |
| `git_sha`、`source_run_id`、`source_session_dir` | 代码版本与产出溯源 |

`baseline_hash` 是这组里最关键的一个。lift 是相对量，脱离基线谈 lift 没有意义；记下基线 hash，才能判断一条历史准入结论在今天是否还成立。

### 随机性

- 随机搜索、遗传算法记录 seed。
- LLM 调用记录 prompt / response 与 model id，推理路径可回放。
- `pixi.lock` 锁定完整依赖树。

---

## 这三条如何互相支撑

三条铁律不是并列的检查项，而是一条因果链：

**PIT** 保证输入是诚实的 → **护栏咬合** 保证筛选是诚实的 → **可复现** 保证结论是可被审计的。

任何一环松掉，另外两环的价值都会归零：PIT 破了，护栏卡的是被污染的数字；护栏只算不判，可复现记录的是没人用的结论；不可复现，前两者事后无从验证。

---

## 相关阅读

- [防过拟合护栏](guardrails.md) —— 护栏的具体统计方法与阈值
- [因子库与增量准入](factor-library.md) —— 最终裁决的机制
- [架构](architecture.md) —— 这些原则落在哪些模块
- [多市场适配](multi-market.md) —— 跨市场口径与 PIT 例外
