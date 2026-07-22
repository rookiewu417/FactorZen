# 快速上手：跑通核心闭环

> [FactorZen](../../README.md) · [文档](../README.md) · **快速上手**

本页只做一件事——把平台**最核心的那条线**跑通：

```text
挖掘候选  →  相对现有因子库做增量裁决（lift）  →  只有过关的进库  →  从库里消费因子做组合
```

其它平台的教程会带你跑「评估一个因子」。FactorZen 的教程从这里开始，因为**准入才是这个平台的主张**：一个因子好不好不重要，它对你已有的库还有没有增量才重要。想先理解为什么这样设计，读[因子库与增量准入](../concepts/factor-library.md)。

**前提**：已完成[安装与环境](installation.md)，`.env` 里填好 `TUSHARE_TOKEN`，`pixi run smoke` 通过。全程不需要 LLM。

---

## 第 0 步：把数据拉下来

```bash
pixi run fz data fetch daily --start 20200101 --end 20241231
pixi run fz data fetch daily-basic --start 20200101 --end 20241231
```

**你会看到**：按月分区拉取的进度。首次全 A 五年数据受 Tushare 限速影响，约需十几到几十分钟；已有分区会跳过，可以中断后重跑。

**产物**：`data/raw/daily/year=YYYY/month=MM/data.parquet`（每日指标同理落 `data/raw/daily_basic/`）。

**怎么判断这步做对了**：

```bash
pixi run smoke-data --skip-tushare
```

`[审计:daily] OK` 且行数量级合理（全 A 五年约百万行级）即可。

> ℹ️ 数据只落 `data/`，研究产出只落 `workspace/`，这条边界是硬约定。详见[产物布局](../reference/artifacts.md)。

---

## 第 1 步：挖一批候选因子

```bash
pixi run fz mine search \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --method genetic --trials 200 --top-k 10 --seed 42
```

**你会看到**：搜索进度与收尾的候选排行。窗口会自动按 `--set holdout_ratio=`（默认 0.2）切出一段**永久隔离的 holdout**，挖掘过程碰不到它——这段是后面做准入裁决的证据来源。

**产物**：`workspace/mining_sessions/session_42_genetic/`，里面是 `candidates.csv`（头部候选与各项指标）+ `manifest.json`（seed、试验数、holdout 起点、全部候选详情）。

> ⚠️ **session 目录名是 `session_{seed}_{method}`，不带时间戳。** 同 seed 同方法重跑会**原地覆盖**上一次的产物。要保留多次实验请换 `--seed`。

**这一步已经发生了第一次准入**：护栏 `passed` 的候选在收尾时自动 upsert 进因子库（这是**入库第一通道**；`--set no_library=true` 可关）。

**怎么判断这步做对了**：

```bash
pixi run fz mine leaderboard workspace/mining_sessions/session_42_genetic --all
```

默认只列 `passed` 的候选，`--all` 才把被护栏挡下的也列出来。

---

## 第 2 步：看看灰区里有什么

不是所有有价值的因子都能过单因子门槛。一个单因子指标平平、但方向与库内因子完全不同的候选，很可能**对组合有真实增量**——直接按单因子门槛丢掉它就是浪费。

平台把这类候选标进 **lift 队列**（灰区），等着走**入库第二通道**：组合增量裁决。

session 的 `manifest.json` 里 `n_gray_zone` 就是这次挖掘攒下的灰区候选数。同时看一眼现在库里有什么：

```bash
pixi run fz factor-library list --market ashare
```

**你会看到**：库内每条因子的 rank / expression / holdout_ic / status。`status` 是四态之一：`active`（增量显著且后半段确认）、`probation`（过门槛但待向前确认）、`correlated`（与在库因子高度相关，打标收录）、`no_lift`（无增量）。

---

## 第 3 步：增量准入 —— 平台的核心一步

这一步问的是：**把这个候选加进现有基线组合，样本外表现有没有统计显著的提升？**

做法是配对比较——同一折、同一 universe、同一窗口下，「仅基线」与「基线 + 候选」两个组合直接相减，共同的市场噪声被抵消，剩下的才是候选的净贡献。裁决门槛是 `bar = max(threshold, se_mult × lift_se)`：绝对阈值挡幅度太小的，SE 倍数挡噪声太大的，两道都要过。

### 先 dry-run，看裁决

```bash
pixi run fz factor-library lift-test \
  --session workspace/mining_sessions/session_42_genetic \
  --market ashare \
  --start 20200101 --end 20241231 \
  --universe csi500
```

**你会看到**：逐候选打印 `lift` / `lift_se` / 门槛 / 裁决结果。默认按 `|residual_ic_train|` 取 top-20 控成本（`--set top_m=0` 全测），发生截断时会在 stderr 大声打印。

**产物**：**什么都不写。**

> ⚠️ **`lift-test` 默认就是 dry-run。** 不加 `--apply` 时既不写因子库，也不把 lift 拒绝回灌实验登记簿。上面这条命令跑完，库里一条记录都不会变。（`forward-review` 同样默认 dry-run。）

### 确认后，写库

```bash
pixi run fz factor-library lift-test \
  --session workspace/mining_sessions/session_42_genetic \
  --market ashare \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --apply
```

**产物**：更新 `workspace/factor_library/ashare.jsonl`（新记录 `status=probation`），并把 lift 拒绝写回该 session 所属的实验登记簿 `experiment_index.jsonl`——**被拒的信息也是资产**，下次挖掘会避开同一方向。

> ℹ️ 登记簿路径优先取 session `manifest.json` 里记录的 `index_path`；没有记录时回退到 **session 目录的父目录**下的 `experiment_index.jsonl`。本例即 `workspace/mining_sessions/experiment_index.jsonl`；`fz mine team` 的 session 则按其 `--set index_path=`（默认 `workspace/mine_team/experiment_index.jsonl`）。

> ℹ️ 通过 lift 的候选**默认封顶在 `probation` 而不是直接 `active`**。要转正得靠真实时间累积证据：`fz factor-library forward-track` 逐日记录 paper forward RankIC，攒够天数后 `fz factor-library forward-review --apply` 裁决晋升或降级。完整生命周期见[因子库与增量准入](../concepts/factor-library.md)。

**怎么判断这步做对了**：

```bash
pixi run fz factor-library list --market ashare
```

对比第 2 步的输出，看有没有新记录、状态是不是 `probation`。

> ✅ **一条真实的结论：全过也是结论，全拒也是结论。** 库积累到一定规模后，「本轮 0 个候选通过」是常态而非故障——它说明这批候选相对现有库没有增量。这正是这套机制要告诉你的事。

---

## 第 4 步：从库里消费因子做组合

因子库不是一张只进不出的清单，它有正式的消费出口：

```bash
pixi run fz combine from-library \
  --market ashare \
  --statuses active,probation \
  --start 20200101 --end 20241231 \
  --universe csi500 \
  --horizon 5
```

**你会看到**：按 `--statuses` 选品、按 `--decorr-threshold`（默认 0.7）贪心去相关后，四种合成方法在**同一套样本外切分**上的横向对比：等权 / IC 加权 / max_ir / LightGBM。切分带 purge（默认 5 天）防止前向收益重叠泄漏。

**产物**：`workspace/combinations/<run_id>/`，含四份 `combined_*.parquet`、横向对比 `comparison.csv`、LightGBM 重要度 `importance.csv`、`report.md` 与 `manifest.json`。

> ⚠️ 本子命令**没有 `--all`**（那是 `from-session` 才有的）。放宽选品范围请改 `--statuses`；默认只取 `active`。

> ⚠️ 库里含手写 python 因子时 **`--universe` 必填**。

**怎么判断这步做对了**：run 目录里存在 `manifest.json` 才算真跑完。只有 `input_manifest.json` 而没有 `combined_*.parquet` 的目录，是准备好输入但尚未合成的半成品。

---

## 闭环跑完了，你手上有什么

| 产物 | 位置 | 意义 |
|---|---|---|
| 挖掘 session | `workspace/mining_sessions/session_42_genetic/` | 候选与全部指标，可复现 |
| 因子库 | `workspace/factor_library/ashare.jsonl` | **唯一登记簿**：每条记录带 lift 证据、评估窗口、CV 参数、阈值、基线 hash |
| 实验登记簿 | `experiment_index.jsonl`（见上方路径说明） | 被拒的方向，供后续挖掘避开 |
| 组合实验 | `workspace/combinations/<run_id>/` | 四方法样本外对比 |

每一份都带 `manifest.json`，记着配置、命令、`git_sha`、seed、窗口与 universe。

---

## 下一步

- [端到端教程](end-to-end-tutorial.md) —— 接着往下走：风险模型 → 组合优化 → 模拟交易 → 报告
- [因子库与增量准入](../concepts/factor-library.md) —— lift 的完整规则、四态状态机、向前确认
- [因子挖掘指南](../guides/mining.md) —— LLM 单 Agent 与 4 角色团队挖掘、日内叶子
- [多因子组合](../guides/combination.md) —— 四方法对比的细节与选品策略
- [CLI 参考](../reference/cli.md) —— 全部命令与参数
