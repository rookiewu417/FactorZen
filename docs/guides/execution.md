# 模拟与向前执行

> [FactorZen](../../README.md) · [文档](../README.md) · **模拟与向前执行**

组合优化产出的是**目标权重**。要知道这套权重值不值钱，还得回答两个问题：

1. 在历史上按这套权重交易，净值曲线长什么样？ —— `fz sim run`
2. 如果从今天开始逐日执行，实际能拿到多少、又漏掉了多少？ —— `fz live`

前者是**回测**，一次性跑完整段历史；后者是**向前执行**，有状态、可续跑、逐日推进，且能把「理想 vs 实际」的缺口拆开归因。

参数全表见 [CLI 参考](../reference/cli.md#fz-sim)。目标权重从哪来见[风险与组合优化](risk-and-portfolio.md)。

---

## 实盘状态：先说清楚

> ⚠️ **`fz live` 全家桶跑的全部是纸面撮合。**
>
> - `BrokerAdapter`（`execution/broker.py:66`）定义了 4 个方法：`get_positions` / `get_cash` / `place_orders` / `poll_fills`。
> - **唯一实现是 `PaperBroker`**（`execution/brokers/paper.py:23`）。`--broker` 参数当前只有 `paper` 一个取值。
> - **实盘下单没有任何实现代码**。仓库里 `xtquant` / `miniQMT` 只出现在 `broker.py` 的注释里。
>
> 但这**不等于实盘不在路线上**。数据类的字段是照 xtquant/miniQMT 模型逐条设计的，注释里标了映射关系（`broker.py:26-35`）：
>
> | FactorZen 字段 | xtquant 对应 |
> |---|---|
> | `Position.volume` | `volume` |
> | `Position.can_use_volume` | `can_use_volume`（T+1 冻结） |
> | `Position.avg_cost` | `open_price` |
> | `Cash.available` | `cash` |
> | `Cash.total_asset` | `total_asset` |
> | `Cash.market_value` | `market_value` |
>
> 设计目标写在 `broker.py:3`：**「实盘期零改动映射」**。也就是说接口形状已经按券商 API 预留好了，接实盘是**分阶段推进的路线目标**，落地方式是新增一个 `BrokerAdapter` 实现，上层 `engine.step` 不需要动（它「只调 4 个方法，不知后端真假」）。
>
> 在此之前，任何把 `fz live` 的输出当作实盘业绩的说法都是错的。

---

## 日期格式的唯一例外

> ⚠️ **`fz live replay` 的 `--from-date` / `--to-date` 用带横杠的 `YYYY-MM-DD`**（如 `2024-06-01`），而**本 CLI 其余所有日期参数都是紧凑的 `YYYYMMDD`**——包括同一条命令里的 `--start` / `--end`。
>
> ```bash
> pixi run -- fz live replay --session-dir workspace/live/s2 \
>   --portfolio-run-dir workspace/portfolios/20241231 \
>   --start 20240101 --end 20241231 \
>   --from-date 2024-06-01 --to-date 2024-12-31
> #        ↑ YYYYMMDD          ↑ YYYY-MM-DD
> ```
>
> 原因是这两个参数直接喂给 `date.fromisoformat()`（`cli/main.py:2537-2538`），没有做格式规整。传成 `20240601` 会抛 `ValueError`。

---

## 模拟交易：`fz sim run`

### 做什么

读一批组合产物目录，把 `signal_date → 目标权重` 的映射喂进日频回测引擎，产出净值与绩效指标。

```bash
pixi run -- fz sim run --portfolio-dir workspace/portfolios \
  --start 20200101 --end 20241231
```

> ⚠️ **`--portfolio-dir` 在这里是「根目录」**（`workspace/portfolios/`），命令会遍历其下每个 `{run_id}/` 子目录组成调仓日程。而 `fz report portfolio` 的同名参数指的是**单个 run 目录**。同名异义，是最常见的传参错误。

**产物**落 `workspace/sim/<run_id>/`（`--run-id` 缺省时目录名为 `sim`）：

| 文件 | 内容 |
|---|---|
| `nav.parquet` | `trade_date` / `gross_return` / `cost` / `borrow_cost` / `net_return` / `nav` / `cash_weight` |
| `metrics.json` | `ann_ret` / `ann_vol` / `sharpe` / `max_dd` / `avg_turnover` / `total_cost` / `ann_turnover` |
| `manifest.json` | `n_signals` / `n_exec_dates` / `inputs` / `cost_model` / `config` / `git_sha` |

用 `fz sim show --sim-dir <目录>` 快速打印指标。

### 哪些组合目录会被跳过（而且是静默的）

`sim/engine.py:92` 的 `_load_weights_by_date` 有三条过滤，每条都会打 warning——**跑完请看日志，不要只看最终 Sharpe**：

| 情况 | 行为 |
|---|---|
| 目录缺 `manifest.json` | 跳过（`portfolio build` 先写 weights 后写 manifest，崩溃会留半成品目录） |
| manifest 缺 `signal_date` | 跳过，无法作为有效信号执行 |
| manifest 的 `status` 不在 `{optimal, optimal_inaccurate}` | **跳过** |

第三条特别重要：组合优化 infeasible / unbounded / error 时，`portfolio_build` 会把全零权重兜底写盘（为了让 `weights.parquet` 总是可写）。**那不是「清仓信号」**，sim 必须拒绝执行它。manifest 完全没有 `status` 字段的历史产物视为有效，保持向后兼容。

还有一条运行期告警：某个 `signal_date` **晚于或等于回测末日**时，该信号永远不会产生调仓。典型场景是「每天 build 完立刻 sim，但行情数据还没更新到位」——此时整体 nav 非空、看起来一切正常，但最新的信号被悄悄忽略了。

### 成本与暴露口径

- **默认不是零成本**。`cost_model=None` 时用项目默认费率的 `CostModel()`（佣金 + 滑点 + 印花税）。要零成本对照必须显式构造 `CostModel(commission=0, stamp_tax=0, slippage=0)`。
- **不做二次暴露校验**。sim 把 `max_gross_exposure` / `max_abs_weight` 放到 `inf`（`sim/engine.py:230-233`），因为权重已经受过优化器自身约束；再套 daily-research 的默认上限（gross 2.0 / 单票 1.0）会让杠杆或多空组合直接 `ValueError` 崩掉整批模拟。NaN / inf / 重复 `ts_code` 的数据损坏防线仍然保留。
- **PIT ST 阈值**：全程复用一份 `build_is_st_by_date`，ST 股票的涨跌停阈值收窄到 4.8%（主板非 ST 是 9.8%）。

---

## 向前执行：`fz live`

### 会话模型

一个「会话」由 `--session-dir` 标识，落盘在该目录下（默认约定 `workspace/execution/<session_id>/`）：

| 文件 | 内容 |
|---|---|
| `manifest.json` | `broker` / `initial_cash` / `slippage_bps` / `seed` / `command` / `git_sha` |
| `ledger.parquet` | 逐日 `as_of_date` / `nav_before` / `nav_after` / `payload`（JSON：orders + acks + fills） |
| `nav.parquet` | `as_of_date` / `nav_after`，净值序列 |
| `state.json` | 可续跑态：`cash` / `pos` / `order_seq` / `last_price` / `_last_as_of` |
| `attribution.json` | `fz live report` 产出的分歧归因 |

五个子命令：

| 命令 | 作用 |
|---|---|
| `fz live init` | 建会话，写 manifest（`--initial-cash` / `--slippage-bps` / `--broker`） |
| `fz live replay` | 在历史窗口上一次性重放出向前 NAV |
| `fz live step` | 推进**一个**交易日，供每日调度调用 |
| `fz live status` | 打印末记录日 / 现金 / 持仓数 |
| `fz live report` | 生成 A 类分歧归因报告 |

> ℹ️ `fz live replay` 可以直接在 `fz live init` 建好的会话上跑——`SessionStore.init` 检测到 manifest 已存在就**不覆盖**（`store.py:26-27`），否则 init 设的 `slippage_bps` / `initial_cash` 会被 replay 的默认 config 静默清掉。

### 一步推进做了什么

`execution/engine.py:32` 的 `step`，broker 无关，只调 `BrokerAdapter` 的 4 个方法：

```text
1. broker.get_positions()          查当前持仓
2. nav_before = get_cash().total_asset
3. 目标权重 → 目标股数：round_lot(w × nav_before / ref_price)
4. build_orders(target, positions)  差额单，先卖后买（先腾现金）
5. broker.place_orders(orders)      → OrderAck 列表
6. broker.poll_fills()              → Fill 列表
7. nav_after = get_cash().total_asset
```

两个 PIT 细节：

- **执行定量参考价用 `pre_close`**，不用当日 `close`（`drivers.py:40-47`）。收盘价要收盘才有，用它定量就是前视。
- **信号次一交易日才执行**：只取 `signal_date < as_of` 的最新一次权重（`drivers.py:92`、`drivers.py:169`）。`signal_date` 是组合建仓的数据截止日（已经用了当日收盘），用 `s <= d` 会在信号当日开盘就按当日收盘算出的权重成交 = 未来函数。这一口径与 `fz sim run` 对齐。
- 「有适用信号但目标权重为空」（risk-off 全现金）**仍会正常 step 以清仓**，只有真的没有任何适用信号才跳过。

`round_lot`（`broker.py:13`）向零取整到 100 股整手，带 `+1e-6` 手的容差——吸收「权重 → 股数 → 权重」往返的浮点误差，否则 12900 整手在往返后常变成 12899.999999999998 被砍掉一整手。

### 纸面撮合的真实约束

`PaperBroker._exec_one`（`paper.py:99`）按顺序过五道：

1. **无行情** → 拒单 `missing_price`
2. **共享约束内核**（`apply_trade_constraints`，与回测同一份实现）——在权重空间判：
   - `suspended`（`vol == 0`）
   - `limit_up`（买单撞涨停）/ `limit_down`（卖单撞跌停），ST 阈值 4.8% vs 主板 9.8%
   - `capacity`（相对 ADV 的容量上限，ADV 由 trailing 20 日成交额均值 shift(1) 得到，无未来函数）
   - `invalid_portfolio_value`
3. **整手取整** → 部分成交时标 `lot_round`
4. **现金 / 持仓约束**：
   - 买单现金不足 → 截断，标 `insufficient_cash`
   - 卖单 `can_use_volume == 0` → 拒单 `t1_frozen`；部分可卖 → 截断标 `t1_frozen`
5. **计成本落账**：买单用 `one_way_cost()`，卖单用 `sell_cost()`（含印花税）

执行价是 `open × (1 ± slippage_bps)`，买加卖减。

**T+1 的实现**：`advance_to` 进入新交易日时把全部持仓 `can_use_volume` 解冻为 `volume`；当日买入不计入可卖（`_apply_fill` 只加 `volume` 不加 `can_use_volume`）。

> ℹ️ **停牌日不会让 NAV 塌陷。** `PaperBroker` 维护一份 `_last_price`（最近已知收盘价），当日无行情的标的按它估值而不是按 0。否则 `get_cash().total_asset` 被低估，`engine.step` 会拿错误的 `nav_before` 去误卖其他正常持仓。这份价格随 `state.json` 持久化，续跑后仍可用。

### replay 与 step 的分工

| | `fz live replay` | `fz live step` |
|---|---|---|
| 粒度 | 单进程循环整段历史交易日 | 一次一个交易日 |
| 场景 | 回看、扩窗、崩溃后重建 | 每日调度（cron / 无人值守链路） |
| 状态 | 进程内连续 + 落盘 | 每次进程重启，靠 `load_state` 续跑 |

两者共用同一份 `SessionStore`，可以混着用（先 replay 补历史，再每日 step 往前走）。

### 续跑与幂等的四道守卫

`run_daily_step`（`drivers.py:115`）为了扛住「每天起一个新进程」的调度模式，加了四道防线：

| 守卫 | 行为 | 防的是什么 |
|---|---|---|
| **幂等哨兵** | `store.has_date(as_of)` 命中 → 直接返回 `skipped` | 重复下单、ledger 追加重复行 |
| **交易日历守卫** | `as_of` 不在 `daily` 的交易日里 → 跳过**且不落盘** | 非交易日照常 step 会落一条纯现金塌陷的 nav 行，还被 `has_date` 永久锁死无法修复 |
| **崩溃恢复一致性** | `state._last_as_of` 与 ledger 末行日期不符 → **抛 `RuntimeError` 要求重建会话** | 「写完 ledger、没写完 state 就崩」导致的账实分叉。宁可报错也不静默用错状态续跑 |
| **日期单调性** | `as_of <= _last_as_of` → 跳过，`reason="stale_as_of"` | 乱序补跑：用「未来的」broker 状态去步进过去的日期，ledger 乱序、state 被污染 |

`SessionStore.append`（`store.py:38`）对三个文件都做 **tmp + `os.replace` 原子替换**，写到一半崩溃不会留下损坏的 parquet/json。

> ⚠️ **`fz live step` 的 `--start` 要往前留足回看天数。** 容量约束用的 ADV 是 trailing 20 日均值，只给 `--date` 当天会让 ADV 为空、容量约束静默失效（`daily` 缺 `amount` 列时 `_precompute_adv_20d_by_date` 优雅降级返回 `{}`，不报错）。建议至少覆盖 `--date` 前 20 个交易日。

`fz live replay` 也有幂等：重跑同一 `session_dir` 时 `has_date` 命中的日子直接跳过。resume 时会先 `broker.load_state()` 重建状态，否则跳过已落盘日之后 broker 还停在空仓 `initial_cash`，续跑日的 NAV 全错。

---

## A 类分歧归因：`fz live report`

### 回答什么问题

「理论上这套权重能赚多少，实际执行只拿到多少，差额去哪了？」

做法是造一个 **frictionless 孪生**（`attribution.py:41`）：同一批目标权重、同一段执行窗口，但按 `close` 全额成交、零成本、无容量/整手/现金/T+1 约束。两条 NAV 的年化收益之差就是总缺口。

```text
total_gap = ideal.ann_ret − real.ann_ret
residual  = total_gap_bps − cost_bps − slippage_bps
```

### 报告字段

`attribution.json`：

| 字段 | 含义 |
|---|---|
| `ideal` / `real` | 各自的 `ann_ret` / `sharpe` / `max_dd` |
| `total_gap_ann_ret` | 年化收益缺口 |
| `cost_bps` | 累计佣金+印花+滑点成本，折算成**年化 bps** |
| `slippage_bps` | 逐笔滑点 = `filled × (成交价 − close) × side`，年化 bps |
| `residual_bps` | `total_gap − cost − slippage`，**余量，不强制为 0** |
| `ann_turnover` | 年化双边换手 = 累计成交额 / 平均 NAV / 年数 |
| `n_fills` / `n_days` | 成交笔数 / 执行天数 |
| `missed_by_reason` | 按拒单原因分组的 `{count, notional}` |

### 三条设计上的诚实

1. **总缺口独立测量。** 不是把各桶加起来当总数，而是两条 NAV 各自算年化收益再相减。桶（成本、滑点）逐笔精确算，**剩下的进 `residual`，不做配平**。residual 大说明有未建模的分歧来源，这是要看见的信息，不是要抹平的误差。

2. **年化口径必须对齐。** `ideal` / `real` 的 `ann_ret` 是「日均收益 × 252」，所以 `cost_sum` / `slip_sum` 这些整段累计的美元成本也要乘 `252 / n_days` 折成年化 bps 才能相减。短窗口尤其明显——把几天的一次性成本外推成一整年会被放大 `252/n_days` 倍，两边不做同一折算就没有可比性（`attribution.py:152-161`）。

3. **部分成交也算踏空。** 归因不只看 `accepted=False` 的拒单。容量/现金/整手/T+1 截断的单子 `accepted=True` 但 `filled < volume`，这个缺口同样按 `shortfall × close` 归到对应 reason 下。只统计拒单会漏掉一大半（`attribution.py:129-131`）。

**两条不静默的告警**：

- `daily` 窗口未覆盖某个 `exec_date`（窗口过窄或真实停牌无行）→ 打 warning 列出日期。该日的 ideal 与桶归因会静默按 0 处理、结果失真，所以先告警再尽力算，不阻断。
- 旧 payload 无 `acks` 或数量与 `orders` 对不上 → 跳过该行的 miss 归因（成本/滑点已从 fills 算完不受影响），而不是让 `zip(strict=True)` 抛 `ValueError` 崩掉整个报告。

### 运行

```bash
pixi run -- fz live report --session-dir workspace/live/s1 \
  --portfolio-run-dir workspace/portfolios/20241231 \
  --start 20240101 --end 20241231
```

`initial_cash` 从会话 manifest 的 `config` 里读，不用重复传。

输出形如：

```text
[live] 归因: 总缺口=xxx.xbps/年 成本=xx.x 滑点=xx.x residual=xx.x | 年化换手(双边)=x.xx 成交=NNN笔
        未成交[limit_up]: N次 名义额=NNNNNN
        未成交[t1_frozen]: N次 名义额=NNNNNN
```

---

## 需要成对修改的路径

> ⚠️ `fz sim run`（`sim/engine.py`）与 `fz live step`（`execution/drivers.py`）是**配对的双路径**，共享两件事：**信号执行时点**（都必须是次一交易日）与**成本口径**。改任一侧必须检查另一侧。

两侧已收敛的部分：约束判定走同一个 `apply_trade_constraints` 内核，回测走同一个日环引擎（`daily/evaluation/backtest.py`），`PaperBroker` 用的是该内核的标量包装。

历史教训见[架构](../concepts/architecture.md#需要成对修改的路径)。

---

## 典型流程

```bash
# 0) 先有目标权重（见「风险与组合优化」）
#    workspace/portfolios/20241231/ 等若干 run 目录

# 1) 历史回测：一次跑完
pixi run -- fz sim run --portfolio-dir workspace/portfolios \
  --start 20200101 --end 20241231
pixi run -- fz sim show --sim-dir workspace/sim/sim

# 2) 向前执行：建会话
pixi run -- fz live init --session-dir workspace/execution/s1 \
  --initial-cash 1000000 --slippage-bps 5

# 3a) 一次性重放历史窗口
pixi run -- fz live replay --session-dir workspace/execution/s1 \
  --portfolio-run-dir workspace/portfolios/20241231 \
  --start 20240101 --end 20241231

# 3b) 或每日推进一步（可续跑、幂等）
pixi run -- fz live step --session-dir workspace/execution/s1 \
  --date 20241231 --portfolio-run-dir workspace/portfolios/20241231 \
  --start 20241101 --end 20241231     # start 留足 ADV 回看

# 4) 看状态
pixi run -- fz live status --session-dir workspace/execution/s1

# 5) 分歧归因：理想 vs 实际，差额去哪了
pixi run -- fz live report --session-dir workspace/execution/s1 \
  --portfolio-run-dir workspace/portfolios/20241231 \
  --start 20240101 --end 20241231
```

> ℹ️ 多个组合目录用**重复旗标**给出：`--portfolio-run-dir A --portfolio-run-dir B`。不是逗号分隔，也不是空格分隔多值（那是 `fz factor-library lift-test --session` 的风格）。

> ✅ 把第 3b 步接进每日调度就是无人值守链路的 `live_step` 阶段，见[无人值守运营](operations.md)。

---

## 相关阅读

- [风险与组合优化](risk-and-portfolio.md) —— 目标权重从哪来
- [无人值守运营](operations.md) —— 把逐日推进接进 8 阶段日链路
- [架构](../concepts/architecture.md) —— sim 与 execution 的双路径关系
- [CLI 参考](../reference/cli.md#fz-live) —— `fz sim` / `fz live` 参数全表
- [产物布局](../reference/artifacts.md) —— `workspace/sim/` 与 `workspace/execution/` 字段
