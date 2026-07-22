# 无人值守运营

> [FactorZen](../../README.md) · [文档](../README.md) · **无人值守运营**

`fz ops daily` 把「一个 A 股交易日该做的事」串成固定 8 阶段：守卫 → 取数 → 审计 → 日内特征 → 信号 → 执行 → 报告 → 发布。逐阶段幂等、失败即告警、调度完全外置。

一条 cron 每天叫它一次，剩下的它自己管。读完能写好 `ops.yaml`、理解状态与退出码，并把日链路挂上定时触发。

参数见 [CLI 参考](../reference/cli.md#fz-ops)；部署方式（systemd timer / Docker）见[部署](deployment.md)。

---

## 8 个阶段

单一真源是 `ops/runner.py` 的 `STAGES` 列表，**顺序固定、不可配置**：

| # | 阶段 | 做什么 | 可跳过 |
|---|---|---|---|
| 1 | `guard` | 查交易日历，非交易日整链短路 | — |
| 2 | `data` | 补齐日线 / 复权因子 / 估值 / 基准指数 | — |
| 3 | `audit` | 数据质量门，按 `audit_fail_on` 级别拦截 | — |
| 4 | `intraday_features` | 增量 build 日内特征面板 | `intraday_leaves: false` 时 no-op |
| 5 | `signal` | 执行外部信号命令（重建组合） | `signal_command` 省略时 no-op |
| 6 | `live_step` | 纸面执行推进一个交易日 | — |
| 7 | `report` | 从会话账本取当日 NAV / 成交，产出摘要文本 | — |
| 8 | `publish` | 渲染 track record 静态页 | `publish_enabled: false` 时 no-op |

每个阶段签名统一为 `(cfg, as_of, ctx) -> dict`（`ops/stages.py`），返回的摘要写进 `ctx` 供后续阶段读。阶段本身**无状态**，幂等由 runner 的 `OpsState` 与 `SessionStore.has_date` 两层保证。

### 逐阶段说明

**`guard`**（`stages.py` 的 `stage_guard`）—— 拉当日交易日历，判断 `is_open == 1`。

**`data`**（`stages.py` 的 `stage_data`）—— 调 `ensure_daily` / `ensure_adj_factor` / `ensure_daily_basic` / `ensure_index_daily`，窗口是 `[as_of − lookback_days, as_of]`。这些函数是 `strict=True` 的，缺口补不齐直接让异常冒泡，runner 统一接住。

**`audit`**（`stages.py` 的 `stage_audit`）—— 对 `audit_types` 里的每种数据跑 `build_raw_data_audit`。`audit_fail_on: error` 只拦 error；设成 `warning` 则 warning 也拦。任一被拦就抛 `OpsStageError`，链路停在这里。

**`intraday_features`**（`stages.py` 的 `stage_intraday_features`）—— 只有 `intraday_leaves: true` 才跑，在 `signal` **之前**把窗口内的 `i_*` 面板物化好，供后续信号的 `i_*` 因子 attach。`overwrite=False`，是增量 build。

**`signal`**（`stages.py` 的 `stage_signal`）—— `signal_command` 是一个命令数组，用 `subprocess.run(check=True, timeout=3600)` 执行。非零退出码 → `OpsStageError` 带 stderr 末 200 字符；超时 1 小时 → `OpsStageError`。省略该配置则跳过，直接消费已有的 portfolio 产物。

**`live_step`**（`stages.py` 的 `stage_live_step`）—— 拉行情 →（可选）universe 过滤 → 按 `portfolio_run_dirs_glob` 收组合产物 → 调 `run_daily_step` 推进一天。glob 无匹配直接报错（不静默空跑）。会话目录没有 `manifest.json` 时自动 `init` 一次。

**`report`**（`stages.py` 的 `stage_report`）—— 从 ledger 里找当日记录，算 NAV 与期间收益，拼成通知用的摘要文本。**当日无执行记录时不报错**，返回「无执行记录(空目标/跳过)」。

**`publish`**（`stages.py` 的 `stage_publish`）—— 把净值序列渲染成 `<publish_site_dir>/index.html`（Jinja2 模板，含总收益与最大回撤）。

---

## 幂等重入：状态怎么记

每个交易日一个 `<state_dir>/<YYYY-MM-DD>.json`，记录各阶段的 `{status, ts, detail}`（`ops/state.py`）。

- runner 开跑前先读这个文件，`is_done(name)` 为真的阶段**直接跳过**（`runner.py` 的 `run_ops_daily`）。
- 阶段成功 → `mark_done`；抛异常 → `mark_failed` + 告警 + `return 1`。
- 写入走「临时文件 + `os.replace`」原子替换，崩溃不会留半截 JSON。

所以重跑 `fz ops daily --date 同一天` 是安全的：**已完成的阶段跳过，从失败处续跑**。

> ⚠️ **非交易日不落 done。** `guard` 判定非交易日时，runner 短路返回 0，**但不调 `mark_done`**（`runner.py` 的 `run_ops_daily`）。这是有意的——如果把「非交易日」固化成已完成状态，将来日历修正或调休补交易日时就再也重跑不了了。重跑时会重新判断。

> ⚠️ **失败不会崩掉全链路，但会中断后续阶段。** 任意阶段抛异常都被 runner 捕获（`runner.py` 的 `run_ops_daily`），标 failed、发告警、返回退出码 1。**后面的阶段不会执行**——阶段之间有数据依赖（没取到数就没法执行、没执行就没法报告），继续跑只会产生垃圾产物。修好问题重跑，前面成功的阶段会跳过。

**退出码约定**：`0` = 成功或非交易日；`1` = 某阶段失败。调度器据此判断是否需要人工介入。

查状态：

```bash
pixi run -- fz ops status --config deploy/ops.example.yaml --date 20241231
```

直接打印那天的 state JSON，形如：

```json
{
  "guard":   {"status": "done",   "ts": "2024-12-31T18:00:03", "detail": "{'trading_day': True}"},
  "data":    {"status": "done",   "ts": "2024-12-31T18:01:22", "detail": "..."},
  "audit":   {"status": "failed", "ts": "2024-12-31T18:01:30", "detail": "[audit] 数据质量门未通过: ..."}
}
```

---

## 配置

`fz ops daily --config` 指向一份 YAML，经 `OpsConfig`（pydantic，`extra="forbid"`）校验。模板见仓库内 `deploy/ops.example.yaml`。

> ⚠️ **`extra="forbid"` 意味着字段拼错会被直接拒绝**，而不是静默忽略。这是有意的——运维配置最忌静默失配（写了 `notify_kind: webhok` 却以为告警配好了）。

| 字段 | 默认 | 说明 |
|---|---|---|
| `session_dir` | **必填** | 纸面执行会话目录，首跑自动 init |
| `portfolio_run_dirs_glob` | **必填** | 目标组合产物的 glob，如 `workspace/portfolios/prod-*` |
| `signal_command` | `None` | 信号生成命令数组；省略 = 跳过，直接消费已有产物 |
| `lookback_days` | `90` | 行情窗口天数，必须 > 0 |
| `benchmark` | `000300.SH` | 基准指数 |
| `universe` | `None` | 限制股票池；省略 = 全市场 |
| `intraday_leaves` | `false` | 是否每日增量 build 日内特征面板 |
| `intraday_freq` | `5min` | 面板 bar 频率 |
| `audit_types` | `["daily", "daily_basic"]` | 参与质量门的数据类型 |
| `audit_fail_on` | `error` | `error` \| `warning` |
| `initial_cash` | `1000000.0` | 纸面本金，必须 > 0 |
| `slippage_bps` | `0.0` | 滑点基点，必须 ≥ 0（0 = 零滑点对照） |
| `notify_kind` | `stdout` | `stdout` \| `webhook` |
| `notify_url_env` | `FACTORZEN_NOTIFY_WEBHOOK` | webhook 模式下从哪个环境变量读 URL |
| `publish_enabled` | `false` | 是否渲染 track record 页 |
| `publish_site_dir` | `workspace/ops/site` | 静态页输出目录 |
| `state_dir` | `workspace/ops/state` | 幂等状态目录 |

一份最小可用配置：

```yaml
session_dir: workspace/execution/prod
portfolio_run_dirs_glob: 'workspace/portfolios/prod-*'
lookback_days: 90
benchmark: '000300.SH'
audit_types: ['daily', 'daily_basic']
audit_fail_on: error
initial_cash: 1000000.0
slippage_bps: 5.0
notify_kind: stdout
```

> ℹ️ `lookback_days: 90` 的窗口除了给因子留 lookback，也给 `live_step` 的容量约束留 ADV 回看（trailing 20 交易日）。调小要谨慎。

---

## 通知

两个后端（`ops/notify.py` 的 `build_notifier`）：

| 后端 | 行为 |
|---|---|
| `stdout` | 打印 `[{level}] {title}\n{content}`，本地开发与无 webhook 时的默认 |
| `webhook` | POST JSON `{"title", "content", "level"}` 到 URL，`Content-Type: application/json` |

`WebhookNotifier` 用标准库 `urllib`，**零额外依赖**，兼容企业微信机器人 / PushPlus 这类接受简单 JSON POST 的端点。默认超时 10 秒，失败重试 2 次（间隔 1 秒）。

两条设计上的取舍：

> ✅ **通知失败不炸主链路。** 重试用尽后 `send` **返回 `False` 而不抛异常**。通知是旁路，推送不出去不该让整条日链路失败。

> ⚠️ **但配置错要在启动期就炸。** `notify_kind: webhook` 而 `notify_url_env` 指的环境变量没设时，`build_notifier` 直接抛 `RuntimeError`（`notify.py`）——在链路开跑前就暴露，而不是等到跑完要发日报时才发现告警全丢了。设置方式：
>
> ```bash
> export FACTORZEN_NOTIFY_WEBHOOK='https://…'
> pixi run -- fz ops daily --config workspace/configs/ops.yaml
> ```

告警时机：

- **任一阶段失败** → `[FactorZen ops] {stage} 失败 {日期}`，level=`error`
- **全链路完成** → `[FactorZen ops] 日报 {日期}`，level=`info`，内容是 `report` 阶段的摘要文本
- **非交易日短路** → 不发通知（`runner.py` 直接 return，不走末尾的日报）

---

## 真实接线缺口：向前确认还需人工跑

> ⚠️ **`fz factor-library forward-track` 尚未接进这 8 个阶段。**
>
> 因子库里 `status=probation` 的因子需要**每日**记录一条 paper forward RankIC，攒够 `--min-days`（默认 60 天）后才能由 `forward-review` 裁决转正或降级。这个每日动作**不在 `STAGES` 里**——`ops/stages.py` 完全没有引用 `forward_track`。
>
> 两个命令的 `--help` 自己就带着这条待办标注（`cli/parser.py` 的 `forward-track` / `forward-review` help）：
>
> > 「确认窗口随真实时间累积；**ops 每日链路接线为后续工作**」
>
> **当前只能人工每天跑**：
>
> ```bash
> pixi run -- fz factor-library forward-track --market ashare --universe csi500
> ```
>
> **漏跑的后果不是报错，而是静默拖长确认周期**：`forward-track` 默认拒绝历史回灌（`--max-backfill-days=10`，超期的 as_of 会被拒），所以漏掉的那些天补不回来，probation 因子的转正时间被无声推后。
>
> **临时对策**：在自己的调度里，紧挨着 `fz ops daily` 之后再排一条 `forward-track`。注意 `--universe` 必须与因子准入时的口径一致，否则 forward 证据与准入证据不可比（缺省值已按库记录众数自动对齐，一般不用手动指定）。

probation → forward → promote 的完整生命周期见[因子库与增量准入](../concepts/factor-library.md)。

---

## 运行

```bash
# 跑当天
pixi run -- fz ops daily --config deploy/ops.example.yaml

# 跑指定日（YYYYMMDD）
pixi run -- fz ops daily --config deploy/ops.example.yaml --date 20241231

# 查某天各阶段状态
pixi run -- fz ops status --config deploy/ops.example.yaml --date 20241231
```

`--date` 缺省取**今天**（本机日期，`cli/main.py` 的 `_ops_as_of()` / `_cmd_ops_daily()`）。

### 接进调度

调度完全外置——`run_ops_daily` 本身是一个可重入的编排函数，不含任何定时逻辑。仓库里备了两套现成模板：

- `deploy/systemd/factorzen-ops.service` + `factorzen-ops.timer`
- `deploy/docker/Dockerfile` + `compose.yaml`

具体接法见[部署](deployment.md)。

> ✅ **调度器该怎么用退出码**：`0` 正常（含非交易日），`1` 需要人工看。配合幂等重入，最稳的做法是让调度器在失败后按固定间隔重试同一天——修好数据源或权限后，重试会自动从失败阶段续上，不会重复执行已完成的阶段。

---

## 排查清单

| 现象 | 先看哪 |
|---|---|
| 链路每天都在第 2 步停 | `data` 阶段：`TUSHARE_TOKEN` 是否有效、当日数据是否已发布 |
| 停在 `audit` | 跑 `fz ops status` 看 detail 里的具体 data_type 与错误；必要时临时把 `audit_fail_on` 从 `warning` 放回 `error` |
| `live_step` 报「无匹配组合产物」 | `portfolio_run_dirs_glob` 写错，或 `signal_command` 没生成新的组合目录 |
| 全链路 0 退出但没有成交 | 看 `report` 摘要。常见原因：目标权重的 `signal_date` 不早于 `as_of`（信号次日才执行），或组合 manifest 的 `status` 非 optimal 被 sim/live 跳过 |
| 配了 webhook 但收不到告警 | `notify_url_env` 指的环境变量在**调度器的环境**里是否可见（cron 不继承登录 shell 的 export） |
| 重跑没反应 | 正常——该日阶段已 `done`。要强制重跑删掉 `<state_dir>/<YYYY-MM-DD>.json` 对应阶段的记录 |
| probation 因子迟迟不转正 | 见上文接线缺口，检查 `forward-track` 是否真的每天在跑 |

---

## 相关阅读

- [模拟与向前执行](execution.md) —— `live_step` 阶段背后的执行引擎与会话模型
- [因子库与增量准入](../concepts/factor-library.md) —— probation / forward 确认的完整生命周期
- [部署](deployment.md) —— systemd timer、Docker、Web 展示的部署方式
- [CLI 参考](../reference/cli.md#fz-ops) —— `fz ops daily` / `fz ops status` 参数
- [环境变量](../reference/environment.md) —— `TUSHARE_TOKEN` / `FACTORZEN_NOTIFY_WEBHOOK` 等
