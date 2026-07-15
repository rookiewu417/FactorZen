# 日内特征（Intraday Features）

A 股 1 分钟行情 → 日内微观结构特征 → 日频因子叶子（`i_*`）→ 进入挖掘、组合与无人值守链路。

## 概述

分钟 bar 在交易日内刻画波动、量价结构与路径形态；v1 电池将这些信息**聚合为每个交易日一个标量**，命名以 `i_` 为前缀，作为日频叶子接入既有因子表达式与评估流水线。

- **输入**：A 股 1min 行情湖（canonical session）。
- **输出**：日频面板（`ts_code` × `trade_date` × 17 个 `i_*` 列），可增量 build。
- **消费**：挖掘搜索、研究跑批、live 信号物化（表达式含 `i_*` 时自动 attach 面板）。

## 数据口径

口径单一真源见 `src/factorzen/intraday/sessions.py`，摘要如下：

| 项 | 约定 |
| --- | --- |
| bar 标签 | **bar-end**：标签 `t` 覆盖 `(t−1min, t]` |
| 竞价 | 含 **09:30 开盘竞价 bar**；**15:00 含收盘集合竞价** |
| 盘后 | `>15:00` 一律 drop（北交所盘后 bar 不纳入） |
| 单位 | 分钟 `vol`=股、`amount`=元（与日线手/千元不同） |
| 覆盖 | 2017 全年、2018 约十个月、**2019 整年缺失**、2020–2025 全、2026 至约 04-10 |

默认重采样频率为 **5min**（全日 48 桶）。

## 特征叶子清单（v1，17 个）

| 叶子 | 语义 |
| --- | --- |
| `i_rv` | 日内已实现波动率 |
| `i_rskew` | 日内收益偏度 |
| `i_rkurt` | 日内收益峰度 |
| `i_downvol_ratio` | 下行波动占比 |
| `i_updown_vol` | 上下行波动比的对数 |
| `i_ret_open30` | 开盘约 30 分钟收益 |
| `i_ret_close30` | 收盘约 30 分钟收益 |
| `i_ret_mid` | 中间时段收益 |
| `i_vwap_dev` | 收盘相对全日 VWAP 偏离 |
| `i_pv_corr` | 价量 Pearson 相关 |
| `i_smart_money` | 高冲击桶 VWAP 相对全日 VWAP |
| `i_vol_open30_share` | 开盘约 30 分钟成交量占比 |
| `i_vol_close30_share` | 收盘约 30 分钟成交量占比 |
| `i_vol_entropy` | 成交量时间分布归一化熵 |
| `i_amihud` | Amihud 非流动性 |
| `i_path_eff` | 价格路径效率 |
| `i_max_ret_share` | 最大单桶绝对收益占比 |

叶子在 **t 日收盘后** 可得，PIT 安全；已是日频标量，可与日线算子直接组合。

## 用法

### 1. 物化面板

```bash
# 增量 build（默认 5min；已有同 battery 分区不强制重写）
fz data intraday-features build --start 20200101 --end 20260410 [--freq 5min]

# 查看 manifest / 分区状态
fz data intraday-features status [--freq 5min]
```

### 2. 挖掘与研究

```bash
# 搜索时启用 i_* 叶子（需面板已 build）
fz mine search --intraday-leaves ...

# 研究跑批透传
fz research run --intraday-leaves ...
```

### 3. 无人值守 ops

在 ops YAML 中开启后，每日链路在 **data / audit 之后、signal 之前** 增量 build 当日窗口面板，再出信号：

```yaml
intraday_leaves: true
intraday_freq: "5min"   # 可选，默认 5min
```

默认 `intraday_leaves: false` 时该阶段 no-op（返回 skipped），既有 ops 行为不变。

live step / forward_track 侧：若因子表达式含 `i_*`，物化时会自动 attach 日内面板（需面板已存在）。

## 限制（诚实标注）

- **v1 手工电池**：固定 17 个特征，非自动生成。
- **默认 5min**：其它频率可用，但部分叶子在粗频（如 30min）上样本不足可能恒为 null。
- **分钟特征 → 日频调仓**：叶子按日聚合；**不是**日内调仓/盘中再平衡。
- **表达式特征待做**：二期才支持把 `i_*` 以外的分钟级自定义表达式进电池。
- **覆盖缺口**：2019 整年分钟源缺失；2026 仅到约 4 月，以湖内实际数据为准。
