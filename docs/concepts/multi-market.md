# 多市场适配

> [FactorZen](../../README.md) · [文档](../README.md) · **多市场适配**

平台用 Ports & Adapters 结构支持多个市场：挖掘引擎、护栏、因子库、回测引擎全部共用一套实现，市场差异收敛进适配器。

---

## Port 定义

抽象基类在 `markets/base.py`，共 6 个必需 Port + 1 个可选：

| Port | 位置 | 关键方法 |
|---|---|---|
| `DataProvider` | `base.py` | `fetch_bars(symbols, start, end, freq)`、`fetch_symbol_meta()` |
| `Calendar` | `base.py` | `sessions` / `is_session` / `next_session` / `prev_session` / `periods_per_year` |
| `TradingRules` | `base.py` | `allow_short` / `settlement_lag` / `execution_price_col` / `tradable_mask` |
| `CostModel` | `base.py` | `trade_cost(side, notional, is_maker)` / `carry_cost(..., funding_rate)` |
| `Universe` | `base.py` | `snapshot(d)` / `benchmark(start, end)` |
| `FactorSet` | `base.py` | `leaf_features()` / `basic_features()` / `derived_columns(bars)` |
| `RiskModel`（可选） | `base.py` | `style_factors()` / `sector_classification(symbols, d)` |

`MarketProfile`（`base.py`）把它们组装成一个市场的完整描述。`registry.py` 提供 `register` / `get` / `list_markets`，**采用惰性构造 + 缓存**——避免 import 阶段就去建 ccxt client 这类重对象。

设计约定是「**参数化带 A 股默认值**」：新市场通过注入参数接入，A 股行为保持默认值不变。A 股零回归是接入任何新市场的底线。

---

## 四市场真实状态

| 市场 | 数据源 | 规模 | 成熟度 |
|---|---|---:|---|
| `ashare` | Tushare（经 `core/loader`） | 235 行 | **薄适配层**——真实重逻辑在 `core/` 与 `daily/`，profile 只是把既有实现包成 Port |
| `crypto` | Binance Vision 数据湖（默认）· ccxt（备用） | 1,725 行 | **最完整的 Port 实现**：唯一自带 `RiskModel`，另有自己的回测、组合、挖掘、重采样、板块分类 |
| `futures` | Tushare `fut_daily` / `fut_mapping` / `fut_basic` | 790 行 | 真实实现：主力连续拼接 + **乘法后复权**，有 ground-truth 测试；用交易日历覆盖审计判缺失 |
| `us` | **Yahoo Finance**（非 Tushare） | 713 行 | 真实实现但 MVP universe：限流 + 指数退避，按 symbol parquet 缓存 |

`markets/` 全树内 `NotImplementedError` / `TODO` / `FIXME` 数量为 **0**——四个市场都是可跑的适配器，没有存根。

---

## 能力边界不均（重要）

「四市场都能跑」指的是**挖掘链路**。下游能力的覆盖并不整齐：

| 命令 | 支持的市场 |
|---|---|
| `fz mine search/agent/team/pool-prebuild`、`fz factor-library *`、`fz validate overfit` | ashare · crypto · futures · us |
| **`fz data fetch`** | **仅 ashare · crypto** |
| **`fz portfolio build`** | **仅 ashare · crypto** |

因此准确的说法是：

- **ashare / crypto** —— 全链路可跑：取数 → 挖掘 → 准入 → 风险/组合 → 回测 → 报告。
- **futures / us** —— **只通到挖掘与因子库**。没有数据拉取子命令（数据需自行准备），没有组合优化接线，没有风险模型。

这不是「未实现的存根」，而是「实现了一段、没接通另一段」。

> ⚠️ **美股 universe 有幸存者偏差。** `markets/us/sp500_snapshot.py` 用约 2024 年的静态成分快照（约 490 支），不是 PIT 历史成分。这是[三条铁律](design-principles.md)中 PIT 那条的已知例外。

---

## 风险模型未统一

风险模型是当前 Port 化最不完整的一块：

| 市场 | 风险模型 |
|---|---|
| `ashare` | Barra 模型在**独立的 `risk/` 包**里，未接进 `MarketProfile`（profile 的 `risk=None`），由 `portfolio/` 与 `attribution/` 直接调用 |
| `crypto` | 走 Port 化的 `markets/crypto/risk.py`，是唯一 `risk` 非 None 的市场 |
| `futures` / `us` | 无 |

也就是说 A 股与 crypto 各有一套风险模型，走的还是两条不同的接线方式。统一到 Port 是待办。

---

## 频率支持

| 市场 | 支持的 bar 频率 |
|---|---|
| `ashare` | 日频 + 分钟级（1m/5m/15m/1h，经 `intraday/`） |
| `crypto` | 日频 + 分钟级（Vision 湖原生分钟数据） |
| `futures` / `us` | **仅日频**——provider 对其他频率显式抛 `ValueError` |

> ⚠️ `--freq` 参数在不同命令下有两套语义：**因子频率**（daily / weekly / monthly）与 **bar 粒度**（1m / 5m / 15m / 1h / daily）。看清所在命令，见 [CLI 参考](../reference/cli.md)。

---

## 单位口径

各市场的金额字段单位不同，是历史上两个 P1 级 bug 的根源。任何金额/市值阈值出现魔法数字前，先核对单位：

| 字段 | 单位 |
|---|---|
| A 股 `daily.amount` | **千元** |
| A 股 `daily_basic.total_mv` / `circ_mv` | **万元** |
| 期货 `fut_daily.amount` | **万元** |

完整口径与代码落点见[数据源与口径](../reference/data-sources.md)。

---

## 相关阅读

- [架构](architecture.md) —— `markets/` 在整体结构中的位置
- [数据源与口径](../reference/data-sources.md) —— 各市场数据源细节
- [CLI 参考](../reference/cli.md) —— `--market` 在各命令上的取值域
