# 多市场适配

> [FactorZen](../../README.md) · [文档](../README.md) · **多市场适配**

平台默认叙事与示例以 **A 股** 为主。挖掘引擎、护栏、因子库、回测引擎共用一套实现；市场差异收敛进 `markets/` 的适配层。本文是 crypto / 期货 / 美股内容的**唯一家园**——其它文档只保留一句指引并链到这里。

---

## 多市场架构（Ports & Adapters）

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

`MarketProfile`（`base.py`）把它们组装成一个市场的完整描述。`registry.py` 提供 `register` / `get` / `list_markets`，**采用惰性构造 + 缓存**——避免 import 阶段就去建 ccxt client 这类重对象。四个 profile 分别在 `markets/{ashare,crypto,futures,us}/profile.py` 末尾 `registry.register(...)`。

设计约定是「**参数化带 A 股默认值**」：新市场通过注入参数接入，A 股行为保持默认值不变。A 股零回归是接入任何新市场的底线。

`markets/` 全树内 `NotImplementedError` / `TODO` / `FIXME` 数量为 **0**——四个市场都是可跑的适配器，没有存根。

---

## 各市场支持现状

| 市场 | 数据源 | 落盘位置 | 适配层规模 | 成熟度 |
|---|---|---|---:|---|
| `ashare` | Tushare Pro（经 `core/loader`） | `data/raw/` | 235 行 | **薄适配层**——真实重逻辑在 `core/` 与 `daily/`，profile 只是把既有实现包成 Port。主市场：日频 + 分钟 |
| `crypto` | Binance Vision 数据湖（默认）· ccxt（备用） | `data/crypto_lake/` | 1,725 行 | **最完整的 Port 实现**：唯一自带 `RiskModel`，另有自己的回测、组合、挖掘、重采样、板块分类。USDT-M 永续，日频 + 分钟 |
| `futures` | Tushare `fut_daily` / `fut_mapping` / `fut_basic` | `data/raw/fut_*` | 790 行 | 真实实现：主力连续拼接 + **乘法后复权**，有 ground-truth 测试；用交易日历覆盖审计判缺失。国内商品期货 |
| `us` | **Yahoo Finance** chart API（自建 provider，非 Tushare） | `data/raw/us_daily/` | 713 行 | 真实实现但 MVP universe：限流 + 指数退避，按 symbol parquet 缓存 |

### 能力边界不均（重要）

「四市场都能跑」指的是**挖掘链路**。下游能力的覆盖并不整齐：

| 命令 | 支持的市场 |
|---|---|
| `fz mine search/agent/team/pool-prebuild`、`fz factor-library *`、`fz validate overfit`、`fz combine from-library` | ashare · crypto · futures · us |
| **`fz data fetch`** | **仅 ashare**（无 `--market`；crypto 数据走 `fz data crypto backfill`） |
| **`fz portfolio build` / `fz sim run` / `fz report portfolio`** | **仅 ashare · crypto** |
| **`fz risk build` / `fz research run`** | **仅 ashare**（无 `--market`） |

因此准确的说法是：

- **ashare / crypto** —— 全链路可跑：取数 → 挖掘 → 准入 → 风险/组合 → 回测 → 报告。
- **futures / us** —— **只通到挖掘与因子库**（含过拟合校验）。没有数据拉取子命令（数据需自行准备），没有组合优化接线，没有风险模型。

这不是「未实现的存根」，而是「实现了一段、没接通另一段」。

> ⚠️ **`--market` 不是全局统一参数。** `fz mine` / `fz factor-library` / `fz validate overfit` 的取值域是 `{ashare, crypto, futures, us}`（默认 `ashare`）；`fz portfolio build` / `fz sim run` / `fz report portfolio` 只有 `{ashare, crypto}`。跨命令拼脚本时不要假设取值域一致。

### A 股（`ashare`）

主市场。Tushare 接口、PIT universe、财务公告日对齐、停牌/涨跌停/ST/次新/T+1、Barra 风险模型、无人值守日链路均以 A 股为默认路径。日频 + 分钟级（1m/5m/15m/1h，经 `intraday/`）。取数需 `TUSHARE_TOKEN`。

详见[数据源与口径](../reference/data-sources.md) 与[端到端教程](../getting-started/end-to-end-tutorial.md)。

### Crypto（`crypto`）

USDT-M 永续合约。数据默认走本地 **Binance Vision 数据湖**，无需 token；ccxt 为备用路径（REST 当前 451 不可达）。`markets/crypto/` 是最完整的 Port 实现，也是唯一 `risk` 非 None 的市场。

#### 数据湖布局

根目录 `data/crypto_lake/`（常量 `CRYPTO_LAKE`）。权威布局见 `markets/crypto/lake.py` 顶部 docstring：

```text
<root>/klines_1m/symbol=BTCUSDT/2026-05.parquet
<root>/funding/symbol=BTCUSDT/2026-05.parquet
<root>/metrics/symbol=BTCUSDT/2026-06-27.parquet
<root>/meta.parquet          # ts_code, name, list_date
<root>/manifest.json         # 回填区间/gaps
```

**时间戳一律 naive-UTC `Datetime("us")`。** 写入时自动追加 `ts_code` 列，读取即拼接过滤，无需解析目录名。

manifest 记录 `start` / `end` / `symbols` / `gaps`：

```json
{
  "start": "20230701", "end": "20241231",
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
  "gaps": []
}
```

> ⚠️ `lake.py` 的 docstring 说 manifest 含 `git_sha`，但**实际生成的 manifest 里没有这个字段**。别依赖它做复现校验，以 `start`/`end`/`gaps` 为准。

#### 数据源解析规则

`markets/crypto/profile.py`：

```python
# 默认走数据湖(REST 当前 451 不可达);注入 client → ccxt(既有测试零改动);source 显式优先
resolved = source or ("ccxt" if client is not None else "lake")
```

| `source` | provider |
|---|---|
| 未指定且未注入 client | **`lake`** → `CryptoLakeProvider(lake_root=CRYPTO_LAKE)` |
| 未指定但注入了 client | `ccxt` → `CryptoDataProvider` |
| `"lake"` / `"ccxt"` | 显式指定，优先级最高 |
| 其它 | `ValueError("未知 source: ...,支持 'lake' / 'ccxt'")`（`profile.py`） |

两条守卫：

- **湖为空**：`klines_1m` 目录不存在 → 报错「crypto 数据湖为空(...)：先运行 `fz data crypto backfill`」（`lake_provider.py`）。
- **ccxt 只支持日频**：intraday 调用抛 `ValueError("CryptoDataProvider(ccxt) 仅支持 daily;intraday 请用数据湖 provider")`（`provider.py`）。分钟级研究必须走湖。

#### 回补数据湖

从 Binance Vision 回补 1 分钟 K 线 / 资金费率 / 持仓量到本地数据湖：

```bash
pixi run -- fz data crypto backfill --start 20240101 --end 20241231 --top-n 50
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--start` / `--end` | —（必填） | `YYYYMMDD` |
| `--symbols` | 无 | 逗号分隔；缺省 = 按上月成交额 Top-N 自动选池 |
| `--top-n` | `50` | 自动选池规模 |
| `--lake-root` | `data/crypto_lake` | 数据湖根目录 |

**产物**：`data/crypto_lake/` 下分区 parquet。

#### Crypto 全链路（分步）

`fz research run` **没有 `--market`**，是 A 股专属。crypto 需分步走：

`fz mine` → `fz portfolio build --market crypto` → `fz sim run --market crypto`

| 步骤 | 命令要点 |
|---|---|
| 数据 | `fz data crypto backfill`（见上） |
| 挖掘 | `fz mine search/agent/team --market crypto`；池用 `--top-n`（默认 `50`）或 `--symbols`；`--freq` 可选分钟 bar |
| 准入 | `fz factor-library lift-test --market crypto`（默认 dry-run，写库加 `--apply`） |
| 组合 | `fz portfolio build --market crypto`：市场中性做空；`--gross-limit` 默认 `1.0`（毛敞口上限 Σ\|w\|）；`--top-n` 默认 `50` |
| 模拟 | `fz sim run --market crypto`：计资金费 + 做空的 NAV 回测 |
| 过拟合 | 非 A 股没有注册因子表，必须 `--expression`（示例见下） |

过拟合验收示例（CLI 原文）：

```bash
pixi run -- fz validate overfit --market crypto \
  --expression "rank(ts_std(close,20))" --start 20230101 --end 20241231 --freq 1h
```

报告侧：`fz report portfolio --market crypto` 时语义为 **USDT 计价 / 365 日年化 / 计资金费 / 按 sector 归因**；`--market` 缺省时从 sim 的 `manifest.json` 自动识别。

### 期货（`futures`）

国内商品期货。Tushare `fut_daily` / `fut_mapping` / `fut_basic`；主力连续拼接 + **乘法后复权**，有 ground-truth 测试；用交易日历覆盖审计判缺失。

- **只通到挖掘 + 因子库 + 过拟合校验**。没有 `fz data fetch` 子命令（数据需自行准备到 `data/raw/fut_*`），没有组合优化，没有风险模型。
- 池选法：`--top-n`，或 `--symbols`。
- 频率：**仅日频**——provider 对其他频率显式抛 `ValueError`。

### 美股（`us`）

Yahoo Finance chart API（自建 provider，**非 Tushare**）。限流 + 指数退避，按 symbol parquet 缓存，落盘 `data/raw/us_daily/`。

- **只通到挖掘 + 因子库 + 过拟合校验**。没有数据拉取子命令，没有组合优化，没有风险模型。
- 池选法：S&P 500 静态池按 `--top-n` 截断；也可用 `--symbols`。
- 频率：**仅日频**。

> ⚠️ **美股 universe 有幸存者偏差（PIT 铁律的已知例外）。**
> `markets/us/sp500_snapshot.py` 用的是约 2024 年的静态成分快照（约 490 支），**不是**历史 point-in-time 成分。代码 docstring 自认「用它回看历史窗口会引入幸存者偏差」。用美股做历史回看时，这个偏差需要你自己承担和折算。A 股侧的 PIT 是严格的。

---

## 口径与边界差异

### 频率支持

| 市场 | 支持的 bar 频率 |
|---|---|
| `ashare` | 日频 + 分钟级（1m/5m/15m/1h，经 `intraday/`） |
| `crypto` | 日频 + 分钟级（Vision 湖原生分钟数据） |
| `futures` / `us` | **仅日频**——provider 对其他频率显式抛 `ValueError` |

分钟 bar 可经 `fz data intraday-features build` 聚合成日频特征面板（battery_v1，20 特征），直接作为挖掘叶子使用。

> ⚠️ `--freq` 参数在不同命令下有**两套语义**：
>
> | 语境 | 取值 | 含义 |
> |---|---|---|
> | `fz mine` / `fz factor-library` / `fz validate overfit` 等 | `1m` / `5m` / `15m` / `1h` / `daily` | **crypto 的 bar 粒度**，默认 `daily`；**A 股只支持 `daily`** |
> | `fz factor` 注册 | daily / weekly / monthly | 因子频率 |
> | `fz data intraday-features` | 如 `5min` | 日内特征面板频率 |
>
> 三种不同语义，看清所在命令。详见 [CLI 参考](../reference/cli.md)。

### 单位口径

各市场的金额字段单位不同，是历史上两个 P1 级 bug 的根源。任何金额/市值阈值出现魔法数字前，先核对单位：

| 数据 | 字段 | 单位 | 换算到元 | 现场 |
|---|---|---|---|---|
| A 股日线 | `daily.amount` | **千元** | ×1e3 | `daily/data/flows.py` 的 `_AMOUNT_TO_YUAN` |
| A 股每日指标 | `daily_basic.total_mv` / `circ_mv` | **万元** | ×1e4 | `flows.py` 的 `_CIRC_MV_TO_YUAN` |
| **期货** | `fut_daily.amount` | **万元** | ×1e4 | `markets/futures/factors.py` |
| 期货 | `fut_daily.vol` / `oi` | 手（lots） | — | 同上 |

> ⚠️ **期货的 `amount` 是万元，与 A 股日线的千元不同。** 跨市场复用金额逻辑时这是第一个会踩的地方。`markets/futures/factors.py` 还提示 `vwap = amount/vol = 万元/手`，是品种内的活跃度代理，**跨品种不可比**。

完整口径（龙虎榜、资金流、两融、分钟 bar 等）见[数据源与口径](../reference/data-sources.md)。

### 风险模型未统一

风险模型是当前 Port 化最不完整的一块：

| 市场 | 风险模型 |
|---|---|
| `ashare` | Barra 模型在**独立的 `risk/` 包**里，未接进 `MarketProfile`（profile 的 `risk=None`），由 `portfolio/` 与 `attribution/` 直接调用 |
| `crypto` | 走 Port 化的 `markets/crypto/risk.py`，是唯一 `risk` 非 None 的市场 |
| `futures` / `us` | 无 |

A 股与 crypto 各有一套风险模型，走的还是两条不同的接线方式。统一到 Port 是待办。

### 挖掘时的池选法

挖掘类命令的 `--market` 是 4 值域 `{ashare, crypto, futures, us}`，默认 `ashare`。

| 市场 | 池的选法 | 注意 |
|---|---|---|
| `ashare` | `--universe`（如 `csi500`） | 唯一支持日内叶子与 scout 的市场 |
| `crypto` | `--top-n` 按成交额，或 `--symbols` 指定 | `--freq` 可选分钟级 bar 粒度 |
| `futures` | `--top-n`，或 `--symbols` | 主力连续 + 乘法后复权 |
| `us` | S&P 500 静态池按 `--top-n` 截断 | 静态成分有幸存者偏差（见上） |

`--top-n` 默认 `50`。

### Crypto 组合 / 模拟语义

| 命令 | crypto 语义差异 |
|---|---|
| `fz portfolio build --market crypto` | 市场中性做空；`--gross-limit` 默认 `1.0`（毛敞口上限 Σ\|w\|） |
| `fz sim run --market crypto` | 计资金费 + 做空的 NAV 回测 |
| `fz report portfolio --market crypto` | USDT 计价 / 365 日年化 / 计资金费 / 按 sector 归因 |

### 成对修改提示

以下跨市场路径历史上是 bug 来源——改任一侧必须检查另一侧：

| 路径 A | 路径 B | 共享的是什么 |
|---|---|---|
| `markets/crypto` ccxt provider | Vision 湖 provider（默认） | 叶子语义、end 边界、分页 |
| A 股引擎默认参数 | crypto 参数注入 | 「参数化带 A 股默认值」，A 股零回归是底线 |

---

## 命令用法速查

**数据**

```bash
# A 股（主路径）
pixi run fz data fetch daily --start 20200101 --end 20241231
# crypto 数据湖
pixi run -- fz data crypto backfill --start 20240101 --end 20241231 --top-n 50
# futures / us：无 CLI 拉取，数据需自行准备
```

**挖掘 / 因子库 / 过拟合**（四市场；`--market` 默认 `ashare`）

- `fz mine search|agent|team --market {ashare,crypto,futures,us}`
- `fz factor-library * --market {ashare,crypto,futures,us}`
- 非 A 股过拟合必须带表达式：

```bash
pixi run -- fz validate overfit --market crypto \
  --expression "rank(ts_std(close,20))" --start 20230101 --end 20241231 --freq 1h
```

**组合 / 模拟 / 报告**（仅 `ashare` · `crypto`）

- `fz portfolio build --market crypto`（市场中性做空，`--gross-limit` 默认 `1.0`）
- `fz sim run --market crypto`（计资金费 + 做空 NAV）
- `fz report portfolio --market crypto`（USDT / 365 日年化 / 资金费 / sector 归因）

crypto 完整分步链路见上文「Crypto 全链路」。全部参数见 [CLI 参考](../reference/cli.md)。

---

## 相关阅读

- [架构](architecture.md) —— `markets/` 在整体结构中的位置
- [设计铁律](design-principles.md) —— PIT 铁律与美股例外的原则层表述
- [数据源与口径](../reference/data-sources.md) —— A 股单位口径与缓存审计细节
- [因子挖掘](../guides/mining.md) —— 挖掘流程（默认 A 股）
- [CLI 参考](../reference/cli.md) —— `--market` 在各命令上的取值域
