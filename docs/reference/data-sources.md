# 数据源与口径

> [FactorZen](../../README.md) · [文档](../README.md) · **数据源与口径**

正文默认 A 股。本页写清数据从哪来、落在哪、金额/市值**单位口径**——后者是本仓两条 P1 级 bug 的根源，任何涉及金额/市值阈值的改动都必须先对照 §3。

多市场（crypto/期货/美股）的数据源与能力边界见 [多市场](../concepts/multi-market.md)。产物目录见 [产物布局](artifacts.md)，凭据见 [环境变量](environment.md)。

---

## 1. 各市场数据源一览

| 市场 | 数据源 | 落盘位置 | 状态 |
|---|---|---|---|
| A 股（`ashare`） | Tushare Pro | `data/raw/` | **主市场**，日频 + 分钟 |
| Crypto（`crypto`） | Binance Vision 数据湖 | `data/crypto_lake/` | USDT-M 永续，日频 + 分钟；见 [§4](#4-多市场数据源) |
| 期货（`futures`） | Tushare（主力连续） | `data/raw/fut_*` | 国内商品期货；见 [多市场](../concepts/multi-market.md) |
| 美股（`us`） | Yahoo Finance chart API（自建 provider，非 Tushare） | `data/raw/us_daily/` | 真实实现但 MVP universe：限流 + 指数退避；静态 universe 有幸存者偏差；见 [多市场](../concepts/multi-market.md) |

市场注册走 `markets/registry.py`（builder 惰性构造 + 缓存），四个 profile 分别在 `markets/{ashare,crypto,futures,us}/profile.py` 末尾 `registry.register(...)`。

---

## 2. A 股：Tushare

### 2.1 真实调用的接口

代码里实际用到的 Tushare 接口全集，以及 `core/loader.py` 里对应的 fetch 函数：

| 接口 | fetch 函数 | 用途 |
|---|---|---|
| `daily` | `fetch_daily()` | 日线行情 |
| `daily_basic` | `fetch_daily_basic()` | 每日指标（市值/换手/估值） |
| `stk_mins` | `fetch_minute()` | 分钟线 |
| `fina_indicator` 等 | `fetch_finance()` | 财务指标（按公告日 PIT 对齐） |
| `moneyflow` | `fetch_moneyflow()` | 资金流 |
| `hk_hold` | `fetch_hk_hold()` | 北向持股 |
| `margin_detail` | `fetch_margin_detail()` | 两融明细 |
| `stk_holdernumber` | `fetch_stk_holdernumber()` | 股东户数 |
| `top_list` | `fetch_top_list()` | 龙虎榜 |
| `stock_basic` | `fetch_stock_basic()` | 股票列表（`list_status="L,D,P"`，含退市与暂停，PIT 必需） |
| `namechange` | `fetch_namechange()` | 曾用名（ST 判定） |
| `adj_factor` | `fetch_adj_factor()` | 复权因子 |
| `index_daily` | `fetch_index_daily()` | 指数日线 |
| `index_member_all` | `fetch_index_member_all()` | 指数成分（逐日快照） |
| `trade_cal` | `fetch_trade_cal()` | 交易日历 |

另外还调用了 `index_weight`、`index_classify`、`fut_daily`、`fut_basic`、`fut_mapping`。

> ℹ️ `fetch_stock_basic` 特意用 `list_status="L,D,P"`（上市 + 退市 + 暂停）而不是只取 `L`。只取在市股票会造成幸存者偏差——历史回测里那些后来退市的股票必须存在。

### 2.2 落盘格式

`core/storage.py` 的分区路径格式：

```text
{base_dir}/{data_type}/year={YYYY}/month={MM}/data.parquet
```

`base_dir` 默认 `DATA_RAW`（= `data/raw`）。四个 API：

| 函数 | 作用 |
|---|---|
| `save_parquet(df, data_type, ...)` | 写入分区 |
| `load_parquet(data_type, start, end, base_dir=DATA_RAW)` | 读区间（内部 `pl.scan_parquet(base/data_type/"**/*.parquet")`） |
| `scan_parquet(...)` | 惰性扫描 |
| `partition_exists(data_type, year, month)` | 分区存在性检查 |

> ⚠️ **指数日线按指数拆目录**：磁盘上是 `data/raw/index_daily_000300_SH/`、`data/raw/index_daily_000905_SH/`，不是单一的 `index_daily/`。

### 2.3 分钟数据

分钟数据在 **`data/raw/minute_1min/`**（27 GB），常量 `DATA_RAW_MINUTE` 指向它。

- 写入方：`dataio/minute_ingest.py`，`MINUTE_DATA_TYPE = "minute_1min"`
- 消费方统一走 `load_parquet("minute_1min", ...)`：`intraday/audit.py`、`intraday/features/engine.py`、`intraday/bars_cache.py`、`discovery/intraday_expr.py`

A 股分钟 bar 的口径写死在 `intraday/sessions.py`（单一真源，跨年抽查无漂移）：

- **bar 标签 = bar-end**：常规 09:31..11:30（120 根）∪ 13:01..15:00（120 根），共 240 根。标签 `t` 的 bar 覆盖 `(t-1min, t]`。**没有 13:00 标签，有 11:30 标签。**
- 另有 **09:30 竞价 bar**（开盘集合竞价）；**15:00 bar 含收盘集合竞价**。
- `15:01..15:30` 标签 bar 仅出现在北交所 920 前缀代码上（2024 起），量极小 → 政策 `AFTER_HOURS_POLICY = "drop"`，**一律剔除 >15:00 的 bar**。

> ⚠️ 本机分钟数据覆盖有缺口：2017 全年、2018 十个月、**2019 整年缺失**、2020-2025 全、2026 到 04-10。跨 2019 的分钟研究会静默少数据，用覆盖审计确认而不是假设连续。

派生的 5 分钟 bar 与日内特征落在 `data/derived/`（`bars_5min/`、`intraday_features/{version}/{freq}/`）。

---

## 3. 单位口径（两条 P1 bug 的根源）

> ⚠️ **任何金额/市值阈值出现魔法数字之前，先回来核对这张表。** 单位错一位等于阈值错 1000 倍，而且不会报错——只会静默返回一个空得离谱或宽得离谱的股票池。

### 3.1 口径总表

| 数据 | 字段 | 单位 | 换算到元 | 现场 |
|---|---|---|---|---|
| A 股日线 | `daily.amount` | **千元** | ×1e3 | `daily/data/flows.py` 的 `_AMOUNT_TO_YUAN` |
| A 股每日指标 | `daily_basic.total_mv` | **万元** | ×1e4 | — |
| A 股每日指标 | `daily_basic.circ_mv` | **万元** | ×1e4 | `daily/data/flows.py` 的 `_CIRC_MV_TO_YUAN` |
| 龙虎榜 | `top_list.net_amount` | **万元** | ×1e4 | `daily/data/flows.py` 的 `_NET_AMOUNT_TO_YUAN`、`core/loader.py` 的 `TOP_LIST` schema 注释 |
| 龙虎榜 | `top_list.amount` | **千元** | ×1e3 | 同上 |
| 资金流 | `moneyflow.net_mf_amount` | **万元** | ×1e4 | `daily/data/flows.py` 的 `_FLOW_SRC` 映射 |
| 两融 | `margin_detail.rzye` / `rzmre` | **元** | ×1 | `daily/data/flows.py` 模块 docstring / `_attach_margin` |
| 两融 | `margin_detail.rqyl` | 股 | — | 同上 |
| A 股分钟 | `vol` | 股 | — | `intraday/sessions.py` |
| A 股分钟 | `amount` | **元** | ×1 | 同上 |
| A 股日线 | `vol` | 手 | — | 同上 |

期货等非 A 股字段的单位见 [§3.5](#35-多市场单位差异)。

### 3.2 换算常量与实际用法

`daily/data/flows.py` 把换算收在三个常量里：

```python
_CIRC_MV_TO_YUAN = 1e4    # flows.py
_AMOUNT_TO_YUAN  = 1e3    # flows.py
_NET_AMOUNT_TO_YUAN = 1e4 # flows.py  top_list net_amount 万元
```

比值计算一律**先统一到元再相除**：

| 派生量 | 公式 | 现场 |
|---|---|---|
| `margin_ratio` | `rzye(t') / (circ_mv(t') × 1e4)` | `flows.py` 的 `_attach_margin` |
| 龙虎榜净买占比 | `(net_amount × 1e4) / (amount × 1e3)` | `flows.py` 的 `_attach_toplist` |

连喂给 LLM 的 prompt 里都显式带上了单位说明（`llm/generation.py`：「net_amount 万元、amount 千元，比前统一到元」），避免生成的表达式在单位上出错。

### 3.3 踩过的真 bug：流动性门槛

`core/universe.py` 保留着事故现场的注释：

```python
# Tushare daily.amount 单位是千元，min_amount 语义是元 → 先换算再比较，
# 否则等价于要求 1000×min_amount 元（默认 1000万→假门槛 100 亿，股票池塌缩）。
liquid = daily.filter(pl.col("amount") * 1000.0 >= min_amount).select("ts_code").unique()
```

漏掉 `* 1000.0` 时，默认的 1000 万门槛实际变成 **100 亿**，A 股几乎没有股票能过——股票池直接塌缩，而且不报错。

**真实生效的流动性门槛是 `min_amount = 10_000_000`（1000 万元）**，定义在 `core/universe.py`。

> ⚠️ `config/constants.py` 定义了 `MIN_MARKET_CAP_CNY = 3e8`（3 亿），但它在 `src/` 里**没有任何消费方**——常量已声明、**未接线**。不要把它当成生效的默认市值过滤。

### 3.4 `total_mv` 的特殊性

`total_mv` 在三处使用，**都只取对数、不做单位换算**：

| 用途 | 位置 |
|---|---|
| Barra Size 因子 `ln(total_mv)` | `risk/style_factors.py` |
| 市值中性化用 `log(total_mv)` | `daily/preprocessing/neutralizer.py` |
| 内置 size 因子 | `builtin_factors/daily/size.py` |

因为 `ln(万元 × 1e4) = ln(万元) + ln(1e4)`，单位差异只是一个**平移常数**，对截面 zscore 与回归残差没有影响。所以这三处不做换算是正确的——但也意味着**如果你新增一个直接用 `total_mv` 数值（而非对数）的因子或阈值，必须自己 ×1e4**。

### 3.5 多市场单位差异

> 多市场（crypto/期货/美股）支持见 [多市场](../concepts/multi-market.md)。跨市场复用金额逻辑时先核对下表。

| 数据 | 字段 | 单位 | 换算到元 | 现场 |
|---|---|---|---|---|
| **期货** | `fut_daily.amount` | **万元** | ×1e4 | `markets/futures/factors.py` |
| 期货 | `fut_daily.vol` / `oi` | 手（lots） | — | 同上 |

> ⚠️ **期货的 `amount` 是万元，与 A 股日线的千元不同。** `markets/futures/factors.py` 还提示 `vwap = amount/vol = 万元/手`，是品种内的活跃度代理，**跨品种不可比**。

---

## 4. 多市场数据源

> 多市场（crypto/期货/美股）的完整能力边界见 [多市场](../concepts/multi-market.md)。本节只保留参考文档必需的数据源与落盘事实；期货/美股接口细节以适配器代码为准。

### 4.1 Crypto：Binance Vision 数据湖

根目录 `data/crypto_lake/`（常量 `CRYPTO_LAKE`）。权威布局见 `markets/crypto/lake.py` 顶部 docstring：

```text
<root>/klines_1m/symbol=BTCUSDT/2026-05.parquet
<root>/funding/symbol=BTCUSDT/2026-05.parquet
<root>/metrics/symbol=BTCUSDT/2026-06-27.parquet
<root>/meta.parquet          # ts_code, name, list_date
<root>/manifest.json         # 回填区间/gaps
```

**时间戳一律 naive-UTC `Datetime("us")`。** 写入时自动追加 `ts_code` 列，读取即拼接过滤，无需解析目录名。

回填：

```bash
pixi run -- fz data crypto backfill --help
```

manifest 记录 `start` / `end` / `symbols` / `gaps`：

```json
{
  "start": "20230701", "end": "20241231",
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
  "gaps": []
}
```

> ⚠️ `lake.py` 的 docstring 说 manifest 含 `git_sha`，但**实际生成的 manifest 里没有这个字段**。别依赖它做复现校验，以 `start`/`end`/`gaps` 为准。

### 4.2 数据湖是默认源

`markets/crypto/profile.py` 的解析规则：

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

---

## 5. 缓存与覆盖审计

### 5.1 缓存键完整性

> ⚠️ **缓存键必须含全部影响结果的维度。** 漏一个维度 = 缓存毒化：改了参数却读回旧结果，且不报错。

`discovery/python_factor.py` 的 `_panel_cache_key` 是本仓的参考实现，注释里直接写着「缓存键完整性是本仓 P1 教训」：

```python
payload = f"{market}|{name}|{start}|{end}|{universe}|{impl_sha}|lb{lookback_days}"
return hashlib.sha1(payload.encode()).hexdigest()[:24]
```

七个维度缺一不可：

| 维度 | 为什么必须在键里 |
|---|---|
| `market` | 不同日历/数据源 |
| `name` | 因子身份 |
| `start` / `end` | 裁窗后面板不同 |
| `universe` | PIT membership 并集变 → 面板变 |
| `impl_sha` | **实现源码指纹**，改源码必须让缓存失效 |
| `lookback_days` | 预热窗口影响区间头部取值；漏掉则不同预热窗口共用缓存 |

`impl_sha` 取因子实现源文件的 sha1（`_impl_source_sha`，`python_factor.py`）。**动态类（`type()` 生成）取不到源码 → 返回 `None` → 整条路径跳过缓存**，宁可慢也不毒化。

面板缓存落 `DATA_CACHE/python_factor_panels/{market}/{name}/{key}.parquet`。缓存文件损坏或读失败时**删文件并返回 None**，不崩。

同理，Tushare 取数的缓存键也须含 `freq` / `api_name` / `fields` / `ts_codes` / 日期覆盖。

### 5.2 用交易日历覆盖审计，别用文件存在启发式

> ⚠️ **「文件存在」≠「数据完整」。** 一个分区文件可能只有半个月数据。

`core/loader.py` 的注释记录了这个教训：

```text
用交易日历覆盖审计替代「季度首月分区存在」启发式：后者会把部分年数据误判为整年
```

所以增量拉取的判据统一是**逐交易日覆盖审计，只拉缺失的交易日**，不是分区存在性检查：

| 位置 | 说明 |
|---|---|
| `core/loader.py` 的 `fetch_daily()` | 全市场：交易日历覆盖审计，只拉缺失交易日 |
| `core/loader.py` 的 `fetch_margin_detail()` 等 | 日频按缺失交易日市场级拉取 + 缓存 |
| `core/loader.py` 的 `fetch_index_daily()` | 交易日历覆盖审计替代分区存在启发式 |

### 5.3 事件类数据的「诚实缺测」

龙虎榜这类事件数据不能简单 fill 0——「没拉到」和「确定没上榜」是两回事。`daily/data/flows.py` 的 `_attach_toplist` 规则：

| 情形 | 处理 |
|---|---|
| `trade_date` ∈ 已知日集合，且 join 缺失 | **fill 0**（确定没上榜） |
| `trade_date` ∉ 已知日集合 | **保持 null**（未拉取 ≠ 没上榜） |
| 全空源 | 全 null（覆盖审计诚实，缺口可见） |

已知日集合 = 源表 distinct `trade_date`（真实行 ∪ `__EMPTY__` sentinel）。极端行情下全日无人上榜是正常的，此时写 sentinel 行，既不算失败也避免永久重拉。

> ⚠️ 无差别 fill 0 会让「数据没拉全」伪装成「事件确实没发生」，覆盖体检也就跟着失明。新增事件类叶子时照抄这套三分支规则，别图省事。

---

## 6. 其它数据口径注意事项

- **PIT**：universe 逐日快照；财务数据按**公告日**对齐；执行定价用 `pre_close` 而非当日收盘；滚动因子需扩窗预热。
- **`polars` 的 NaN ≠ null**：聚合跳过 null 但会被 NaN 传染，`rank` 把 NaN 排最大，`NaN > x` 为 `True`。**截面计算前默认 `fill_nan(None)`。**
- 数据链路自检：`pixi run smoke-data`（连通性检查需 `TUSHARE_TOKEN`，本地 `data/raw/` 审计可离线跑）。
