# 因子编写

> [FactorZen](../../README.md) · [文档](../README.md) · **因子编写**

平台里的因子有两种形态：**表达式因子**（挖掘产出的字符串，如 `rank(ts_std(close, 20))`）和 **python 因子**（手写的 `DailyFactor` 子类）。本文只讲后者——你有一个想法、公式复杂到表达式算子库表达不了，需要自己写代码的情况。

表达式因子由[因子挖掘](mining.md)自动产出，不需要手写。

---

## 因子放哪里

| 位置 | 用途 | 是否随包分发 |
|---|---|---|
| `workspace/factor_store/<market>/<name>/` | **你自己的 python 因子**（三件套：meta.json + factor.py + 可选 parquet） | ❌ 属于研究产出，不进 pip 包 |
| `src/factorzen/builtin_factors/<freq>/` | 平台自带因子（动量/反转/波动率/Barra 风格等） | ✅ 随包分发 |
| `src/factorzen/builtin_factors/qlib/` | 框架自动生成的 Alpha158 / Alpha360 移植 | ✅ 随包分发，**不要手写** |

用户因子**唯一**加载路径：`load_library_factors()` 扫描 `factor_store` 并把 `DailyFactor` 子类注入 registry。内置同名时 builtin 优先（`register(override=False)` 让位）。

裁决真相仍是 `workspace/factor_library/<market>.jsonl`；资产库是载体，用 `fz factor-library store sync` / `verify` 维护。

---

## 用模板起步

```bash
pixi run -- fz factor new my_reversal --freq daily
```

命令在 `workspace/factor_store/ashare/my_reversal/` 生成 `factor.py` + 最小 `meta.json` 并打印路径。`--freq` 写入 meta 并决定骨架继承哪个基类；已存在同名文件时需要 `--force` 才覆盖。

> ⚠️ 这里的 `--freq` 是**因子注册频率**（`daily/weekly/monthly/intraday`），跟 `fz mine` 的 bar 粒度 `--freq {1m,5m,15m,1h,daily}` 完全是两回事。全 CLI 有三套 `--freq` 语义，见 [CLI 参考](../reference/cli.md)。

---

## 要实现什么

一个日频因子就是一个 `DailyFactor` 子类（`daily/factors/base.py`），声明几个类属性 + 实现 `compute()`：

```python
"""20 日成交量-收益相关性。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class VolumeReturnCorr20D(DailyFactor):
    name = "volume_return_corr_20d"      # 注册键，全局唯一
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]            # 决定 ctx 加载哪些数据
    lookback_days = 30
    description = "20 日滚动 Pearson 相关：1 日收益 vs log 成交量"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        ...  # 返回 [trade_date, ts_code, factor_value]
```

完整可跑的实现见 `workspace/factor_store/ashare/volume_return_corr_20d/factor.py`。

### 类属性契约

| 属性 | 说明 |
|---|---|
| `name` | 注册键。**为空则不会被注册**，`fz factor list` 里看不到 |
| `category` | 分类标签，注册表按它过滤 |
| `frequency` | 频率标签 |
| `required_data` | 列表，决定 `ctx` 允许访问哪些数据源；取 `"daily"` / `"daily_basic"`。**没声明就访问会抛 `ValueError`** |
| `lookback_days` | 声明的预热天数（见下方陷阱） |
| `description` | 一句话说明，会进报告 |

### `compute(ctx)` 的输入

`ctx` 是 `FactorDataContext`（`daily/data/context.py`），惰性加载、按需缓存：

| 属性 | 内容 |
|---|---|
| `ctx.daily` | 日线行情 `LazyFrame`，**已自动 join 复权因子**，多出 `close_adj` / `open_adj` / `high_adj` / `low_adj` 四列 |
| `ctx.daily_basic` | 每日估值（市值、换手、PE/PB 等），需在 `required_data` 里声明 |
| `ctx.weekly` / `ctx.monthly` | 日线下采样到周/月快照日 |
| `ctx.start` / `ctx.end` | 请求区间 `YYYYMMDD` |
| `ctx.expanded_start` | 往前扩了预热期后的实际取数起点 |

数据已按票池过滤好，你不需要自己处理 universe。

> ✅ **一律用 `*_adj` 复权列算价格类因子。** 用裸 `close` 会在除权日产生假跳空。`adj_factor` 未落盘时框架会优雅回退成原始价，不会中断——所以「跑通了」不代表复权生效，拉数据时记得把 `adj_factor` 一起拉。

### `compute(ctx)` 的输出

必须返回 Polars **`DataFrame`**（不是 `LazyFrame`，记得 `.collect()`），至少含三列：

```text
trade_date · ts_code · factor_value
```

多余的中间列会被下游 `select` 掉，但保持干净更省内存。

---

## 写因子时最容易踩的坑

> ⚠️ **不要给因子类加 `@dataclass`。**
> `DailyFactor` 刻意是普通类而非 dataclass。子类以无注解的类属性声明 `lookback_days = 30` 这种写法，只有在基类不是 dataclass 时才生效——一旦某层加上 `@dataclass`，生成的 `__init__` 会在实例化时用基类默认值把子类声明覆盖掉，而消费方（`pipelines/daily_single.py`、`discovery/python_factor.py`）读的都是**实例**属性，预热窗口会静默退化成 20 天且不报错。
> `tests/daily/test_correlation_registry.py` 有全量守卫盯着这件事（原 `test_factor_class_attr_declaration` 已并入该文件）。
>
> （日内因子 `IntradayFactor` 走的是另一套约定：基类与子类**都**用 `@dataclass` + 注解字段，那样是自洽的，照 `builtin_factors/intraday/` 的样子写即可。）

其余逐条对照：

1. **所有 `shift` / `rolling_*` 必须 `.over("ts_code")`**。漏掉就是跨股票串号，结果看着有 IC 其实是数据泄漏。
2. **用 `ctx.start` 裁掉预热段**，只输出请求区间——预热期的因子值窗口不满，混进去会污染首段 IC。
3. **PIT**：`t` 日的因子值只能用 `trade_date <= t` 的信息。任何 `shift(-n)`、任何用到未来行的聚合都是未来函数。详见[设计铁律](../concepts/design-principles.md)。
4. **中性化不写进因子**。行业/市值中性、去极值、标准化都属于预处理，配在 YAML 的 `preprocessing` 段里（见[配置参考](../reference/configuration.md)），写进 `compute()` 会让因子无法复用、也无法和挖掘因子同口径比较。
5. **polars 的 NaN ≠ null**。聚合跳过 null 但会被 NaN 传染，`rank` 把 NaN 排最大。截面计算前先 `fill_nan(None)`，输出前用 `is_not_null() & is_finite()` 过一道。
6. **退化截面要守卫**。`E[x²] − E[x]²` 在近常数序列上会微负，开方直接 NaN 穿透——参考 `volume_return_corr_20d` 的 `when(var > 0)` 写法。
7. **停牌污染**。连续停牌会让 `shift(5)` 跨越很长的真实时间，内置 `reversal_5d` 的做法是统计窗口内 `vol > 0` 的天数，不足则置 null。

---

## 本地验证

```bash
# 1) 确认注册成功
pixi run -- fz factor list --freq daily

# 2) 单因子评估：RankIC / 衰减 / 单调性 / 信号多空 / 换手（毛口径）
pixi run -- fz factor eval my_reversal --start 20220101 --end 20241231 --universe csi500
#    模拟交易回测：日环撮合 + walk-forward / benchmark（净口径）
pixi run -- fz factor backtest my_reversal --start 20220101 --end 20241231 --universe csi500

# 3) 出单页 HTML 报告
pixi run -- fz report build my_reversal --start 20220101 --end 20241231 \
  --universe csi500 --reuse

# 4) 参数网格扫描
pixi run -- fz factor sweep my_reversal --start 20220101 --end 20241231 \
  --grid backtest.top_n=30,50,100 --sort-by ir
```

`fz factor eval` / `fz factor backtest` 的产物落 `workspace/factor_evaluations/<run_id>/`，含指标、图表与 `manifest.json`。参数全表见 [CLI 参考](../reference/cli.md#fz-factor)。

> ℹ️ `fz factor list` 打印的不只有手写因子——它会顺带把因子库里的**表达式型**记录动态注入注册表（`discovery/library_provider.py` 的 `load_library_factors`），这些条目默认叫 `mined_<sha1 前 8 位>`。目的是让入库的挖掘因子也能用 `fz factor eval` 复现。库缺失或损坏时只打印一行跳过提示，不影响列表本身。

> ⚠️ 因子名与内置/库因子重名时，注册表按「builtin/workspace 优先，library provider 让位」处理并打 warning。想覆盖内置因子是可以的（workspace 扫描在后），但要有意为之。

调试单因子时用 `--dry-run` 先看生效配置，用 `--set key=value` 临时改参数而不动 YAML：

```bash
pixi run -- fz factor backtest my_reversal --start 20220101 --end 20241231 \
  --set backtest.top_n=30 --dry-run
```

> ⚠️ `--set` **不是全局旗标**，只挂在 `fz factor eval` / `backtest` / `sweep` 上。值经 `yaml.safe_load` 解析后在 pydantic 校验前注入。

---

## 让手写因子进因子库

单因子指标好看不构成入库理由——平台的裁决是「**相对现有库有没有增量**」。手写 python 因子和挖掘出的表达式因子走**同一道 lift 准入**，没有特权通道。

```bash
# 1) 先 dry-run 看裁决（默认就是 dry-run，不写库）
pixi run -- fz factor-library lift-test \
  --factor my_reversal \
  --market ashare --universe csi500 --start 20200101 --end 20241231

# 2) 确认结果后再写库
pixi run -- fz factor-library lift-test \
  --factor my_reversal \
  --market ashare --universe csi500 --start 20200101 --end 20241231 --apply
```

几条硬约束（都在 `cli/main.py` 的 `_cmd_factor_library_lift_test` 里 fail-loudly，不会静默降级）：

- `--factor` 目前**只支持 `--market ashare`**，其它市场直接报错退出。
- `--factor` 时 **`--universe` 必填**（如 `csi500`）——python 因子的物化需要 PIT membership 口径。
- 因子名必须已在 registry 里；未注册直接报错，不会跳过。
- `--factor` 是空格分隔多值：`--factor a b c`，不是逗号、也不是重复旗标。
- 默认封顶写 `probation`，要让 lift 裁决直接写 `active` 需显式加 `--set allow_active=true`。

准入通过后，登记簿里会多一条 `kind="python"` 的记录，三个显式键承载身份：

| 字段 | 含义 |
|---|---|
| `kind` | `"python"`（对表达式因子是 `"expression"`） |
| `name` | registry 因子名，等于你的 `DailyFactor.name` |
| `impl` | 实现标识，缺省与 `name` 相同 |

`expression` 字段则填成 `py::{name}` 这样一个**故意不合法的哨兵串**——让所有以 `expression` 为主键的既有逻辑（去重、池键、台账）零改动继续工作。语义细节见[因子库与增量准入](../concepts/factor-library.md#表达式因子与-python-因子共存)。

> ℹ️ **显式键优先于哨兵推断**（`discovery/factor_library.py`）。老记录可能只有哨兵没有 `kind`，两种都能被正确识别。

入库之后，这个因子就会被 [`fz combine from-library`](combination.md) 当作候选参与多因子组合，和挖掘因子一视同仁。若裁决落在 `probation`，还要走[向前确认](../concepts/factor-library.md#probation-的完整生命周期)才能转正。

---

## python 因子面板的磁盘缓存

python 因子的物化（`discovery/python_factor.py`）比表达式因子贵得多——要实例化类、拉全窗口数据、跑用户代码。因此**只有 python 面板有磁盘缓存**，表达式因子刻意不缓存（物化便宜，且会流经多种预处理帧，缓存容易被毒化）。

缓存位置：

```text
data/cache/python_factor_panels/{market}/{name}/{key}.parquet
```

缓存键（`_panel_cache_key`）是这七个维度的 sha1：

```text
market | name | start | end | universe | impl_sha | lb{lookback_days}
```

其中 `impl_sha` 是**实现源文件的 sha1 前 16 位**——你改一行代码，缓存自动失效，不会拿旧面板骗你；`lookback_days` 预热天数也进键，避免不同预热窗口共用缓存。

三条设计值得知道：

- **动态类不缓存。** `inspect.getsource()` 取不到类体（`type()` 动态生成的类）时 `impl_sha` 为 `None`，整条路径跳过缓存不读不写。指纹不完整的缓存键会毒化，宁可重算。
- **空面板不落盘。** 数据还没回补时算出来是空的，一旦落盘就会持续命中空缓存直到实现或窗口变化——这正是「文件存在 ≠ 数据完整」那类静默数据缺失。所以空结果直接不写。
- **缓存层的任何异常都不影响计算结果。** 读失败会删掉坏文件并重算，写失败只打 warning。落盘是 `tmp → rename` 原子替换。

> ⚠️ 缓存键**不含预处理配置**——它缓存的是因子原始面板，不是预处理后的结果。改中性化配置不需要（也不会）清缓存。
>
> 真要手工清理，删掉对应目录即可：`rm -rf data/cache/python_factor_panels/ashare/my_reversal`。

---

## 相关阅读

- [因子库与增量准入](../concepts/factor-library.md) —— lift 裁决、四态状态机、`py::` 哨兵的完整语义
- [因子挖掘](mining.md) —— 表达式因子怎么自动产出
- [多因子组合](combination.md) —— 入库因子如何被组合消费
- [配置参考](../reference/configuration.md) —— 预处理、回测、中性化的 YAML 字段
- [CLI 参考](../reference/cli.md#fz-factor) —— `fz factor` 全参数
