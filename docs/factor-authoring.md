# 因子编写

> [FactorZen](../README.md) · [文档](README.md) · [架构](architecture.md) · **因子编写** · [运行手册](runbook.md) · [示例报告](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html)

日常研究因子写在 `workspace/factors/`（你的空间），框架代码写在 `src/factorzen/`。框架自带的示例/测试因子在 `src/factorzen/builtin_factors/`（随包分发）——注册表同时扫描两处，同名时 `workspace` 覆盖内置。**新增因子默认不要改 `src`。**

## 1. 创建模板

```bash
pixi run fz factor new my_alpha --frequency daily
```

生成文件：

```text
workspace/factors/daily/my_alpha.py
```

每个频率目录下还放了一份手写模板，列出编写约定、可复制代码与检查点，可直接照着改：

- [`workspace/factors/daily/TEMPLATE.md`](../workspace/factors/daily/TEMPLATE.md)
- [`workspace/factors/weekly/TEMPLATE.md`](../workspace/factors/weekly/TEMPLATE.md)
- [`workspace/factors/monthly/TEMPLATE.md`](../workspace/factors/monthly/TEMPLATE.md)
- [`workspace/factors/intraday/TEMPLATE.md`](../workspace/factors/intraday/TEMPLATE.md)

`fz factor new` 生成最小骨架，TEMPLATE.md 提供更完整的约定与示例；两者择一起步即可。

可选频率与落点：

| `--frequency` | 落点 | 基类 |
|---------------|------|------|
| `daily` | `workspace/factors/daily/` | `DailyFactor` |
| `weekly` | `workspace/factors/weekly/` | `DailyFactor`（`frequency="weekly"`）|
| `monthly` | `workspace/factors/monthly/` | `DailyFactor`（`frequency="monthly"`）|
| `intraday` | `workspace/factors/intraday/` | `IntradayFactor`（当前非主线）|

低频因子都继承 `DailyFactor`，通过 `frequency` 区分日频、周频与月频；分钟线因子继承 `IntradayFactor`。

## 2. 实现约定

一个因子由四个类属性 + 一个 `compute` 方法构成，`compute(ctx)` 返回的 Polars `DataFrame` 至少包含三列：

```text
trade_date
ts_code
factor_value
```

常用数据入口：

| 入口 | 内容 |
|------|------|
| `ctx.daily` | 日行情（含 `close_adj` 等复权字段）|
| `ctx.daily_basic` | 估值、市值等 `daily_basic` 字段 |
| `ctx.weekly` / `ctx.monthly` | 周频、月频快照 |
| `ctx.weekly_basic` / `ctx.monthly_basic` | 对应频率的基础数据快照 |
| `ctx.start` | 运行起始日（`YYYYMMDD` 字符串）|

因子值只能使用**当时可获得**的数据。需要行业、市值中性化时，在 YAML 里开启 `neutralize` 与 `neutralize_by`，**不要在因子内部重复做管线已有的预处理。**

## 3. 最小端到端示例

下面是一个 5 日反转因子的完整实现（`reversal_5d` 也是内置因子之一，这里展示其写法作模板）。放进 `workspace/factors/daily/` 时请改用你自己的名字，以免与内置同名因子混淆：

```python
"""5 日反转因子。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class Reversal5D(DailyFactor):
    name = "reversal_5d"
    category = "daily"
    description = "5 日反转：-(close_adj[t] / close_adj[t-5] - 1)"
    lookback_days = 10

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        return (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (-(pl.col("close_adj") / pl.col("close_adj").shift(5).over("ts_code") - 1.0)).alias(
                    "factor_value"
                )
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )


# 模块级实例化，供 registry 自动发现
Reversal5D()
```

要点：

- `lookback_days` 要覆盖因子用到的最长回看窗口（这里 `shift(5)` 取 10 留余量）。
- 截面计算用 `.over("ts_code")` 按股票分组，避免跨股票串号。
- 用 `ctx.start` 过滤掉预热期，只输出请求区间内的因子值。
- 文件末尾**实例化一次**即完成注册。

## 3.1 进阶示例：量价相关因子

`volume_return_corr_20d` 是一个更接近真实研究的示例：20 日「1 日收益 × 对数成交量」滚动 Pearson 相关。它展示了多中间列、`rolling_*` 的 `min_samples`、以及方差非正时的守卫写法。完整实现见 [`workspace/factors/daily/volume_return_corr_20d.py`](../workspace/factors/daily/volume_return_corr_20d.py)；另附一个精简 YAML 配置 [`workspace/configs/daily/volume_return_corr_20d.yaml`](../workspace/configs/daily/volume_return_corr_20d.yaml)（固定区间并显式关闭 walk-forward）。

运行并查看报告（无 `--config` 会套用内置研究级预设，即示例报告的生成方式）：

```bash
pixi run fz factor run volume_return_corr_20d --start 20160606 --end 20260606
pixi run fz report path <run_id>
```

它产出的真实 tear sheet 已收录为示例报告：[https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html](https://rookiewu417.github.io/FactorZen/volume_return_corr_20d-tear-sheet.html)。该因子预测能力偏弱，正好演示报告如何诚实暴露「统计显著但经济意义弱」的结论——这是 FactorZen 的设计取向，而非缺陷。

要点（相对最小示例新增的）：

- 多个 `rolling_mean(...).over("ts_code")` 中间列拼出协方差与方差，再算相关系数。
- `min_samples` 控制窗口内最少有效样本，避免早期窗口噪声。
- 用 `pl.when(...).then(...).otherwise(None)` 守卫方差非正的退化情形，并 `clip(-1.0, 1.0)` 约束到合法相关系数区间。

## 4. 注册与发现

因子注册表同时扫描"框架自带"与"用户自定义"两组包：

```text
factorzen.builtin_factors.{daily,weekly,monthly,qlib,intraday}   # 框架自带，随包分发
workspace.factors.{daily,weekly,monthly,intraday}                # 你的因子（默认空）
```

同名因子以 `workspace`（用户）覆盖 `builtin_factors`（框架）。`builtin_factors/qlib/` 暴露 qlib Alpha158/Alpha360 特征，每个 qlib 特征注册为一个 FactorZen 因子。运行 qlib 因子前需要准备 qlib 数据包，详见 [`src/factorzen/builtin_factors/qlib/README.md`](../src/factorzen/builtin_factors/qlib/README.md)。

## 5. 验证

**列出因子**

```bash
pixi run fz factor list
pixi run fz factor list --frequency intraday
```

**运行单个因子**

```bash
pixi run fz factor run reversal_5d --start 20230101 --end 20241231 --universe csi500
```

**运行配置文件**

```bash
pixi run fz config validate workspace/configs/daily/daily_factor_template.yaml
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
```

新增因子建议至少补一条单元测试，覆盖输出 schema、缺失字段与不可用数据场景。涉及收益对齐、停牌、涨跌停、容量或样本切分时，**必须补回归测试以防未来函数。**
