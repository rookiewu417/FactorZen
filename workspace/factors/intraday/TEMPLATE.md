# 分钟频因子模板

分钟频因子放在 `workspace/factors/intraday/{factor_name}.py`。

> ℹ️ 如果你的目标是把日内信息用进日频挖掘，通常**不需要写分钟频因子**——平台已把分钟 bar 聚合成 17 个日频日内微观结构特征（`i_*` 叶子），可直接作为挖掘叶子使用，见[因子挖掘指南](../../../docs/guides/mining.md)。本模板适用于确实需要在分钟粒度上产出因子值的场景。

## 编写约定

- 继承 `IntradayFactor`。
- 返回列是 `trade_time`、`ts_code`、`factor_value`，不是 `trade_date`。
- 默认数据入口是 `ctx.minute`，常见字段包括 `open`、`high`、`low`、`close`、`vol`、`amount`。
- 日内累计计算要按 `["ts_code", trade_date]` 分组，避免跨交易日串联。
- 滚动或滞后计算必须按 `ts_code` 分组，并确认不会使用未来 bar。
- 注意 `ctx.max_bars` 的内存保护，分钟线不要无界 collect 后再做大范围计算。

## 可复制代码

```python
"""分钟频示例因子：5 bar 动量。"""

from dataclasses import dataclass, field

import polars as pl

from factorzen.intraday.data.context import IntradayDataContext
from factorzen.intraday.factors.base import IntradayFactor


@dataclass
class MyIntradayAlpha(IntradayFactor):
    name: str = "my_intraday_alpha"
    description: str = "5 bar 动量：close[t] / close[t-5] - 1"
    bar_size: str = "1min"
    lookback_bars: int = 10
    required_data: list[str] = field(default_factory=lambda: ["minute"])

    def compute(self, ctx: IntradayDataContext) -> pl.DataFrame:
        return (
            ctx.minute.sort(["ts_code", "trade_time"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(5).over("ts_code") - 1.0)
                .alias("factor_value")
            )
            .filter(pl.col("trade_time").dt.date() >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_time", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            .collect()
        )


MyIntradayAlpha()
```

## 日内累计示例

VWAP、成交量占比这类因子通常需要按交易日重置累计值：

```python
frame = (
    ctx.minute.with_columns(pl.col("trade_time").dt.date().alias("_trade_date"))
    .with_columns(
        pl.col("amount").cum_sum().over(["ts_code", "_trade_date"]).alias("_cum_amount")
    )
)
```

## 验证

```bash
pixi run fz factor list --frequency intraday
```

## 检查点

- `trade_time` 必须是时间戳粒度，不能只输出日期。
- 日内累计量必须在每个交易日开盘后重新开始。
- 因子计算不要跨午休、收盘后或不同股票直接滚动。
- 分钟线数据量大，优先使用 LazyFrame 链式表达式，最后再 `.collect()`。

---

## 相关文档

- [因子编写指南](../../../docs/guides/factor-authoring.md) —— 完整接口说明、如何让手写因子进因子库
- [因子库与增量准入](../../../docs/concepts/factor-library.md) —— 因子入库的裁决机制
- [CLI 参考](../../../docs/reference/cli.md) —— `fz factor` 全部参数
