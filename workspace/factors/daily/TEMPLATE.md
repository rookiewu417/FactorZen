# 日频因子模板

日频因子放在 `workspace/factors/daily/{factor_name}.py`。复制下面代码后，先改 `name`、类名、`description`、`lookback_days` 和公式，再运行验证命令。

## 编写约定

- 继承 `DailyFactor`。
- `category = "daily"`，`frequency` 可以省略或显式设为 `"daily"`。
- `required_data` 按实际使用声明，常见值是 `["daily"]` 或 `["daily", "daily_basic"]`。
- `compute(ctx)` 返回 Polars `DataFrame`，至少包含 `trade_date`、`ts_code`、`factor_value`。
- 有复权价格需求时优先使用 `close_adj`、`open_adj`、`high_adj`、`low_adj`。
- 所有 `shift`、`rolling_*` 必须 `.over("ts_code")`，避免跨股票串号。
- 用 `ctx.start` 过滤预热期，只输出请求区间。
- 行业、市值中性化放在 YAML 的 `preprocessing` 配置里，不要写进因子本身。

## 可复制代码

```python
"""日频示例因子：20 日复权动量。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class MyDailyAlpha(DailyFactor):
    name = "my_daily_alpha"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 30
    description = "20 日复权动量：close_adj[t] / close_adj[t-20] - 1"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        return (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(20).over("ts_code") - 1.0)
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            .collect()
        )


MyDailyAlpha()
```

## 验证

```bash
pixi run fz factor list
pixi run fz factor run my_daily_alpha --start 20230101 --end 20241231 --universe csi500
```

## 检查点

- `lookback_days` 大于最长回看窗口，给停牌和节假日留余量。
- 因子值只使用当前日期及以前可获得的数据。
- 输出列名固定为 `factor_value`，不要输出多个因子列。
- 如果公式依赖估值、市值或换手字段，先把对应数据类型加入 `required_data`。

---

## 相关文档

- [因子编写指南](../../../docs/guides/factor-authoring.md) —— 完整接口说明、如何让手写因子进因子库
- [因子库与增量准入](../../../docs/concepts/factor-library.md) —— 因子入库的裁决机制
- [CLI 参考](../../../docs/reference/cli.md) —— `fz factor` 全部参数
