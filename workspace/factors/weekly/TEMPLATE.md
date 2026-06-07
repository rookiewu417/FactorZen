# 周频因子模板

周频因子放在 `workspace/factors/weekly/{factor_name}.py`。周频仍继承 `DailyFactor`，通过 `category = "weekly"` 和 `frequency = "weekly"` 区分。

## 编写约定

- 适合中低换手信号，例如 20 日动量、20 日波动率、换手变化。
- 多数周频因子应先在日频序列上完成滚动计算，再筛选 `ctx.snapshot_dates`。
- 如果公式只需要周频快照字段，可以直接使用 `ctx.weekly` 或 `ctx.weekly_basic`。
- 返回列仍是 `trade_date`、`ts_code`、`factor_value`。
- 不要用周末或自然日补齐数据，使用框架提供的交易日快照。

## 可复制代码

```python
"""周频示例因子：20 日波动率，最终只输出周频快照。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class MyWeeklyAlpha(DailyFactor):
    name = "my_weekly_alpha"
    category = "weekly"
    frequency = "weekly"
    required_data = ["daily"]
    lookback_days = 35
    description = "周频 20 日收益波动率：日频滚动计算后按周频快照采样"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        result = (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(
                (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code")).log()
                .alias("_log_ret")
            )
            .with_columns(
                pl.col("_log_ret")
                .rolling_std(20, min_samples=10)
                .over("ts_code")
                .alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            .collect()
        )
        return result.filter(pl.col("trade_date").is_in(ctx.snapshot_dates))


MyWeeklyAlpha()
```

## 验证

```bash
pixi run fz factor list
pixi run fz factor run my_weekly_alpha --frequency weekly --start 20230101 --end 20241231 --universe csi500
```

## 检查点

- 先日频计算、后周频采样，避免把滚动窗口误写成“滚动 20 周”。
- 使用 `ctx.snapshot_dates` 保持评估、换手和 IC 的频率一致。
- 若使用 `ctx.weekly_basic`，声明 `required_data = ["daily_basic"]`。
- 周频结果覆盖率低时，优先检查快照日期、停牌和 `lookback_days`。
