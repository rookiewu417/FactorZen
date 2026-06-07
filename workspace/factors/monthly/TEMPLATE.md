# 月频因子模板

月频因子放在 `workspace/factors/monthly/{factor_name}.py`。月频仍继承 `DailyFactor`，通过 `category = "monthly"` 和 `frequency = "monthly"` 区分。

## 编写约定

- 适合估值、质量、盈利、成长、低换手风格类信号。
- 用估值或市值字段时，通常声明 `required_data = ["daily_basic"]` 并读取 `ctx.monthly_basic`。
- 用价格字段时，声明 `required_data = ["daily"]` 并读取 `ctx.monthly`，或先日频滚动计算再筛选月频快照。
- 返回列仍是 `trade_date`、`ts_code`、`factor_value`。
- 财务报表类因子必须确认字段是当时可得数据，不能用未来披露值回填。

## 可复制代码

```python
"""月频示例因子：EP，即 PE-TTM 的倒数。"""

import polars as pl

from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class MyMonthlyAlpha(DailyFactor):
    name = "my_monthly_alpha"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 10
    description = "月频 EP：1 / pe_ttm，使用 monthly_basic 快照"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        return (
            ctx.monthly_basic.filter(pl.col("pe_ttm").is_not_null() & (pl.col("pe_ttm") > 0))
            .with_columns((1.0 / pl.col("pe_ttm")).alias("factor_value"))
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            .collect()
        )


MyMonthlyAlpha()
```

## 验证

```bash
pixi run fz factor list
pixi run fz factor run my_monthly_alpha --frequency monthly --start 20230101 --end 20241231 --universe csi500
```

## 检查点

- 估值类因子常需要确认正负号，例如 PE 越低越便宜时，可用 `1 / pe_ttm` 或 `-pe_ttm`。
- 月频信号不要输出日频全量结果，否则换手和 IC 会被按日频解释。
- 如果使用 `daily_basic` 中的市值暴露，是否还需要中性化应交给 YAML 配置决定。
- 财务字段必须有披露日或可得性约束，没有把握时不要把它当成月末已知。
