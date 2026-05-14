# daily/factors/custom/ — 用户自定义因子

## 用途

在此目录放置不属于标准因子库的**自定义因子**，例如：
- 实验性因子（还未经过充分评估）
- 特定策略专用因子
- 外部研究复现

## 命名规则

- 文件名：`{factor_name}.py`，全小写 + 下划线，例如 `my_momentum.py`
- 类名：大驼峰，例如 `MyMomentum`
- `name` 属性：与文件名一致，例如 `name = "my_momentum"`

## 最简模板

```python
# daily/factors/custom/my_factor.py
import polars as pl
from daily.factors.base import LFTFactor
from daily.data.context import FactorDataContext


class MyFactor(LFTFactor):
    name = "my_factor"
    category = "daily"          # "daily" / "weekly" / "monthly"
    description = "一句话描述"
    lookback_days = 25          # 计算需要的历史回看天数

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        daily = ctx.daily       # pl.LazyFrame，含 trade_date/ts_code/close/vol/amount...
        result = (
            daily
            .sort(["ts_code", "trade_date"])
            .with_columns(
                # 在此计算因子值
                pl.lit(0.0).alias("factor_value")
            )
            .filter(pl.col("trade_date") >= pl.lit(ctx.start).str.strptime(pl.Date, "%Y%m%d"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )
        return result


MyFactor()  # 模块级实例化，registry 自动发现需要此行
```

## 注意事项

- **必须有 `Name()` 模块级实例化**，否则 registry 不会发现该因子
- `compute()` 返回的 DataFrame 必须包含 `trade_date`, `ts_code`, `factor_value` 三列
- `factor_value` 为原始值，后续由 pipeline 处理（去极值 / 标准化 / 中性化）
- 自定义因子不会自动包含在测试中，请自行在 `tests/` 中补充单测
