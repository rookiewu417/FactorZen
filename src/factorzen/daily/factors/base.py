"""日频因子抽象基类。所有 daily 因子必须继承此类并实现 compute() 方法。"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from factorzen.core.factor import BaseFactor

if TYPE_CHECKING:
    from factorzen.daily.data.context import FactorDataContext


class DailyFactor(BaseFactor):
    """日/周/月频因子基类，继承自 BaseFactor。

    这里**刻意不用 `@dataclass`**：子类一律以无注解的类属性声明
    `name` / `category` / `frequency` / `lookback_days`（见 workspace/factors 模板）。
    无注解的属性不会成为 dataclass 字段，一旦本类是 dataclass，生成的 `__init__`
    就会在实例化时用下面这些默认值把子类声明覆盖掉，而消费方读的都是实例属性
    （pipelines/daily_single.py、discovery/python_factor.py），导致预热窗口
    静默退化成 20 天。保持普通类，属性查找按正常 MRO 走。
    """

    category: str = "daily"
    frequency: str = "daily"
    required_data: ClassVar[list[str]] = ["daily"]
    lookback_days: int = 20

    @abstractmethod
    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        """计算因子值，返回列: trade_date, ts_code, factor_value"""
        ...

    def validate(self, result: pl.DataFrame, time_col: str = "trade_date") -> dict[str, Any]:
        return super().validate(result, time_col=time_col)

