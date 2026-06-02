"""日频因子抽象基类。所有 daily 因子必须继承此类并实现 compute() 方法。"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from factorzen.core.factor import BaseFactor

if TYPE_CHECKING:
    from factorzen.daily.data.context import FactorDataContext


@dataclass
class DailyFactor(BaseFactor):
    """日/周/月频因子基类，继承自 BaseFactor。"""

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

