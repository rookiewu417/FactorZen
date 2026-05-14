"""分钟频因子抽象基类。所有 intraday 因子必须继承此类并实现 compute() 方法。"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import polars as pl

from common.factor import BaseFactor

if TYPE_CHECKING:
    from intraday.data.context import MFTDataContext


@dataclass
class MFTFactor(BaseFactor):
    """分钟频因子基类（历史名称保留，继承自 BaseFactor）。

    子类必须设置 name、bar_size，并实现 compute() 方法。
    """

    name: str = ""
    frequency: str = "minute"
    bar_size: str = "1min"
    required_data: list[str] = field(default_factory=lambda: ["minute"])
    lookback_bars: int = 500
    description: str = ""

    @abstractmethod
    def compute(self, ctx: "MFTDataContext") -> pl.DataFrame:
        """计算因子值，返回: trade_time, ts_code, factor_value"""
        ...

    def validate(self, result: pl.DataFrame, time_col: str = "trade_time") -> dict[str, Any]:
        return super().validate(result, time_col=time_col)
