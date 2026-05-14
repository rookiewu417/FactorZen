"""因子公共抽象基类。所有频率的因子基类均继承此类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import polars as pl


class BaseFactor(ABC):
    """跨频率共享的因子基础能力：name/frequency/description 字段 + validate()。

    子类需覆盖 name、frequency、description，并实现 compute()。
    """

    name: str = ""
    frequency: str = ""
    description: str = ""

    @abstractmethod
    def compute(self, ctx: Any) -> pl.DataFrame:
        """计算因子值，返回含 factor_value 列的 DataFrame。"""
        ...

    def validate(self, result: pl.DataFrame, time_col: str = "trade_date") -> dict[str, Any]:
        """校验因子计算结果，返回覆盖率统计与预警信息。

        Args:
            result: compute() 的返回值，必须含 factor_value 列。
            time_col: 时间列名，日频用 "trade_date"，分钟频用 "trade_time"。

        Returns:
            dict，含 coverage / n_stocks / n_periods / null_count / inf_count / warnings。
        """
        if result.is_empty():
            return {"error": "Empty DataFrame", "warnings": ["Empty result"]}

        total = len(result)
        has_fv = "factor_value" in result.columns
        null_count = result["factor_value"].null_count() if has_fv else 0
        inf_count = (
            result.filter(pl.col("factor_value").is_infinite()).height if has_fv else 0
        )
        n_stocks = result["ts_code"].n_unique() if "ts_code" in result.columns else 0
        n_periods = result[time_col].n_unique() if time_col in result.columns else 0
        coverage = (total - null_count) / total if total > 0 else 0.0

        warnings: list[str] = []
        if coverage < 0.8:
            warnings.append(f"Low coverage: {coverage:.1%}")
        if inf_count > 0:
            warnings.append(f"Found {inf_count} infinite values")
        if n_periods == 0:
            warnings.append(f"No {time_col} found in result")

        return {
            "coverage": coverage,
            "n_stocks": n_stocks,
            "n_periods": n_periods,
            "null_count": null_count,
            "inf_count": inf_count,
            "warnings": warnings,
        }
