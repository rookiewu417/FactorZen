"""数据契约校验:对评估输入做前置列检查,缺列时早失败并给出清晰错误。

报告与评估链路整体偏防御(缺失即降级),但**数据准备入口**应当 fail-fast——
畸形输入若被静默吞掉,会产出看似正常实则空洞的结论。此处提供可复用的列契约校验。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def require_columns(df: Any, required: Iterable[str], *, context: str = "") -> None:
    """校验 ``df`` 含全部 ``required`` 列,缺失则抛 ``ValueError``。

    错误信息同时列出缺失列与实际列,便于快速定位 schema 问题。

    Args:
        df: 任意含 ``.columns`` 的表(polars / pandas DataFrame)。
        required: 必需列名集合。
        context: 错误信息前缀,标明校验场景(如函数名)。
    """
    columns = list(getattr(df, "columns", []) or [])
    missing = [c for c in required if c not in columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}缺少必需列 {missing};实际列为 {columns}")
