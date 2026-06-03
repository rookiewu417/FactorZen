"""报告通用工具:有限数判定、指标格式化、安全取值、数值钳制。

无项目内部依赖,供 reports 包内各模块共享,避免在巨石文件中重复定义。
"""

from typing import Any

import numpy as np


def _finite_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _format_metric_number(value: Any, digits: int = 4, empty: str = "样本不足") -> str:
    numeric = _finite_float(value)
    if numeric is None:
        return empty
    text = f"{numeric:.{int(digits)}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def _format_metric_percent(value: Any, digits: int = 1, empty: str = "样本不足") -> str:
    numeric = _finite_float(value)
    if numeric is None:
        return empty
    text = f"{numeric * 100:.{int(digits)}f}"
    return f"{text[1:] if text.startswith('-') and float(text) == 0.0 else text}%"


def _is_finite_metric(value: Any) -> bool:
    return _finite_float(value) is not None


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """安全获取对象属性。"""
    if obj is not None and hasattr(obj, attr):
        return getattr(obj, attr)
    return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _num(value: Any, default: float = 0.0) -> float:
    numeric = _finite_float(value)
    return default if numeric is None else numeric


def _same_direction(a: float, b: float) -> bool:
    return a == 0 or b == 0 or (a > 0) == (b > 0)
