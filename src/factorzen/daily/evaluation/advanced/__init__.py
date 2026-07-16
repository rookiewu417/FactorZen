"""高级单因子评估指标（按分析拆分为子模块，对外接口不变）。

保留：
1. Monotonicity — 分位收益单调性（``monotonicity``）
2. Factor Crowding — 因子拥挤度检测（``crowding``）
3. Factor Correlation — 多因子截面 Rank 相关性 + FDR（``correlation``）

已摘除：IC Decay 独立模块、行业/市值分层 IC、市场状态 IC、排名自相关、
中性化 IC、事件研究（报告层改用 ``ic_analysis`` 内置 multi_period/decay）。
"""

from __future__ import annotations

from factorzen.daily.evaluation.advanced.correlation import (
    apply_fdr_correction,
    compute_factor_correlation,
)
from factorzen.daily.evaluation.advanced.crowding import (
    CrowdingResult,
    compute_factor_crowding,
)
from factorzen.daily.evaluation.advanced.monotonicity import (
    MonotonicityResult,
    compute_monotonicity,
)

__all__ = [
    "CrowdingResult",
    "MonotonicityResult",
    "apply_fdr_correction",
    "compute_factor_correlation",
    "compute_factor_crowding",
    "compute_monotonicity",
]
