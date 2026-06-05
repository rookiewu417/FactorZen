"""高级单因子评估指标（按分析拆分为子模块，对外接口不变）。

包含：
1. IC Decay 增强分析 — 多持有期 IC 衰减（``ic_decay``）
2. Monotonicity — 分位收益单调性（``monotonicity``）
3. Sector-stratified IC — 行业分层 IC（``sector_ic``）
4. Size-stratified IC — 市值分层 IC（``size_ic``）
5. Factor Crowding — 因子拥挤度检测（``crowding``）
6. Market Regime IC — 市场状态分层 IC（``market_regime_ic``）
7. Rank Autocorrelation — 因子排名自相关（``rank_autocorr``）
8. Neutralized IC — 中性化后的 Rank IC（``neutralized_ic``）
9. Event Study — 事件前后窗口累计收益（``event_study``）
10. Factor Correlation — 多因子截面 Rank 相关性 + FDR（``correlation``）

历史上这些指标集中在单个 ``advanced.py``；现按分析拆分以便导航，
通过本 ``__init__`` re-export 保持 ``from factorzen.daily.evaluation.advanced import X`` 不变。
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
from factorzen.daily.evaluation.advanced.event_study import (
    EventStudyResult,
    compute_event_study,
)
from factorzen.daily.evaluation.advanced.ic_decay import (
    ICDecayResult,
    compute_ic_decay,
)
from factorzen.daily.evaluation.advanced.market_regime_ic import (
    MarketRegimeICResult,
    compute_market_regime_ic,
)
from factorzen.daily.evaluation.advanced.monotonicity import (
    MonotonicityResult,
    compute_monotonicity,
)
from factorzen.daily.evaluation.advanced.neutralized_ic import compute_neutralized_ic
from factorzen.daily.evaluation.advanced.rank_autocorr import (
    RankAutocorrResult,
    compute_rank_autocorr,
)
from factorzen.daily.evaluation.advanced.sector_ic import (
    SectorICResult,
    compute_sector_ic,
)
from factorzen.daily.evaluation.advanced.size_ic import (
    SizeICResult,
    compute_size_ic,
)

__all__ = [
    "CrowdingResult",
    "EventStudyResult",
    "ICDecayResult",
    "MarketRegimeICResult",
    "MonotonicityResult",
    "RankAutocorrResult",
    "SectorICResult",
    "SizeICResult",
    "apply_fdr_correction",
    "compute_event_study",
    "compute_factor_correlation",
    "compute_factor_crowding",
    "compute_ic_decay",
    "compute_market_regime_ic",
    "compute_monotonicity",
    "compute_neutralized_ic",
    "compute_rank_autocorr",
    "compute_sector_ic",
    "compute_size_ic",
]
