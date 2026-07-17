"""Barra 多因子风险模型。

提供因子暴露计算、因子协方差估计、特质风险估计与风险分解功能。
"""

from __future__ import annotations

from factorzen.risk.covariance import (
    eigenvector_adjustment,
    estimate_factor_covariance,
    estimate_specific_risk,
)
from factorzen.risk.exposures import (
    ExposureMatrix,
    compute_exposures,
    materialize_industry_panel,
    materialize_style_panel,
    reindex_exposure,
    standardize_style_panel,
)
from factorzen.risk.industry_factors import get_industry_dummies
from factorzen.risk.model import RiskModel, RiskModelResult
from factorzen.risk.style_factors import (
    STYLE_FACTOR_NAMES,
    STYLE_FACTOR_REGISTRY,
    cs_standardize,
)

__all__ = [
    "STYLE_FACTOR_NAMES",
    "STYLE_FACTOR_REGISTRY",
    "ExposureMatrix",
    "RiskModel",
    "RiskModelResult",
    "compute_exposures",
    "cs_standardize",
    "eigenvector_adjustment",
    "estimate_factor_covariance",
    "estimate_specific_risk",
    "get_industry_dummies",
    "materialize_industry_panel",
    "materialize_style_panel",
    "reindex_exposure",
    "standardize_style_panel",
]
