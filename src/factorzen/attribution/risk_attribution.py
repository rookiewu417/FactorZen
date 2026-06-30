"""风险因子归因：基于 M3，把组合收益/风险分解到风格因子 + 特异。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorzen.risk.model import RiskModel


@dataclass
class RiskAttributionResult:
    factor_return_contrib: dict[str, float]   # 各因子收益贡献 = (Xᵀw)_j × f_j
    factor_risk_contrib: dict[str, float]     # 各因子风险贡献(M3 MCR)
    specific_return: float
    specific_risk: float


def risk_factor_attribution(
    weights,
    risk_result,
    factor_returns_latest: dict,
    *,
    stock_returns,
) -> RiskAttributionResult:
    """收益归因：因子贡献 = 暴露×因子收益；特异 = 组合实际收益 − Σ因子贡献。
    风险归因：复用 M3 decompose_risk。"""
    w = np.asarray(weights)
    X = risk_result.factor_exposures.matrix       # (n, k)
    names = risk_result.factor_names
    Xw = X.T @ w                                  # (k,) 组合因子暴露
    factor_ret_contrib = {
        names[j]: float(Xw[j] * factor_returns_latest.get(names[j], 0.0))
        for j in range(len(names))
    }
    port_ret = float(w @ np.asarray(stock_returns))
    specific_return = port_ret - sum(factor_ret_contrib.values())
    # 风险贡献复用 M3
    decomp = RiskModel().decompose_risk(w, risk_result)
    factor_risk_contrib = {n: float(decomp.get(n, 0.0)) for n in names}
    return RiskAttributionResult(
        factor_return_contrib=factor_ret_contrib,
        factor_risk_contrib=factor_risk_contrib,
        specific_return=specific_return,
        specific_risk=float(decomp.get("specific_risk", 0.0)),
    )
