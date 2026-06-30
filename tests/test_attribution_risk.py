"""风险因子归因测试：收益守恒 + M3 风险分解跨函数验证。"""
import math

import numpy as np

from factorzen.attribution.risk_attribution import RiskAttributionResult, risk_factor_attribution
from factorzen.risk.exposures import ExposureMatrix


class _RiskResult:
    def __init__(self):
        names = ["size", "value"]
        X = np.array([[1.0, 0.5], [0.8, -0.3], [-0.2, 1.1]])
        self.factor_exposures = ExposureMatrix(["A", "B", "C"], names, X)
        self.factor_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        self.specific_risk = np.array([0.10, 0.15, 0.20])
        self.factor_names = names


def test_return_attribution_conserves():
    """因子收益贡献 + 特异 ≈ 组合收益(因子模型口径)。"""
    r = _RiskResult()
    w = np.array([0.5, 0.3, 0.2])
    factor_ret = {"size": 0.02, "value": -0.01}   # 最新一期因子收益
    stock_ret = np.array([0.03, 0.01, -0.02])      # 个股实际收益
    res = risk_factor_attribution(w, r, factor_ret, stock_returns=stock_ret)
    assert isinstance(res, RiskAttributionResult)
    # 组合收益 = Σ 因子贡献 + 特异
    port_ret = float(w @ stock_ret)
    total_attrib = sum(res.factor_return_contrib.values()) + res.specific_return
    assert math.isclose(total_attrib, port_ret, rel_tol=1e-9)
    # 因子收益贡献 = 组合暴露 × 因子收益
    Xw = r.factor_exposures.matrix.T @ w
    assert math.isclose(res.factor_return_contrib["size"], Xw[0] * 0.02, rel_tol=1e-9)


def test_risk_contrib_matches_m3_decompose():
    """风险贡献与 M3 decompose_risk 一致(跨函数验证,非恒真)。"""
    from factorzen.risk.model import RiskModel, RiskModelResult
    r = _RiskResult()
    rr = RiskModelResult(factor_exposures=r.factor_exposures, factor_covariance=r.factor_covariance,
                         specific_risk=r.specific_risk, factor_names=r.factor_names)
    w = np.array([0.5, 0.3, 0.2])
    res = risk_factor_attribution(w, rr, {"size": 0.0, "value": 0.0},
                                  stock_returns=np.zeros(3))
    m3 = RiskModel().decompose_risk(w, rr)
    assert math.isclose(res.factor_risk_contrib["size"], m3["size"], rel_tol=1e-9)
