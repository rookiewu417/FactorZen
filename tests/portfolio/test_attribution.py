"""test_attribution_brinson.py：无 module docstring 的测试。
test_attribution_risk.py：风险因子归因测试：收益守恒 + M3 风险分解跨函数验证。
"""

import math

import numpy as np

from factorzen.attribution.brinson import BrinsonResult, brinson_attribution
from factorzen.attribution.risk_attribution import (
    RiskAttributionResult,
    risk_factor_attribution,
)
from factorzen.risk.exposures import ExposureMatrix


# ==== 来自 test_attribution_brinson.py ====
def test_brinson_conserves_to_excess():
    """配置 + 选股 ≈ 组合超额收益。"""
    # 4 只股票，2 行业(A/B)，单期
    port_w = np.array([0.4, 0.1, 0.4, 0.1])
    bench_w = np.array([0.25, 0.25, 0.25, 0.25])
    sectors = ["A", "A", "B", "B"]
    stock_ret = np.array([0.05, 0.03, -0.01, 0.02])
    res = brinson_attribution(port_w, bench_w, stock_ret, sectors)
    assert isinstance(res, BrinsonResult)
    port_ret = float(port_w @ stock_ret)
    bench_ret = float(bench_w @ stock_ret)
    excess = port_ret - bench_ret
    total = sum(res.allocation.values()) + sum(res.selection.values())
    assert math.isclose(total, excess, rel_tol=1e-9, abs_tol=1e-12)

def test_pure_allocation():
    """组合行业内选股与基准一致、仅行业权重不同 → 全是配置效应。"""
    port_w = np.array([0.3, 0.3, 0.2, 0.2])    # A 行业超配
    bench_w = np.array([0.25, 0.25, 0.25, 0.25])
    sectors = ["A", "A", "B", "B"]
    stock_ret = np.array([0.04, 0.04, 0.01, 0.01])  # 行业内同收益(无选股差异)
    res = brinson_attribution(port_w, bench_w, stock_ret, sectors)
    assert abs(sum(res.selection.values())) < 1e-9   # 选股效应≈0
    assert abs(sum(res.allocation.values())) > 1e-6   # 配置效应非0

def test_brinson_handles_none_sectors():
    """sectors 含 None 时不崩溃；None 归入 "" 行业；守恒仍成立。"""
    port_w = np.array([0.4, 0.1, 0.3, 0.2])
    bench_w = np.array([0.25, 0.25, 0.25, 0.25])
    sectors = ["A", None, "B", None]   # 两只股票 industry 为 null
    stock_ret = np.array([0.05, 0.03, -0.01, 0.02])

    # 不应抛出 TypeError
    res = brinson_attribution(port_w, bench_w, stock_ret, sectors)
    assert isinstance(res, BrinsonResult)

    # None 应归入 "" 行业，因此 "" 键必须存在
    assert "" in res.allocation
    assert "" in res.selection

    # 守恒：Σ(配置+选股) ≈ port_ret − bench_ret（用独立计算的值对账）
    port_ret = float(port_w @ stock_ret)
    bench_ret = float(bench_w @ stock_ret)
    excess = port_ret - bench_ret
    total = sum(res.allocation.values()) + sum(res.selection.values())
    assert math.isclose(total, excess, rel_tol=1e-9, abs_tol=1e-12)

# ==== 来自 test_attribution_risk.py ====
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
    # 特异收益用独立公式验证（非代数余项，有判别力）
    f_vec = np.array([factor_ret["size"], factor_ret["value"]])   # 顺序与 factor_names 一致
    Xw = r.factor_exposures.matrix.T @ w
    expected_specific = float(w @ stock_ret) - float(Xw @ f_vec)
    assert math.isclose(res.specific_return, expected_specific, rel_tol=1e-9)
    # 因子收益贡献 = 组合暴露 × 因子收益（有判别力）
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

