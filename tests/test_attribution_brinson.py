import math

import numpy as np

from factorzen.attribution.brinson import BrinsonResult, brinson_attribution


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
