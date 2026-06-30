# tests/test_risk_model.py
import datetime as dt
import math

import numpy as np
import polars as pl

from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModel, RiskModelResult


def _toy_result():
    """手搓一个 RiskModelResult，绕开截面回归，做确定性 predict/decompose 验证。"""
    codes = ["A", "B", "C"]
    factor_names = ["size", "value"]
    X = np.array([[1.0, 0.5], [0.8, -0.3], [-0.2, 1.1]])  # (3 stocks, 2 factors)
    F = np.array([[0.04, 0.01], [0.01, 0.09]])             # (2,2) 因子协方差
    D = np.array([0.10, 0.15, 0.20])                       # (3,) 特质风险（std）
    exp = ExposureMatrix(codes=codes, factor_names=factor_names, matrix=X)
    return RiskModelResult(factor_exposures=exp, factor_covariance=F,
                           specific_risk=D, factor_names=factor_names)


def test_predict_risk_positive():
    result = _toy_result()
    w = np.array([0.5, 0.3, 0.2])
    risk = RiskModel().predict_risk(w, result)
    assert risk > 0


def test_decompose_risk_variance_conservation():
    """factor_risk² + specific_risk² ≈ total_risk²（方差可加）。"""
    result = _toy_result()
    w = np.array([0.5, 0.3, 0.2])
    d = RiskModel().decompose_risk(w, result)
    assert {"total_risk", "factor_risk", "specific_risk"} <= set(d)
    assert math.isclose(d["factor_risk"]**2 + d["specific_risk"]**2,
                        d["total_risk"]**2, rel_tol=1e-9)
    # 每个因子名都有一个贡献键
    assert "size" in d and "value" in d


def test_build_end_to_end_r_squared_in_range():
    """端到端 build（mock 数据，n_days≥280 让 momentum 有值）→ R²∈[0,1]。"""
    rng = np.random.default_rng(7)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < 290:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(12)]
    daily = pl.DataFrame([{"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
                          for c in codes for dd in days])
    db = pl.DataFrame([{"trade_date": dd, "ts_code": c,
                        "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                        "pb": float(abs(rng.standard_normal()) + 1.5),
                        "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15)}
                       for c in codes for dd in days])
    stocks = pl.DataFrame({"ts_code": codes,
                           "industry": [["银行", "医药", "电子"][i % 3] for i in range(12)]})
    start = days[260].strftime("%Y%m%d")
    end = days[-1].strftime("%Y%m%d")
    result = RiskModel().build(daily, db, stocks, start, end)
    assert 0.0 <= result.r_squared <= 1.0
    assert result.factor_covariance.shape[0] == result.factor_covariance.shape[1]
    assert len(result.factor_names) > 0
