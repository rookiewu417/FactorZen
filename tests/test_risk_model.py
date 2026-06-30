# tests/test_risk_model.py
import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModel, RiskModelResult


@pytest.fixture(autouse=True)
def _pit_industry_unavailable_by_default(monkeypatch):
    """RiskModel.build() 内部循环调用 compute_exposures：默认不触达真实 Tushare，
    PIT 历史行业数据视为不可用，走现有 stocks.industry 降级路径（行为与改造
    PIT 行业暴露之前完全一致）。"""
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield


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

    # Fix 2: 手算期望值，验证 predict_risk 公式正确性
    X = result.factor_exposures.matrix
    F = result.factor_covariance
    D = result.specific_risk
    Xw = X.T @ w
    factor_var = float(Xw @ F @ Xw)
    spec_var = float(np.sum((D * w) ** 2))
    expected = np.sqrt(factor_var + spec_var) * np.sqrt(252)
    assert math.isclose(risk, expected, rel_tol=1e-9), (
        f"predict_risk={risk} 与手算 expected={expected} 不一致，公式有误"
    )


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

    # Fix 1: 跨函数验证 decompose.total_risk == predict_risk（F 用错/转置错会被抓）
    assert math.isclose(
        d["total_risk"], RiskModel().predict_risk(w, result), rel_tol=1e-9
    ), "decompose_risk 的 total_risk 与 predict_risk 不一致，两者公式不同步"

    # Fix 3: per-factor 贡献语义验证
    # MCR 分解：risk_contrib_k = Xw[k]*(F@Xw)[k] / total_var * total_std * sqrt(252)
    # 数学上：sum_k(risk_contrib_k) = factor_var / total_var * total_risk
    #                               = factor_risk² / total_risk
    # （这是加权 MCR 分解，非欧拉分解到 total_risk）
    per_factor_sum = sum(d[n] for n in result.factor_names)
    expected_factor_sum = d["factor_risk"] ** 2 / d["total_risk"]
    assert math.isclose(per_factor_sum, expected_factor_sum, rel_tol=1e-9), (
        f"per-factor 贡献之和 {per_factor_sum} != factor_risk²/total_risk {expected_factor_sum}"
    )
    # 至少一个因子贡献非零（避免退化）
    assert any(abs(d[n]) > 1e-12 for n in result.factor_names), (
        "所有因子贡献均为零，分解结果退化"
    )


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

    # Fix 4: 因子协方差矩阵半正定（核心数学约束）
    assert np.linalg.eigvalsh(result.factor_covariance).min() >= -1e-8, (
        "factor_covariance 不是半正定矩阵，风险模型协方差估计有误"
    )


def test_build_residual_matrix_mid_window_gap_no_misalignment():
    """回归测试：股票在窗口第3期(非首尾)缺失时，重建残差矩阵不能把缺口前的残差
    整体右移一位。

    历史实现"取最后 T_valid 个，右对齐"拼接残差：股票 B 在 5 期窗口里只有 4 期
    数据（第3期停牌缺失），右对齐会把第1、2期残差错位推到第2、3行，
    且把本应是第1期残差的第0行错误置为 NaN（真正的缺口在第2行，反而被掩盖）。
    正确实现必须按真实交易日索引对齐：第1、2、4、5期残差落在各自正确的行，
    第3期（缺口）显式为 NaN。
    """
    from factorzen.risk.model import _build_residual_matrix

    days = [dt.date(2023, 1, 2 + i) for i in range(5)]  # d1..d5，5 期窗口
    residual_dict = {
        "A": [(days[0], 0.1), (days[1], 0.2), (days[2], 0.3), (days[3], 0.4), (days[4], 0.5)],
        # B 第3期(days[2])缺失（停牌/无收益等），非首尾
        "B": [(days[0], 1.1), (days[1], 1.2), (days[3], 1.4), (days[4], 1.5)],
    }
    codes = ["A", "B"]

    matrix = _build_residual_matrix(residual_dict, codes, days)

    assert matrix.shape == (5, 2)
    # A 全勤：5 期精确对应，不受 B 缺口影响
    np.testing.assert_allclose(matrix[:, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
    # B：第1、2、4、5期落在各自正确的行，未被缺口向后挤压一位
    assert math.isclose(matrix[0, 1], 1.1, rel_tol=1e-12)
    assert math.isclose(matrix[1, 1], 1.2, rel_tol=1e-12)
    assert math.isclose(matrix[3, 1], 1.4, rel_tol=1e-12)
    assert math.isclose(matrix[4, 1], 1.5, rel_tol=1e-12)
    # 第3期(缺口本身)应为 NaN，而不是被其他期残差顶替
    assert math.isnan(matrix[2, 1]), f"缺口行应为 NaN，实际: {matrix[2, 1]}"
