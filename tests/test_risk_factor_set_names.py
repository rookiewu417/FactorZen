"""行业因子集稳定化（W2）：全窗并集 + 缺列 0，中途出现新行业不再丢日。

旧逻辑：锁定首个有效截面的因子名，后续日因 ind_* 漂移被静默跳过。
新逻辑：面板并集 + reindex 缺列 0，保留有效回归日。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from factorzen.risk.exposures import ExposureMatrix, reindex_exposure
from factorzen.risk.model import RiskModel


def _make_daily(dates, codes):
    rng = np.random.default_rng(3)
    return pl.DataFrame([
        {"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for d in dates for c in codes
    ])


def test_reindex_exposure_fills_missing_industry_with_zero():
    """缺列填 0、列序对齐固定全集。"""
    exp = ExposureMatrix(
        codes=["A", "B"],
        factor_names=["size", "ind_A"],
        matrix=np.array([[1.0, 1.0], [0.5, 0.0]]),
    )
    fixed = ["size", "ind_A", "ind_C"]
    out = reindex_exposure(exp, fixed)
    assert out.factor_names == fixed
    assert out.matrix.shape == (2, 3)
    np.testing.assert_allclose(out.matrix[:, 0], [1.0, 0.5])
    np.testing.assert_allclose(out.matrix[:, 1], [1.0, 0.0])
    np.testing.assert_allclose(out.matrix[:, 2], [0.0, 0.0])


def test_industry_mid_window_appearance_kept_with_zero_fill():
    """合成场景：某行业中途出现——旧逻辑丢日、新逻辑保留且缺列 0。

    前几日仅 ind_A/ind_B 非零，末日 ind_C 出现；面板并集含三列，早期 ind_C=0。
    """
    dates = [dt.date(2024, 1, i) for i in range(2, 8)]  # 6 个交易日
    codes = [f"{i:06d}.SZ" for i in range(6)]
    daily = _make_daily(dates, codes)
    db = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "total_mv": 5e9, "pb": 1.5, "pe_ttm": 15.0}
        for d in dates for c in codes
    ])
    stocks = pl.DataFrame({"ts_code": codes, "industry": ["银行"] * 6})
    rng = np.random.default_rng(0)

    style_panel = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "size": rng.standard_normal(len(dates) * len(codes)).tolist(),
    })
    # 行业面板：全窗并集含 ind_C；早期 ind_C=0，末日 ind_C=1
    industry_panel = pl.DataFrame([
        {
            "trade_date": d,
            "ts_code": c,
            "ind_A": 0.0 if d == dates[-1] else 1.0,
            "ind_B": 0.0,
            "ind_C": 1.0 if d == dates[-1] else 0.0,
        }
        for d in dates for c in codes
    ])

    result = RiskModel().build(
        daily, db, stocks, "20240102", "20240107",
        style_panel=style_panel,
        industry_panel=industry_panel,
        industry_names=["ind_A", "ind_B", "ind_C"],
    )

    assert result.n_factor_mismatch == 0, (
        f"行业中途出现不应再触发 factor_mismatch，实得 {result.n_factor_mismatch}"
    )
    assert result.n_valid_dates == len(dates), (
        f"全部 {len(dates)} 日应保留，实得 n_valid={result.n_valid_dates}"
    )
    assert "ind_C" in result.factor_names
    assert "ind_C" in result.factor_returns.columns
    # 早期日也有 ind_C 因子收益列（对应暴露为 0）
    assert result.factor_returns.filter(pl.col("trade_date") == dates[0]).height == 1


def test_n_factor_mismatch_visible_on_result():
    """退化可见性：n_factor_mismatch / n_valid_dates 字段存在且默认可读。"""
    dates = [dt.date(2024, 1, i) for i in range(2, 6)]
    codes = [f"{i:06d}.SZ" for i in range(8)]
    daily = _make_daily(dates, codes)
    db = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "total_mv": 5e9, "pb": 1.5, "pe_ttm": 15.0}
        for d in dates for c in codes
    ])
    stocks = pl.DataFrame({
        "ts_code": codes,
        "industry": (["银行", "医药"] * 4),
    })
    rng = np.random.default_rng(2)
    style_panel = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "size": rng.standard_normal(len(dates) * len(codes)).tolist(),
        "value": rng.standard_normal(len(dates) * len(codes)).tolist(),
    })
    industry_panel = pl.DataFrame([
        {
            "trade_date": d,
            "ts_code": c,
            "ind_银行": 1.0 if stocks.filter(pl.col("ts_code") == c)["industry"][0] == "银行" else 0.0,
            "ind_医药": 1.0 if stocks.filter(pl.col("ts_code") == c)["industry"][0] == "医药" else 0.0,
        }
        for d in dates for c in codes
    ])
    result = RiskModel().build(
        daily, db, stocks, "20240102", "20240105",
        style_panel=style_panel,
        industry_panel=industry_panel,
    )
    assert hasattr(result, "n_factor_mismatch")
    assert hasattr(result, "n_valid_dates")
    assert result.n_factor_mismatch == 0
    assert result.n_valid_dates == len(dates)
