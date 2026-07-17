"""W3：research 风格面板复用与单独 build(start,d) 的 PIT 等价。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module
from factorzen.risk.exposures import materialize_style_panel, standardize_style_panel
from factorzen.risk.model import RiskModel


@pytest.fixture(autouse=True)
def _pit_off(monkeypatch):
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield


def _mock_long(n_stocks=10, n_days=120, seed=5):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([
        {"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for c in codes for dd in days
    ])
    db = pl.DataFrame([
        {
            "trade_date": dd, "ts_code": c,
            "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
            "pb": float(abs(rng.standard_normal()) + 1.5),
            "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
            "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1),
        }
        for c in codes for dd in days
    ])
    stocks = pl.DataFrame({
        "ts_code": codes,
        "industry": [["银行", "医药", "电子"][i % 3] for i in range(n_stocks)],
    })
    return daily, db, stocks, days


def test_research_style_reuse_matches_standalone_build():
    """全窗 raw 物化 → 按 ≤d + universe 标准化 再 build，≡ 单独 build(start,d)。"""
    daily, db, stocks, days = _mock_long()
    start_d = days[60]
    rebal_d = days[100]
    start = start_d.strftime("%Y%m%d")
    d_str = rebal_d.strftime("%Y%m%d")
    codes = stocks["ts_code"].to_list()

    # standalone
    daily_d = daily.filter(
        (pl.col("trade_date") <= rebal_d) & pl.col("ts_code").is_in(codes)
    )
    db_d = db.filter(
        (pl.col("trade_date") <= rebal_d) & pl.col("ts_code").is_in(codes)
    )
    standalone = RiskModel().build(daily_d, db_d, stocks, start, d_str)

    # research reuse path
    raw = materialize_style_panel(daily, db, standardize=False)
    style_d = standardize_style_panel(
        raw.filter(
            (pl.col("trade_date") <= rebal_d) & pl.col("ts_code").is_in(codes)
        )
    )
    reused = RiskModel().build(
        daily_d, db_d, stocks, start, d_str, style_panel=style_d
    )

    assert standalone.n_valid_dates == reused.n_valid_dates
    assert standalone.factor_names == reused.factor_names
    assert abs(standalone.r_squared - reused.r_squared) < 1e-10

    # 协方差逐值
    np.testing.assert_allclose(
        standalone.factor_covariance, reused.factor_covariance, atol=1e-12
    )
    # 末日期暴露
    np.testing.assert_allclose(
        standalone.factor_exposures.matrix,
        reused.factor_exposures.matrix,
        atol=1e-12,
    )
    assert standalone.factor_exposures.codes == reused.factor_exposures.codes
