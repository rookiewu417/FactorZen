"""W1：风格因子一次物化与逐日重算数值等价；行业并集稳定化。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module
from factorzen.risk.exposures import (
    compute_exposures,
    materialize_industry_panel,
    materialize_style_panel,
)
from factorzen.risk.style_factors import cs_standardize


@pytest.fixture(autouse=True)
def _pit_off(monkeypatch):
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield


def _trade_days(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def _make(n_stocks=8, n_days=80, seed=42):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for c in codes for d in days
    ])
    db = pl.DataFrame([
        {
            "trade_date": d, "ts_code": c,
            "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
            "pb": float(abs(rng.standard_normal()) + 1.5),
            "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
            "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1),
        }
        for c in codes for d in days
    ])
    inds = ["银行", "医药", "电子", "食品饮料"]
    stocks = pl.DataFrame({
        "ts_code": codes,
        "industry": [inds[i % 4] for i in range(n_stocks)],
    })
    return daily, db, stocks, days


def test_materialize_style_panel_matches_per_day_compute():
    """全窗一次物化再切片 ≡ 逐日 compute_exposures 的风格列（atol 1e-12）。"""
    daily, db, stocks, days = _make()
    panel = materialize_style_panel(daily, db, standardize=True)
    # 静态风格：size/value 全窗必有
    target = days[-1]
    exp_day = compute_exposures(daily, db, stocks, target)  # 无 panel → 旧路径
    exp_panel = compute_exposures(
        daily, db, stocks, target, style_panel=panel
    )
    for name in ("size", "value", "liquidity", "quality", "leverage"):
        if name not in exp_day.factor_names or name not in exp_panel.factor_names:
            continue
        # 对齐 codes
        day_map = {c: i for i, c in enumerate(exp_day.codes)}
        pan_map = {c: i for i, c in enumerate(exp_panel.codes)}
        common = sorted(set(day_map) & set(pan_map))
        assert common, f"{name}: 无共同股票"
        i_d = exp_day.factor_names.index(name)
        i_p = exp_panel.factor_names.index(name)
        v_d = np.array([exp_day.matrix[day_map[c], i_d] for c in common])
        v_p = np.array([exp_panel.matrix[pan_map[c], i_p] for c in common])
        np.testing.assert_allclose(v_p, v_d, atol=1e-12, err_msg=f"style {name}")


def test_cs_standardize_is_by_trade_date_not_pooled():
    """查证：标准化按 trade_date 分组，两日均值各自≈0。"""
    df = pl.DataFrame({
        "trade_date": [dt.date(2024, 1, 2)] * 4 + [dt.date(2024, 1, 3)] * 4,
        "ts_code": [f"{i}.SZ" for i in range(4)] * 2,
        "factor_value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    })
    out = cs_standardize(df, "factor_value", method="mad")
    for d in (dt.date(2024, 1, 2), dt.date(2024, 1, 3)):
        m = out.filter(pl.col("trade_date") == d)["factor_value"].mean()
        assert abs(m) < 1e-10, f"date {d} mean={m}"


def test_industry_panel_union_fills_missing():
    """行业中途出现：并集面板缺列日为 0。"""
    stocks = pl.DataFrame({
        "ts_code": ["A", "B"],
        "industry": ["银行", "医药"],
    })
    # 无 PIT → 两日相同行业
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    panel, cols = materialize_industry_panel(stocks, dates)
    assert set(cols) == {"ind_医药", "ind_银行"}
    assert panel.filter(pl.col("trade_date") == dates[0]).height == 2
    # 强制更大并集
    panel2, cols2 = materialize_industry_panel(
        stocks, dates, industry_names=["银行", "医药", "新能源"]
    )
    assert "ind_新能源" in cols2
    day0 = panel2.filter(pl.col("trade_date") == dates[0])
    assert (day0["ind_新能源"] == 0.0).all()
