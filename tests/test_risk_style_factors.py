# tests/test_risk_style_factors.py
import datetime as dt

import numpy as np
import polars as pl


def _trade_days(start, n):
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def make_daily_basic(n_stocks=8, n_days=10, seed=0):
    rng = np.random.default_rng(seed)
    days = _trade_days(dt.date(2023, 1, 3), n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for d in days:
            rows.append({"trade_date": d, "ts_code": c,
                         "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                         "pb": float(abs(rng.standard_normal()) + 1.5),
                         "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
                         "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1)})
    return pl.DataFrame(rows)


def test_registry_has_eight_named_factors():
    from factorzen.risk.style_factors import STYLE_FACTOR_NAMES, STYLE_FACTOR_REGISTRY
    assert STYLE_FACTOR_NAMES == ["size", "value", "momentum", "volatility",
                                  "liquidity", "quality", "growth", "leverage"]
    assert set(STYLE_FACTOR_REGISTRY.keys()) == set(STYLE_FACTOR_NAMES)


def test_size_factor_shape():
    from factorzen.risk.style_factors import STYLE_FACTOR_REGISTRY
    db = make_daily_basic()
    out = STYLE_FACTOR_REGISTRY["size"](pl.DataFrame(), db)
    assert set(out.columns) >= {"trade_date", "ts_code", "factor_value"}
    assert out.height > 0
    vals = out["factor_value"].drop_nulls().drop_nans()
    assert vals.std() > 0  # 非全零/全同值 stub（size 在不同市值股票间有离散度）


def test_cs_standardize_zero_mean_per_date():
    from factorzen.risk.style_factors import cs_standardize
    # 构造单日截面，标准化后均值≈0
    df = pl.DataFrame({"trade_date": [dt.date(2023, 1, 3)] * 30,
                       "ts_code": [f"{i:06d}.SZ" for i in range(30)],
                       "factor_value": np.random.default_rng(1).standard_normal(30) * 5 + 100})
    std = cs_standardize(df, factor_col="factor_value", method="mad")
    assert abs(std["factor_value"].mean()) < 1e-10  # Z-score 后截面均值数学上严格为 0


def test_cs_standardize_rejects_unknown_method():
    import pytest

    from factorzen.risk.style_factors import cs_standardize

    df = pl.DataFrame({"trade_date": [dt.date(2023, 1, 3)], "factor_value": [1.0]})
    with pytest.raises(ValueError):
        cs_standardize(df, method="zscore")
