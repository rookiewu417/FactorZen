"""中性化 IC 测试。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def test_pure_industry_factor_ic_near_zero():
    """纯行业因子中性化后 IC 应接近 0。"""
    from daily.evaluation.advanced import compute_neutralized_ic

    rng = np.random.default_rng(42)
    n_stocks = 60
    n_dates = 20
    start = date(2023, 1, 3)
    industries = ["ind_A"] * 20 + ["ind_B"] * 20 + ["ind_C"] * 20
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        # factor = industry mean + tiny noise: pure industry effect
        ind_means = {"ind_A": 1.0, "ind_B": -1.0, "ind_C": 0.0}
        for s in range(n_stocks):
            ind = industries[s]
            factor_val = ind_means[ind] + rng.standard_normal() * 0.01
            ret_val = rng.standard_normal() * 0.02  # no real factor signal
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(factor_val),
                    "ret_1d": float(ret_val),
                    "industry": ind,
                    "log_mktcap": float(rng.uniform(10, 20)),
                }
            )
    df = pl.DataFrame(rows)
    result = compute_neutralized_ic(df)
    assert (
        abs(result.ic_mean) < 0.1
    ), f"IC should be near 0 after neutralization, got {result.ic_mean}"


def test_neutralized_ic_returns_ic_stats():
    """compute_neutralized_ic 应返回 IcStats 对象。"""
    from daily.evaluation.advanced import compute_neutralized_ic
    from daily.evaluation.ic_analysis import IcStats

    rng = np.random.default_rng(1)
    n_stocks = 30
    n_dates = 10
    start = date(2023, 1, 3)
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(rng.standard_normal()),
                    "ret_1d": float(rng.standard_normal() * 0.01),
                    "industry": f"ind_{s % 3}",
                    "log_mktcap": float(rng.uniform(10, 20)),
                }
            )
    df = pl.DataFrame(rows)
    result = compute_neutralized_ic(df)
    assert isinstance(result, IcStats)


def test_neutralized_ic_industry_only():
    """仅行业中性化（neutralize_by='industry'）应正常运行。"""
    from daily.evaluation.advanced import compute_neutralized_ic
    from daily.evaluation.ic_analysis import IcStats

    rng = np.random.default_rng(2)
    n_stocks = 30
    n_dates = 10
    start = date(2023, 1, 3)
    rows = []
    for d in range(n_dates):
        dt = (start + timedelta(days=d)).isoformat()
        for s in range(n_stocks):
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"00{s:04d}.SZ",
                    "factor_clean": float(rng.standard_normal()),
                    "ret_1d": float(rng.standard_normal() * 0.01),
                    "industry": f"ind_{s % 3}",
                }
            )
    df = pl.DataFrame(rows)
    result = compute_neutralized_ic(df, neutralize_by="industry")
    assert isinstance(result, IcStats)
