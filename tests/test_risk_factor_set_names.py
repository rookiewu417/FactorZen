"""风险模型逐日截面回归须比较因子集的**名字**而非仅列数（R2）。

根因：build 只用 X.shape[1] != len(factor_names) 判因子集一致性，行业成分漂移导致某日
因子集「名字不同但个数相同」时不被跳过，回归系数被静默套上错误因子名，污染因子收益→
协方差→归因，且 n_factor_mismatch 不计数（无任何告警）。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

import factorzen.risk.model as model_mod
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModel


def _make_daily(dates, codes):
    rng = np.random.default_rng(3)
    return pl.DataFrame([
        {"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for d in dates for c in codes
    ])


def test_factor_set_name_drift_is_dropped_not_mislabeled(monkeypatch):
    dates = [dt.date(2024, 1, i) for i in range(2, 8)]  # 6 个交易日
    codes = [f"{i:06d}.SZ" for i in range(6)]
    daily = _make_daily(dates, codes)
    db = pl.DataFrame([{"trade_date": d, "ts_code": c, "total_mv": 5e9, "pb": 1.5, "pe_ttm": 15.0}
                       for d in dates for c in codes])
    stocks = pl.DataFrame({"ts_code": codes, "industry": ["银行"] * 6})

    rng = np.random.default_rng(0)

    def fake_compute_exposures(daily_data, daily_basic, stk, trade_date, *a, **k):
        # 除最后一天外因子集固定为 [size, ind_A, ind_B]；最后一天漂移成 [size, ind_A, ind_C]
        # （B 调出、C 调入）——列数不变(3)、名字变。
        if trade_date == dates[-1]:
            names = ["size", "ind_A", "ind_C"]
        else:
            names = ["size", "ind_A", "ind_B"]
        mat = rng.standard_normal((len(codes), 3))
        return ExposureMatrix(list(codes), names, mat)

    monkeypatch.setattr(model_mod, "compute_exposures", fake_compute_exposures)

    result = RiskModel().build(daily, db, stocks, "20240102", "20240107")

    assert result.n_dropped_dates == 1, (
        f"名字漂移的那天应被跳过并计入 n_dropped_dates，实得 {result.n_dropped_dates}"
        "（修复前只比列数 → 该天被当成一致、ind_C 系数被错标为 ind_B）"
    )
    # 因子名应是参考集(首个有效截面)，不含漂移进来的 ind_C
    assert result.factor_names == ["size", "ind_A", "ind_B"]
    assert "ind_C" not in result.factor_returns.columns
