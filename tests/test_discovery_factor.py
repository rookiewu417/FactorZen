# tests/test_discovery_factor.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_daily_lf(n_stocks=8, n_days=60, seed=42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": price,
                         "open": price, "high": price, "low": price,
                         "close_adj": price, "open_adj": price, "high_adj": price, "low_adj": price,
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).lazy()


@dataclass
class MockCtx:
    start: str = "20240301"
    end: str = "20240331"
    required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
    lookback_days: int = 30
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily: pl.LazyFrame | None = None
    _basic: pl.LazyFrame | None = None

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._basic if self._basic is not None else pl.DataFrame(
            {"trade_date": [], "ts_code": []}).lazy()


def test_expression_factor_matches_builtin_momentum():
    """pct_change(close, 20) 应与内置 momentum_20d 的 compute 输出一致。"""
    from factorzen.discovery.factor import ExpressionFactor
    from factorzen.builtin_factors.daily.momentum import Momentum20D

    lf = _make_daily_lf()
    ctx = MockCtx(_daily=lf)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builtin = Momentum20D().compute(ctx).sort(["trade_date", "ts_code"])

    mined = ExpressionFactor(expression="pct_change(close, 20)", mined_name="m20",
                             lookback_days=30).compute(ctx).sort(["trade_date", "ts_code"])

    j = builtin.join(mined, on=["trade_date", "ts_code"], suffix="_m")
    diff = (j["factor_value"] - j["factor_value_m"]).abs().max()
    assert diff is None or diff < 1e-9


def test_suspended_rows_masked():
    """vol==0（停牌）行不应产出有限因子值。"""
    from factorzen.discovery.factor import ExpressionFactor
    lf = _make_daily_lf()
    # 注入一只全停牌股票
    extra = pl.DataFrame({"trade_date": [date(2024, 3, 15)], "ts_code": ["999999.SH"],
                          "close": [5.0], "open": [5.0], "high": [5.0], "low": [5.0],
                          "close_adj": [5.0], "open_adj": [5.0], "high_adj": [5.0],
                          "low_adj": [5.0], "amount": [0.0], "vol": [0.0]}).lazy()
    ctx = MockCtx(_daily=pl.concat([lf, extra]))
    out = ExpressionFactor(expression="ts_mean(close, 5)", mined_name="x",
                           lookback_days=30).compute(ctx)
    sus = out.filter(pl.col("ts_code") == "999999.SH")
    assert sus.height == 0 or sus["factor_value"].is_null().all()
