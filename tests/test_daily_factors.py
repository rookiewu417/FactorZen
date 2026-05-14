"""Unit tests for new daily factors (using synthetic data, no disk I/O)."""

from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl
import pytest

from daily.factors.base import LFTFactor


# ── Synthetic data helpers ───────────────────────────────────────────────────

def _make_daily_lf(n_stocks: int = 20, n_days: int = 60, seed: int = 42) -> pl.LazyFrame:
    """Generates a daily LazyFrame with close/amount/vol columns."""
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)

    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for s in stocks:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({
                "trade_date": day,
                "ts_code": s,
                "close": price,
                "open": float(max(price * 0.99, 0.1)),
                "high": float(max(price * 1.01, 0.1)),
                "low": float(max(price * 0.98, 0.1)),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4),
            })
    return pl.DataFrame(rows).lazy()


def _make_monthly_basic_lf(n_stocks: int = 20) -> pl.LazyFrame:
    """Generates monthly daily_basic data (pe_ttm/pb/total_mv)."""
    rng = np.random.default_rng(0)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    months = [date(2024, m, 28) for m in range(1, 5)]
    rows = []
    for s in stocks:
        for d in months:
            rows.append({
                "trade_date": d,
                "ts_code": s,
                "pe_ttm": float(abs(rng.standard_normal() * 10 + 20)),
                "pb": float(abs(rng.standard_normal() * 1 + 2)),
                "total_mv": float(abs(rng.standard_normal() * 1e9 + 5e9)),
            })
    return pl.DataFrame(rows).lazy()


@dataclass
class MockFactorDataContext:
    start: str = "20240301"
    end: str = "20240430"
    required_data: list = field(default_factory=lambda: ["daily"])
    lookback_days: int = 20
    universe: Optional[list] = None
    snapshot_mode: str = "daily"
    _daily_lf: Optional[pl.LazyFrame] = field(default=None, repr=False)
    _monthly_basic_lf: Optional[pl.LazyFrame] = field(default=None, repr=False)

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily_lf

    @property
    def monthly_basic(self) -> pl.LazyFrame:
        return self._monthly_basic_lf

    @property
    def snapshot_dates(self):
        return [date(2024, 3, 29), date(2024, 4, 30)]


@pytest.fixture()
def ctx():
    c = MockFactorDataContext()
    c._daily_lf = _make_daily_lf()
    c._monthly_basic_lf = _make_monthly_basic_lf()
    return c


# ── Generic result checker ───────────────────────────────────────────────────

def _check_result(result: pl.DataFrame, factor_name: str):
    assert isinstance(result, pl.DataFrame), f"{factor_name}: result must be a DataFrame"
    assert "trade_date" in result.columns, f"{factor_name}: missing trade_date column"
    assert "ts_code" in result.columns, f"{factor_name}: missing ts_code column"
    assert "factor_value" in result.columns, f"{factor_name}: missing factor_value column"
    assert result.shape[0] > 0, f"{factor_name}: result is empty"


# ── Individual factor tests ──────────────────────────────────────────────────

def test_amihud_illiquidity(ctx):
    from daily.factors.daily.amihud import AmihudIlliquidity
    factor = AmihudIlliquidity()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "amihud_illiquidity")
    non_null = result["factor_value"].drop_nulls()
    assert (non_null >= 0).all(), "Amihud illiquidity must be non-negative"


def test_max_return_5d(ctx):
    from daily.factors.daily.max_return import MaxReturn5D
    factor = MaxReturn5D()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "max_return_5d")


def test_skewness_20d(ctx):
    from daily.factors.daily.skewness import Skewness20D
    factor = Skewness20D()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "skewness_20d")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(np.abs(non_null) < 50), "Skewness out of reasonable range"


def test_beta_60d(ctx):
    from daily.factors.daily.beta import Beta60D
    factor = Beta60D()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "beta_60d")


def test_idiosyncratic_vol_20d(ctx):
    from daily.factors.daily.idiosyncratic_vol import IdiosyncraticVol20D
    factor = IdiosyncraticVol20D()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "idiosyncratic_vol_20d")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null >= 0), "Idiosyncratic vol must be non-negative"


def test_bm_ratio(ctx):
    from daily.factors.monthly.bm_ratio import BmRatioMonthly
    factor = BmRatioMonthly()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "bm_ratio")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null > 0), "B/M ratio must be positive"


def test_ep_ratio(ctx):
    from daily.factors.monthly.ep_ratio import EpRatioMonthly
    factor = EpRatioMonthly()
    assert isinstance(factor, LFTFactor)
    result = factor.compute(ctx)
    _check_result(result, "ep_ratio")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null > 0), "E/P ratio must be positive"


def test_registry_has_new_factors():
    from daily.factors.registry import list_factors
    factors = list_factors()
    expected = [
        "amihud_illiquidity", "max_return_5d", "skewness_20d",
        "beta_60d", "idiosyncratic_vol_20d", "bm_ratio", "ep_ratio",
    ]
    for name in expected:
        assert name in factors, f"Factor '{name}' not registered"
