"""Tests for Barra-style factors in the personal daily factor library."""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.factors.base import DailyFactor


def _make_daily_lf(n_stocks: int = 10, n_days: int = 310, seed: int = 42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
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
            rows.append({"trade_date": day, "ts_code": s, "close_adj": price})
    return pl.DataFrame(rows).lazy()


def _make_daily_basic_lf(n_stocks: int = 10, n_days: int = 60, seed: int = 0) -> pl.LazyFrame:
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
        for day in days:
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": s,
                    "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1),
                    "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                    "pb": float(abs(rng.standard_normal()) * 1 + 2),
                }
            )
    return pl.DataFrame(rows).lazy()


@dataclass
class MockCtx:
    start: str = "20240101"
    end: str = "20240430"
    required_data: list = field(default_factory=list)
    lookback_days: int = 30
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily_lf: pl.LazyFrame | None = field(default=None, repr=False)
    _daily_basic_lf: pl.LazyFrame | None = field(default=None, repr=False)

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily_lf

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._daily_basic_lf


@pytest.fixture()
def ctx_basic():
    c = MockCtx(start="20240101", end="20240430")
    c._daily_basic_lf = _make_daily_basic_lf()
    return c


@pytest.fixture()
def ctx_daily():
    # start early so the date filter keeps data with shift(252) filled
    c = MockCtx(start="20230601", end="20240430")
    c._daily_lf = _make_daily_lf()
    return c


def _check(result: pl.DataFrame, name: str) -> None:
    assert isinstance(result, pl.DataFrame), f"{name}: expected DataFrame"
    assert "trade_date" in result.columns, f"{name}: missing trade_date"
    assert "ts_code" in result.columns, f"{name}: missing ts_code"
    assert "factor_value" in result.columns, f"{name}: missing factor_value"
    assert result.shape[0] > 0, f"{name}: empty result"


def test_liquidity_style(ctx_basic):
    from workspace.factors.daily.liquidity import LiquidityStyle

    factor = LiquidityStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_basic)
    _check(result, "liquidity_style")


def test_size_style(ctx_basic):
    from workspace.factors.daily.size import SizeStyle

    factor = SizeStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_basic)
    _check(result, "size_style")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(non_null))


def test_value_style(ctx_basic):
    from workspace.factors.daily.value import ValueStyle

    factor = ValueStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_basic)
    _check(result, "value_style")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    # -log(pb) with pb > 0; all finite
    assert np.all(np.isfinite(non_null))


def test_momentum_style(ctx_daily):
    from workspace.factors.daily.momentum_style import MomentumStyle

    factor = MomentumStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_daily)
    assert isinstance(result, pl.DataFrame)
    assert "factor_value" in result.columns


def test_volatility_style(ctx_daily):
    from workspace.factors.daily.volatility_style import VolatilityStyle

    factor = VolatilityStyle()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx_daily)
    assert isinstance(result, pl.DataFrame)
    assert "factor_value" in result.columns
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null >= 0), "Volatility must be non-negative"


def test_beta_style_is_alias_of_beta60d():
    from workspace.factors.daily.beta import Beta60D
    from workspace.factors.daily.beta_style import BetaStyle

    assert BetaStyle is Beta60D
