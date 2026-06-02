"""Unit tests for new daily factors (using synthetic data, no disk I/O)."""

import sys
import types
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.daily.factors.base import DailyFactor

# ── Synthetic data helpers ───────────────────────────────────────────────────


def _make_daily_lf(n_stocks: int = 20, n_days: int = 60, seed: int = 42) -> pl.LazyFrame:
    """Generates a daily LazyFrame with close/amount/vol + *_adj columns."""
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
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": s,
                    "close": price,
                    "open": float(max(price * 0.99, 0.1)),
                    "high": float(max(price * 1.01, 0.1)),
                    "low": float(max(price * 0.98, 0.1)),
                    # adj 列与原始价格相同（测试用，无分红除权）
                    "close_adj": price,
                    "open_adj": float(max(price * 0.99, 0.1)),
                    "high_adj": float(max(price * 1.01, 0.1)),
                    "low_adj": float(max(price * 0.98, 0.1)),
                    "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                    "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4),
                }
            )
    return pl.DataFrame(rows).lazy()


def _make_monthly_basic_lf(n_stocks: int = 20) -> pl.LazyFrame:
    """Generates monthly daily_basic data (pe_ttm/pb/total_mv)."""
    rng = np.random.default_rng(0)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    months = [date(2024, m, 28) for m in range(1, 5)]
    rows = []
    for s in stocks:
        for d in months:
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": s,
                    "pe_ttm": float(abs(rng.standard_normal() * 10 + 20)),
                    "pb": float(abs(rng.standard_normal() * 1 + 2)),
                    "total_mv": float(abs(rng.standard_normal() * 1e9 + 5e9)),
                }
            )
    return pl.DataFrame(rows).lazy()


@dataclass
class MockFactorDataContext:
    start: str = "20240301"
    end: str = "20240430"
    required_data: list = field(default_factory=lambda: ["daily"])
    lookback_days: int = 20
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily_lf: pl.LazyFrame | None = field(default=None, repr=False)
    _monthly_basic_lf: pl.LazyFrame | None = field(default=None, repr=False)

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
    from workspace.factors.daily.amihud import AmihudIlliquidity

    factor = AmihudIlliquidity()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "amihud_illiquidity")
    non_null = result["factor_value"].drop_nulls()
    assert (non_null >= 0).all(), "Amihud illiquidity must be non-negative"


def test_max_return_5d(ctx):
    from workspace.factors.daily.max_return import MaxReturn5D

    factor = MaxReturn5D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "max_return_5d")


def test_skewness_20d(ctx):
    from workspace.factors.daily.skewness import Skewness20D

    factor = Skewness20D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "skewness_20d")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(np.abs(non_null) < 50), "Skewness out of reasonable range"


def test_beta_60d(ctx):
    from workspace.factors.daily.beta import Beta60D

    factor = Beta60D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "beta_60d")


def test_idiosyncratic_vol_20d(ctx):
    from workspace.factors.daily.idiosyncratic_vol import IdiosyncraticVol20D

    factor = IdiosyncraticVol20D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "idiosyncratic_vol_20d")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null >= 0), "Idiosyncratic vol must be non-negative"


def test_bm_ratio(ctx):
    from workspace.factors.monthly.bm_ratio import BmRatioMonthly

    factor = BmRatioMonthly()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "bm_ratio")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null > 0), "B/M ratio must be positive"


def test_ep_ratio(ctx):
    from workspace.factors.monthly.ep_ratio import EpRatioMonthly

    factor = EpRatioMonthly()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "ep_ratio")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null > 0), "E/P ratio must be positive"


def test_registry_has_new_factors():
    from factorzen.daily.factors.registry import list_factors

    factors = list_factors()
    expected = [
        "amihud_illiquidity",
        "max_return_5d",
        "skewness_20d",
        "beta_60d",
        "idiosyncratic_vol_20d",
        "bm_ratio",
        "ep_ratio",
        "asset_growth",
    ]
    for name in expected:
        assert name in factors, f"Factor '{name}' not registered"


def test_registry_has_qlib_factors():
    from factorzen.daily.factors.registry import list_factors

    factors = list_factors()

    assert "qlib_alpha158_kmid" in factors
    assert "qlib_alpha158_ma20" in factors
    assert "qlib_alpha360_close0" in factors
    assert "qlib_alpha360_volume59" in factors


def test_qlib_alpha158_factor_returns_factorzen_schema(ctx, monkeypatch):
    import workspace.factors.qlib.handler as qlib_mod
    from workspace.factors.qlib.handler import QlibAlpha158Kmid

    assert QlibAlpha158Kmid.required_data == ["daily"]

    qlib_df = pl.DataFrame(
        {
            "trade_date": [date(2024, 3, 1), date(2024, 3, 1)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "KMID": [0.1, -0.2],
        }
    )
    monkeypatch.setattr(qlib_mod, "load_qlib_feature_frame", lambda *args, **kwargs: qlib_df)

    result = QlibAlpha158Kmid().compute(ctx)

    assert result.columns == ["trade_date", "ts_code", "factor_value"]
    assert result["factor_value"].to_list() == [0.1, -0.2]


def test_qlib_init_uses_low_memory_defaults(monkeypatch):
    import workspace.factors.qlib.handler as qlib_mod

    init_calls = []

    fake_qlib = types.SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs))
    fake_constant = types.SimpleNamespace(REG_CN="cn")

    monkeypatch.setattr(qlib_mod, "_QLIB_INITIALIZED", False)
    monkeypatch.delenv("QLIB_KERNELS", raising=False)
    monkeypatch.delenv("QLIB_JOBLIB_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "qlib", fake_qlib)
    monkeypatch.setitem(sys.modules, "qlib.constant", fake_constant)

    qlib_mod._init_qlib("provider")

    assert init_calls == [
        {
            "provider_uri": "provider",
            "region": "cn",
            "kernels": 1,
            "joblib_backend": "threading",
        }
    ]


def test_qlib_alpha360_factor_returns_factorzen_schema(ctx, monkeypatch):
    import workspace.factors.qlib.handler as qlib_mod
    from workspace.factors.qlib.handler import QlibAlpha360Close0

    qlib_df = pl.DataFrame(
        {
            "trade_date": [date(2024, 3, 1), date(2024, 3, 1)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "CLOSE0": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(qlib_mod, "load_qlib_feature_frame", lambda *args, **kwargs: qlib_df)

    result = QlibAlpha360Close0().compute(ctx)

    assert result.columns == ["trade_date", "ts_code", "factor_value"]
    assert result["factor_value"].to_list() == [1.0, 1.0]


def _make_finance_lf(n_stocks: int = 20) -> pl.LazyFrame:
    """Synthetic quarterly finance data with assets_yoy.

    Announcement dates are set to be well before the test snapshot dates
    (2024-03-29, 2024-04-30) so PIT alignment finds valid records.
    """
    rng = np.random.default_rng(7)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    # 6 quarters: Q1/2023 through Q2/2024.
    # Announcement dates are set ~1 month after quarter end, but Q1/2024 and
    # Q2/2024 are announced before our snapshot dates (2024-03-29 / 2024-04-30).
    quarter_ann = [
        (date(2023, 3, 31), date(2023, 4, 28)),
        (date(2023, 6, 30), date(2023, 7, 28)),
        (date(2023, 9, 30), date(2023, 10, 28)),
        (date(2023, 12, 31), date(2024, 1, 28)),
        (date(2024, 3, 31), date(2024, 3, 15)),  # announced before 2024-03-29
        (date(2024, 6, 30), date(2024, 4, 15)),  # announced before 2024-04-30
    ]
    rows = []
    for s in stocks:
        for q, ann in quarter_ann:
            rows.append(
                {
                    "ts_code": s,
                    "end_date": q,
                    "ann_date": ann,
                    "assets_yoy": float(rng.standard_normal() * 10),  # YoY growth %
                }
            )
    return pl.DataFrame(rows).lazy()


def test_asset_growth(ctx, monkeypatch):
    import workspace.factors.monthly.asset_growth as ag_mod
    from workspace.factors.monthly.asset_growth import AssetGrowthMonthly

    synthetic_lf = _make_finance_lf()
    monkeypatch.setattr(ag_mod, "scan_parquet", lambda _: synthetic_lf)

    factor = AssetGrowthMonthly()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "asset_growth")
    # YoY growth can be positive or negative, but should be finite
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(non_null)), "Asset growth should be finite"


def test_asset_growth_empty_when_no_finance(ctx, monkeypatch):
    """When finance data unavailable, factor returns empty DataFrame gracefully."""
    import workspace.factors.monthly.asset_growth as ag_mod
    from workspace.factors.monthly.asset_growth import AssetGrowthMonthly

    def _raise(_):
        raise FileNotFoundError("no data")

    monkeypatch.setattr(ag_mod, "scan_parquet", _raise)

    factor = AssetGrowthMonthly()
    result = factor.compute(ctx)
    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()
