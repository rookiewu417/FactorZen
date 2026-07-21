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


def test_daily_factor_compute_suite(ctx):
    """test_amihud_illiquidity；除权造成的未复权 close 断崖不应污染 Amihud 非流动性（须基于 close_adj 计算）。；MomentumWeekly 之前未被任何测试引用，补基本覆盖。；除权造成的未复权 close 断崖不应污染周频动量（须基于 close_adj 计算）。；VolatilityWeekly 之前未被任何测试引用，补基本覆盖。；除权造成的未复权 close 断崖不应污染周频波动率（须基于 close_adj 计算）。；test_max_return_5d；test_skewness_20d"""
    # -- 原 test_amihud_illiquidity --
    def _section_0_test_amihud_illiquidity(ctx):
        from factorzen.builtin_factors.daily.amihud import AmihudIlliquidity

        factor = AmihudIlliquidity()
        assert isinstance(factor, DailyFactor)
        result = factor.compute(ctx)
        _check_result(result, "amihud_illiquidity")
        non_null = result["factor_value"].drop_nulls()
        assert (non_null >= 0).all(), "Amihud illiquidity must be non-negative"

    _section_0_test_amihud_illiquidity(ctx)

    # -- 原 test_amihud_illiquidity_unaffected_by_unadjusted_close_jump --
    def _section_1_test_amihud_illiquidity_unaffected_by_unadjusted_close_jump():
        from factorzen.builtin_factors.daily.amihud import AmihudIlliquidity

        days = _trading_days(date(2024, 1, 2), 45)
        split_index = 15
        start, end = days[0].strftime("%Y%m%d"), days[-1].strftime("%Y%m%d")

        ctx_jump = _DividendJumpContext(
            start=start, end=end, _daily_lf=_make_dividend_jump_daily_lf(days, split_index=split_index)
        )
        ctx_clean = _DividendJumpContext(
            start=start, end=end, _daily_lf=_make_dividend_jump_daily_lf(days, split_index=None)
        )

        # sanity check：确认合成数据本身在除权日确实制造了 close 断崖（否则测试无判别力）
        split_row = ctx_jump.daily.collect().filter(
            (pl.col("trade_date") == days[split_index]) & (pl.col("ts_code") == "000000.SZ")
        )
        assert abs(split_row["close"][0] - split_row["close_adj"][0]) > 1.0

        factor = AmihudIlliquidity()
        result_jump = factor.compute(ctx_jump).sort(["ts_code", "trade_date"])
        result_clean = factor.compute(ctx_clean).sort(["ts_code", "trade_date"])

        np.testing.assert_allclose(
            result_jump["factor_value"].to_numpy(),
            result_clean["factor_value"].to_numpy(),
            rtol=1e-7,
            atol=1e-12,
            equal_nan=True,
        )

    _section_1_test_amihud_illiquidity_unaffected_by_unadjusted_close_jump()

    # -- 原 test_momentum_weekly_basic --
    def _section_2_test_momentum_weekly_basic():
        from factorzen.builtin_factors.weekly.momentum import MomentumWeekly

        days = _trading_days(date(2024, 1, 2), 30)
        snapshot_date = days[25]
        ctx = _DividendJumpContext(
            start=days[0].strftime("%Y%m%d"),
            end=days[-1].strftime("%Y%m%d"),
            _daily_lf=_make_dividend_jump_daily_lf(days, split_index=None),
            _snapshot_dates=[snapshot_date],
        )
        factor = MomentumWeekly()
        assert isinstance(factor, DailyFactor)
        result = factor.compute(ctx)
        _check_result(result, "momentum_weekly")
        assert result["trade_date"].unique().to_list() == [snapshot_date]

    _section_2_test_momentum_weekly_basic()

    # -- 原 test_momentum_weekly_unaffected_by_unadjusted_close_jump --
    def _section_3_test_momentum_weekly_unaffected_by_unadjusted_close_jump():
        from factorzen.builtin_factors.weekly.momentum import MomentumWeekly

        days = _trading_days(date(2024, 1, 2), 45)
        split_index = 15
        snapshot_date = days[30]  # 距 split_index 仅 15 个交易日，落在 20 日回看窗口内
        start, end = days[0].strftime("%Y%m%d"), days[-1].strftime("%Y%m%d")

        ctx_jump = _DividendJumpContext(
            start=start,
            end=end,
            _daily_lf=_make_dividend_jump_daily_lf(days, split_index=split_index),
            _snapshot_dates=[snapshot_date],
        )
        ctx_clean = _DividendJumpContext(
            start=start,
            end=end,
            _daily_lf=_make_dividend_jump_daily_lf(days, split_index=None),
            _snapshot_dates=[snapshot_date],
        )

        factor = MomentumWeekly()
        result_jump = factor.compute(ctx_jump).sort(["ts_code"])
        result_clean = factor.compute(ctx_clean).sort(["ts_code"])

        assert result_jump.shape[0] > 0
        np.testing.assert_allclose(
            result_jump["factor_value"].to_numpy(),
            result_clean["factor_value"].to_numpy(),
            rtol=1e-7,
            atol=1e-12,
            equal_nan=True,
        )

    _section_3_test_momentum_weekly_unaffected_by_unadjusted_close_jump()

    # -- 原 test_volatility_weekly_basic --
    def _section_4_test_volatility_weekly_basic():
        from factorzen.builtin_factors.weekly.volatility import VolatilityWeekly

        days = _trading_days(date(2024, 1, 2), 30)
        snapshot_date = days[25]
        ctx = _DividendJumpContext(
            start=days[0].strftime("%Y%m%d"),
            end=days[-1].strftime("%Y%m%d"),
            _daily_lf=_make_dividend_jump_daily_lf(days, split_index=None),
            _snapshot_dates=[snapshot_date],
        )
        factor = VolatilityWeekly()
        assert isinstance(factor, DailyFactor)
        result = factor.compute(ctx)
        _check_result(result, "volatility_weekly")
        non_null = result["factor_value"].drop_nulls().to_numpy()
        assert np.all(non_null >= 0), "Volatility must be non-negative"

    _section_4_test_volatility_weekly_basic()

    # -- 原 test_volatility_weekly_unaffected_by_unadjusted_close_jump --
    def _section_5_test_volatility_weekly_unaffected_by_unadjusted_close_jump():
        from factorzen.builtin_factors.weekly.volatility import VolatilityWeekly

        days = _trading_days(date(2024, 1, 2), 45)
        split_index = 15
        snapshot_date = days[30]
        start, end = days[0].strftime("%Y%m%d"), days[-1].strftime("%Y%m%d")

        ctx_jump = _DividendJumpContext(
            start=start,
            end=end,
            _daily_lf=_make_dividend_jump_daily_lf(days, split_index=split_index),
            _snapshot_dates=[snapshot_date],
        )
        ctx_clean = _DividendJumpContext(
            start=start,
            end=end,
            _daily_lf=_make_dividend_jump_daily_lf(days, split_index=None),
            _snapshot_dates=[snapshot_date],
        )

        factor = VolatilityWeekly()
        result_jump = factor.compute(ctx_jump).sort(["ts_code"])
        result_clean = factor.compute(ctx_clean).sort(["ts_code"])

        assert result_jump.shape[0] > 0
        np.testing.assert_allclose(
            result_jump["factor_value"].to_numpy(),
            result_clean["factor_value"].to_numpy(),
            rtol=1e-7,
            atol=1e-12,
            equal_nan=True,
        )

    _section_5_test_volatility_weekly_unaffected_by_unadjusted_close_jump()

    # -- 原 test_max_return_5d --
    def _section_6_test_max_return_5d(ctx):
        from factorzen.builtin_factors.daily.max_return import MaxReturn5D

        factor = MaxReturn5D()
        assert isinstance(factor, DailyFactor)
        result = factor.compute(ctx)
        _check_result(result, "max_return_5d")

    _section_6_test_max_return_5d(ctx)

    # -- 原 test_skewness_20d --
    def _section_7_test_skewness_20d(ctx):
        from factorzen.builtin_factors.daily.skewness import Skewness20D

        factor = Skewness20D()
        assert isinstance(factor, DailyFactor)
        result = factor.compute(ctx)
        _check_result(result, "skewness_20d")
        non_null = result["factor_value"].drop_nulls().to_numpy()
        assert np.all(np.abs(non_null) < 50), "Skewness out of reasonable range"

    _section_7_test_skewness_20d(ctx)


# ── close_adj regression tests (复权 close_adj vs 未复权 close) ─────────────
#
# Amihud / MomentumWeekly / VolatilityWeekly 的收益率计算必须基于 close_adj
# （复权收盘价），否则分红/拆股除权日会在未复权 close 上制造虚假价格断崖，污染
# 滚动窗口。下面构造一份 close_adj 平滑、close 在除权日断崖下跌的合成数据，
# 验证三个因子的输出只取决于 close_adj，不受 close 断崖影响。


def _trading_days(start: date, n: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _make_dividend_jump_daily_lf(
    days: list[date],
    split_index: int | None = None,
    n_stocks: int = 4,
    jump_ratio: float = 0.5,
    seed: int = 99,
) -> pl.LazyFrame:
    """构造合成日线数据：close_adj 是平滑随机游走（"真实"复权价格序列）。

    若指定 split_index，未复权 close 从该交易日起按 jump_ratio 折算，模拟分红/
    拆股除权造成的未复权价格断崖（close_adj 保持连续、不受影响）。
    split_index=None 时 close == close_adj，即无除权事件的对照基线。
    """
    rng = np.random.default_rng(seed)
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for s in stocks:
        adj_price = 10.0
        for i, day in enumerate(days):
            adj_price = float(max(adj_price * (1 + rng.standard_normal() * 0.015), 0.5))
            if split_index is not None and i >= split_index:
                raw_close = adj_price * jump_ratio
            else:
                raw_close = adj_price
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": s,
                    "close": raw_close,
                    "close_adj": adj_price,
                    "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                }
            )
    return pl.DataFrame(rows).lazy()


@dataclass
class _DividendJumpContext:
    """轻量 mock context：snapshot_dates 可由调用方指定，便于精确控制除权日是否
    落入因子的回看窗口。"""

    start: str
    end: str
    _daily_lf: pl.LazyFrame
    _snapshot_dates: list = field(default_factory=list)
    required_data: list = field(default_factory=lambda: ["daily"])
    lookback_days: int = 20
    universe: list | None = None
    snapshot_mode: str = "weekly"

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily_lf

    @property
    def snapshot_dates(self) -> list:
        return self._snapshot_dates


def test_beta_60d(ctx):
    from factorzen.builtin_factors.daily.beta import Beta60D

    factor = Beta60D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "beta_60d")


def test_idiosyncratic_vol_20d(ctx):
    from factorzen.builtin_factors.daily.idiosyncratic_vol import IdiosyncraticVol20D

    factor = IdiosyncraticVol20D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "idiosyncratic_vol_20d")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null >= 0), "Idiosyncratic vol must be non-negative"


def test_volume_return_corr_20d(ctx):
    from factorzen.config.settings import FACTOR_STORE_DIR
    from factorzen.discovery.factor_store import load_python_factor_module

    mod = load_python_factor_module(
        FACTOR_STORE_DIR / "ashare" / "volume_return_corr_20d" / "factor.py"
    )
    factor = mod.VolumeReturnCorr20D()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "volume_return_corr_20d")
    assert result["trade_date"].min() >= date(2024, 3, 1)
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(non_null)), "Volume-return correlation should be finite"
    assert np.all(np.abs(non_null) <= 1.0 + 1e-12), "Correlation must be in [-1, 1]"


def test_bm_ratio(ctx):
    from factorzen.builtin_factors.monthly.bm_ratio import BmRatioMonthly

    factor = BmRatioMonthly()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "bm_ratio")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null > 0), "B/M ratio must be positive"


def test_ep_ratio(ctx):
    from factorzen.builtin_factors.monthly.ep_ratio import EpRatioMonthly

    factor = EpRatioMonthly()
    assert isinstance(factor, DailyFactor)
    result = factor.compute(ctx)
    _check_result(result, "ep_ratio")
    non_null = result["factor_value"].drop_nulls().to_numpy()
    assert np.all(non_null > 0), "E/P ratio must be positive"


def test_factor_required_meta_suite():
    """test_registry_has_new_factors；test_registry_has_qlib_factors；store python 单路径"""
    # -- 原 test_registry_has_new_factors --
    def _section_0_test_registry_has_new_factors():
        from factorzen.daily.factors.registry import list_factors

        factors = list_factors()
        # builtin 包扫描路径（不含 factor_store 用户因子）
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

    _section_0_test_registry_has_new_factors()

    # -- factor_store 单路径：load_library_factors 注入 python 用户因子 --
    def _section_0b_store_python_factor_registered():
        from factorzen.daily.factors.registry import list_factors
        from factorzen.discovery.library_provider import load_library_factors

        load_library_factors(market="ashare")
        factors = list_factors()
        assert "volume_return_corr_20d" in factors, (
            "store python factor volume_return_corr_20d not registered via load_library_factors"
        )

    _section_0b_store_python_factor_registered()

    # -- 原 test_registry_has_qlib_factors --
    def _section_1_test_registry_has_qlib_factors():
        from factorzen.daily.factors.registry import list_factors

        factors = list_factors()

        assert "qlib_alpha158_kmid" in factors
        assert "qlib_alpha158_ma20" in factors
        assert "qlib_alpha360_close0" in factors
        assert "qlib_alpha360_volume59" in factors

    _section_1_test_registry_has_qlib_factors()


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


def test_special_event_factor_suite(ctx, monkeypatch):
    """test_asset_growth；When finance data unavailable, factor returns empty DataFrame gracefully.；test_qlib_alpha158_factor_returns_factorzen_schema；test_qlib_init_uses_low_memory_defaults；test_qlib_alpha360_factor_returns_factorzen_schema"""
    # -- 原 test_asset_growth --
    def _section_0_test_asset_growth(ctx, mp):
        import factorzen.builtin_factors.monthly.asset_growth as ag_mod
        from factorzen.builtin_factors.monthly.asset_growth import AssetGrowthMonthly

        synthetic_lf = _make_finance_lf()
        mp.setattr(ag_mod, "scan_parquet", lambda _: synthetic_lf)

        factor = AssetGrowthMonthly()
        assert isinstance(factor, DailyFactor)
        result = factor.compute(ctx)
        _check_result(result, "asset_growth")
        # YoY growth can be positive or negative, but should be finite
        non_null = result["factor_value"].drop_nulls().to_numpy()
        assert np.all(np.isfinite(non_null)), "Asset growth should be finite"

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_asset_growth(ctx, mp)

    # -- 原 test_asset_growth_empty_when_no_finance --
    def _section_1_test_asset_growth_empty_when_no_finance(ctx, mp):
        import factorzen.builtin_factors.monthly.asset_growth as ag_mod
        from factorzen.builtin_factors.monthly.asset_growth import AssetGrowthMonthly

        def _raise(_):
            raise FileNotFoundError("no data")

        mp.setattr(ag_mod, "scan_parquet", _raise)

        factor = AssetGrowthMonthly()
        result = factor.compute(ctx)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_asset_growth_empty_when_no_finance(ctx, mp)

    # -- 原 test_qlib_alpha158_factor_returns_factorzen_schema --
    def _section_2_test_qlib_alpha158_factor_returns_factorzen_schema(ctx, mp):
        import factorzen.builtin_factors.qlib.handler as qlib_mod
        from factorzen.builtin_factors.qlib.handler import QlibAlpha158Kmid

        assert QlibAlpha158Kmid.required_data == ["daily"]

        qlib_df = pl.DataFrame(
            {
                "trade_date": [date(2024, 3, 1), date(2024, 3, 1)],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "KMID": [0.1, -0.2],
            }
        )
        mp.setattr(qlib_mod, "load_qlib_feature_frame", lambda *args, **kwargs: qlib_df)

        result = QlibAlpha158Kmid().compute(ctx)

        assert result.columns == ["trade_date", "ts_code", "factor_value"]
        assert result["factor_value"].to_list() == [0.1, -0.2]

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_qlib_alpha158_factor_returns_factorzen_schema(ctx, mp)

    # -- 原 test_qlib_init_uses_low_memory_defaults --
    def _section_3_test_qlib_init_uses_low_memory_defaults(mp):
        import factorzen.builtin_factors.qlib.handler as qlib_mod

        init_calls = []

        fake_qlib = types.SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs))
        fake_constant = types.SimpleNamespace(REG_CN="cn")

        mp.setattr(qlib_mod, "_QLIB_INITIALIZED", False)
        mp.delenv("QLIB_KERNELS", raising=False)
        mp.delenv("QLIB_JOBLIB_BACKEND", raising=False)
        mp.setitem(sys.modules, "qlib", fake_qlib)
        mp.setitem(sys.modules, "qlib.constant", fake_constant)

        qlib_mod._init_qlib("provider")

        assert init_calls == [
            {
                "provider_uri": "provider",
                "region": "cn",
                "kernels": 1,
                "joblib_backend": "threading",
            }
        ]

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_qlib_init_uses_low_memory_defaults(mp)

    # -- 原 test_qlib_alpha360_factor_returns_factorzen_schema --
    def _section_4_test_qlib_alpha360_factor_returns_factorzen_schema(ctx, mp):
        import factorzen.builtin_factors.qlib.handler as qlib_mod
        from factorzen.builtin_factors.qlib.handler import QlibAlpha360Close0

        qlib_df = pl.DataFrame(
            {
                "trade_date": [date(2024, 3, 1), date(2024, 3, 1)],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "CLOSE0": [1.0, 1.0],
            }
        )
        mp.setattr(qlib_mod, "load_qlib_feature_frame", lambda *args, **kwargs: qlib_df)

        result = QlibAlpha360Close0().compute(ctx)

        assert result.columns == ["trade_date", "ts_code", "factor_value"]
        assert result["factor_value"].to_list() == [1.0, 1.0]

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_qlib_alpha360_factor_returns_factorzen_schema(ctx, mp)


