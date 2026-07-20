"""test_membership_vectorized_equiv.py：universe membership 向量化等价性：与逐日 _load_index_members 完全一致。
test_walk_forward_summary.py：Tests for single-factor walk-forward summary integration.
test_output_paths.py：无 module docstring 的测试。
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from factorzen.config.settings import (
    OUTPUT_DAILY_FACTORS,
    OUTPUT_DAILY_REPORTS,
    OUTPUT_DAILY_RESULTS,
    daily_factor_output_dir,
    daily_report_output_dir,
    daily_result_output_dir,
)


# ==== 来自 test_membership_vectorized_equiv.py ====
def _membership_index_reference(
    start: str,
    end: str,
    universe_name: str,
) -> pl.DataFrame:
    """旧实现：逐交易日 _load_index_members（Wave1 前语义）。"""
    from factorzen.core.calendar import get_trade_dates
    from factorzen.core.universe import _INDEX_CODE_MAP, _load_index_members

    trade_dates = get_trade_dates(start, end)
    if not trade_dates:
        return pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8})

    index_names = (
        ("csi300", "csi500") if universe_name == "csi800" else (universe_name,)
    )
    rows: list[dict[str, str]] = []
    for d in trade_dates:
        day_str = d.strftime("%Y%m%d")
        members: set[str] = set()
        for uname in index_names:
            code = _INDEX_CODE_MAP[uname]
            members.update(_load_index_members(code, day_str))
        for code in members:
            rows.append({"trade_date": day_str, "ts_code": code})

    if not rows:
        return pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8})
    return pl.DataFrame(rows).select(["trade_date", "ts_code"]).unique()

def _assert_membership_equal(got: pl.DataFrame, expected: pl.DataFrame) -> None:
    g = (
        got.select(["trade_date", "ts_code"])
        .unique()
        .sort(["trade_date", "ts_code"])
    )
    e = (
        expected.select(["trade_date", "ts_code"])
        .unique()
        .sort(["trade_date", "ts_code"])
    )
    assert g.equals(e), (
        f"row mismatch: got={g.height} expected={e.height}\n"
        f"got-only sample: {g.join(e, on=['trade_date','ts_code'], how='anti').head(10)}\n"
        f"exp-only sample: {e.join(g, on=['trade_date','ts_code'], how='anti').head(10)}"
    )

def _has_csi500_cache() -> bool:
    from factorzen.config.settings import DATA_CACHE

    return any(DATA_CACHE.glob("index_member_000905_SH_*.parquet"))

@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_matches_daily_loop_cross_month():
    """跨月窗口：逐日成分集合与旧实现完全一致。"""
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    start, end = "20230101", "20230630"
    expected = _membership_index_reference(start, end, "csi500")
    _INDEX_MEMBER_MEMORY_CACHE.clear()
    got = get_universe_membership(start, end, "csi500")
    _assert_membership_equal(got, expected)

@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_month_boundary_and_rebalance():
    """含月初/月末边界与成分调整月（6 月调样窗口）。"""
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    # 5–7 月覆盖半年调样
    start, end = "20230501", "20230731"
    expected = _membership_index_reference(start, end, "csi500")
    _INDEX_MEMBER_MEMORY_CACHE.clear()
    got = get_universe_membership(start, end, "csi500")
    _assert_membership_equal(got, expected)

    # 抽检：月初与月末集合应各自非空（有缓存时）
    days = got["trade_date"].unique().sort().to_list()
    assert days
    for d in (days[0], days[len(days) // 2], days[-1]):
        assert got.filter(pl.col("trade_date") == d).height > 0

@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_two_year_perf():
    """2 年 csi500 membership ≤ 0.5s。"""
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    # warmup disk into OS cache
    _ = get_universe_membership("20230101", "20230131", "csi500")
    _INDEX_MEMBER_MEMORY_CACHE.clear()

    t0 = time.perf_counter()
    mem = get_universe_membership("20230101", "20241231", "csi500")
    elapsed = time.perf_counter() - t0
    assert mem.height > 100_000
    assert elapsed <= 0.5, f"membership {elapsed:.3f}s > 0.5s"

@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_csi800_union_equiv():
    from factorzen.core.universe import (
        _INDEX_MEMBER_MEMORY_CACHE,
        get_universe_membership,
    )

    _INDEX_MEMBER_MEMORY_CACHE.clear()
    start, end = "20240101", "20240331"
    expected = _membership_index_reference(start, end, "csi800")
    _INDEX_MEMBER_MEMORY_CACHE.clear()
    got = get_universe_membership(start, end, "csi800")
    _assert_membership_equal(got, expected)

def test_batch_membership_fetches_missing_window_months(monkeypatch, tmp_path):
    """窗口跨月、后月缓存缺失时，向量化路径须触发拉取，禁止用更早快照静默顶替。

    回归：_batch_index_membership 只要 ≤end 任意月缓存非空就直接 as-of，
    缺月不拉、不告警 → 成分调整丢失。
    """
    from types import SimpleNamespace

    import pandas as pd

    from factorzen.core import universe as U

    monkeypatch.setattr(U, "DATA_CACHE", tmp_path)
    U._INDEX_MEMBER_MEMORY_CACHE.clear()
    U._INDEX_WEIGHT_DF_CACHE.clear()

    index_code = "000300.SH"
    # 仅 1 月本地缓存（旧成分）；2 月文件缺失，拉取应返回全新调样
    pl.DataFrame(
        {
            "con_code": ["A.SZ", "B.SZ"],
            "trade_date": ["20240115", "20240115"],
        }
    ).write_parquet(tmp_path / "index_member_000300_SH_202401.parquet")

    fetch_months: list[str] = []

    def _fake_retry(_fn, **kw):
        start = str(kw.get("start_date", ""))
        ym = start[:6]
        fetch_months.append(ym)
        if ym == "202402":
            return pd.DataFrame(
                {
                    "con_code": ["C.SZ", "D.SZ"],
                    "trade_date": ["20240215", "20240215"],
                }
            )
        return pd.DataFrame()

    monkeypatch.setattr(
        "factorzen.core.loader.init_tushare",
        lambda: SimpleNamespace(index_weight=lambda **_kw: None),
    )
    monkeypatch.setattr("factorzen.core.loader._retry", _fake_retry)

    day_strs = ["20240116", "20240117", "20240216", "20240217"]

    def _setup_partial_cache() -> None:
        feb = tmp_path / "index_member_000300_SH_202402.parquet"
        if feb.exists():
            feb.unlink()
        U._INDEX_MEMBER_MEMORY_CACHE.clear()
        U._INDEX_WEIGHT_DF_CACHE.clear()
        fetch_months.clear()

    # ── 向量化路径：须补拉 2 月，2 月交易日成分 = 新调样 ──────────────
    _setup_partial_cache()
    got = U._batch_index_membership(index_code, day_strs)

    assert "202402" in fetch_months, (
        f"缺月须触发 index_weight 拉取，实际 fetch_months={fetch_months}"
    )
    assert (tmp_path / "index_member_000300_SH_202402.parquet").exists(), (
        "缺月拉取成功后应写入月缓存"
    )

    jan_codes = set(
        got.filter(pl.col("trade_date").is_in(["20240116", "20240117"]))[
            "ts_code"
        ].to_list()
    )
    feb_codes = set(
        got.filter(pl.col("trade_date").is_in(["20240216", "20240217"]))[
            "ts_code"
        ].to_list()
    )
    assert jan_codes == {"A.SZ", "B.SZ"}
    assert feb_codes == {"C.SZ", "D.SZ"}, (
        f"缺失月不得用旧快照顶替：期望 {{C.SZ, D.SZ}}，实际 {feb_codes}"
    )

    # ── 与逐日 _load_index_members 完全一致（独立从部分缓存起步）──
    _setup_partial_cache()
    ref_rows: list[dict[str, str]] = []
    for d in day_strs:
        for code in U._load_index_members(index_code, d):
            ref_rows.append({"trade_date": d, "ts_code": code})
    expected = pl.DataFrame(ref_rows).select(["trade_date", "ts_code"]).unique()

    _setup_partial_cache()
    got2 = U._batch_index_membership(index_code, day_strs)
    _assert_membership_equal(got2, expected)

# ==== 来自 test_walk_forward_summary.py ====
def _make_factor_price(n_dates: int = 40, n_stocks: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]
    factor_rows = []
    price_rows = []
    last_close = {code: 10.0 + idx for idx, code in enumerate(stocks)}
    for i, d in enumerate(dates):
        for code in stocks:
            if i < n_dates - 1:
                factor_rows.append(
                    {
                        "trade_date": d,
                        "ts_code": code,
                        "factor_clean": float(rng.normal()),
                    }
                )
            open_price = last_close[code] * (1.0 + float(rng.normal(0, 0.001)))
            close_price = open_price * (1.0 + float(rng.normal(0, 0.01)))
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "open": open_price,
                    "close": close_price,
                    "pre_close": last_close[code],
                    "pct_chg": (close_price / last_close[code] - 1.0) * 100,
                    "vol": 1000.0,
                    "amount": 1e9,
                }
            )
            last_close[code] = close_price
    return pl.DataFrame(factor_rows), pl.DataFrame(price_rows)

def test_walk_forward_summary_marks_insufficient_data():
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    factor_df, price_df = _make_factor_price(n_dates=12)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        walk_forward={
            "enabled": True,
            "train_days": 20,
            "test_days": 5,
            "step_days": 5,
            "embargo_days": 2,
        },
    )

    summary, result = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert result is None
    assert summary["status"] == "insufficient_data"
    assert summary["n_folds"] == 0
    assert summary["requested_n_trials"] == 50
    assert summary["param_candidates"][-1] == {"top_n": 50}

def test_walk_forward_summary_returns_oos_metrics_when_folds_exist():
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    factor_df, price_df = _make_factor_price(n_dates=36, n_stocks=80)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240228",
        backtest={"quantiles": 4},
        walk_forward={
            "enabled": True,
            "train_days": 12,
            "test_days": 6,
            "step_days": 6,
            "embargo_days": 1,
        },
    )

    summary, result = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert result is not None
    assert summary["status"] == "ok"
    assert summary["n_folds"] > 0
    assert summary["is_sharpe_mean"] == result.is_sharpe_mean
    assert summary["oos_sharpe_mean"] == result.oos_sharpe_mean
    assert summary["oos_sharpe_std"] == result.oos_sharpe_std
    assert summary["oos_max_dd"] == result.oos_max_dd
    assert summary["stability_ratio"] == result.stability_ratio

def test_walk_forward_summary_uses_top_n_candidates_from_n_trials(monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.backtest import PrecomputedWeightsStrategy
    from factorzen.daily.evaluation.walk_forward import WalkForwardResult
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    captured = {}

    def fake_run_walk_forward_search(**kwargs):
        captured["param_candidates"] = kwargs["param_candidates"]
        captured["strategy"] = kwargs["strategy_factory"]({"top_n": 10})
        return WalkForwardResult(
            folds=[],
            oos_returns=pl.DataFrame(),
            is_sharpe_mean=0.0,
            oos_sharpe_mean=0.0,
            oos_sharpe_std=0.0,
            oos_max_dd=0.0,
            stability_ratio=0.0,
        )

    monkeypatch.setattr(
        "factorzen.daily.evaluation.walk_forward_summary.run_walk_forward_search",
        fake_run_walk_forward_search,
    )
    factor_df, price_df = _make_factor_price(n_dates=12)
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        backtest={"top_n": 10},
        walk_forward={"enabled": True, "n_trials": 4},
    )

    summary, _ = run_quantile_walk_forward_summary(
        factor_df,
        price_df,
        cfg,
        factor_name="momentum_20d",
        frequency="daily",
    )

    assert captured["param_candidates"] == [{"top_n": 10}]
    assert isinstance(captured["strategy"], PrecomputedWeightsStrategy)
    assert summary["requested_n_trials"] == 4

def test_walk_forward_summary_skips_runner_when_disabled(monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.daily.evaluation.walk_forward_summary import run_quantile_walk_forward_summary

    def unexpected_runner(**_kwargs):
        raise AssertionError("disabled walk-forward must not invoke the runner")

    monkeypatch.setattr(
        "factorzen.daily.evaluation.walk_forward_summary.run_walk_forward_search",
        unexpected_runner,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
    )

    summary, result = run_quantile_walk_forward_summary(
        pl.DataFrame(),
        pl.DataFrame(),
        cfg,
        factor_name="momentum_20d",
    )

    assert summary == {"status": "disabled", "n_folds": 0}
    assert result is None

def test_walk_forward_optimized_path_matches_sequential_search():
    from factorzen.daily.evaluation.backtest import BacktestConfig, TopNLongOnlyStrategy
    from factorzen.daily.evaluation.walk_forward import WalkForwardSplitter, run_walk_forward_search

    factor_df, price_df = _make_factor_price(n_dates=32, n_stocks=40)
    splitter = WalkForwardSplitter(train_days=10, test_days=5, step_days=5, embargo_days=1)
    candidates = [{"top_n": 10}, {"top_n": 20}]
    cfg = BacktestConfig(max_abs_weight=0.05, max_participation_rate=1.0)

    def strategy_factory(params):
        return TopNLongOnlyStrategy(n=params["top_n"])

    sequential = run_walk_forward_search(
        strategy_factory=strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        param_candidates=candidates,
        config=cfg,
        factor_name="x",
        reuse_is_backtests=False,
        parallel_workers=1,
    )
    optimized = run_walk_forward_search(
        strategy_factory=strategy_factory,
        factor_df=factor_df,
        price_df=price_df,
        splitter=splitter,
        param_candidates=candidates,
        config=cfg,
        factor_name="x",
        reuse_is_backtests=True,
        parallel_workers=2,
    )

    assert optimized.oos_returns.equals(sequential.oos_returns)
    assert optimized.is_sharpe_mean == sequential.is_sharpe_mean
    assert optimized.oos_sharpe_mean == sequential.oos_sharpe_mean
    assert optimized.oos_sharpe_std == sequential.oos_sharpe_std
    assert optimized.oos_max_dd == sequential.oos_max_dd
    assert [fold.params for fold in optimized.folds] == [fold.params for fold in sequential.folds]

# ==== 来自 test_output_paths.py ====
def test_qlib_alpha158_outputs_go_to_qlib158_bucket():
    factor = "qlib_alpha158_kmid"

    assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS / "qlib158"
    assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS / "qlib158"
    assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS / "qlib158"

def test_qlib_alpha360_outputs_go_to_qlib360_bucket():
    factor = "qlib_alpha360_close0"

    assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS / "qlib360"
    assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS / "qlib360"
    assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS / "qlib360"

def test_personal_factor_outputs_stay_in_daily_roots():
    factor = "momentum_20d"

    assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS
    assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS
    assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS

