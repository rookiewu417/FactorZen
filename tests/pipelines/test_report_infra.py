"""合并自: test_generate_report.py, test_pipeline_infra.py
目标: test_report_infra.py

--- 来源 test_generate_report.py ---
test_generate_report_is_st.py：generate_report 回测须传 is_st_by_date（与 daily_single 一致，消除 ST 涨跌停双路径）。
test_generate_report_persistence.py：Tests for generate_report result persistence metadata.
test_report_forward_returns.py：fz report build 前向收益/IC 标签须用复权收盘价，与 fz factor run 口径一致，

--- 来源 test_pipeline_infra.py ---
test_membership_vectorized_equiv.py：universe membership 向量化等价性：与逐日 _load_index_members 完全一致。
test_walk_forward_summary.py：Tests for single-factor walk-forward summary integration.
test_output_paths.py：无 module docstring 的测试。
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from types import SimpleNamespace

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


# ==== 来自 test_generate_report.py ====
# ==== 来自 test_generate_report_is_st.py ====
def test_run_backtest_strategies_threads_is_st_by_date(monkeypatch):
    import factorzen.pipelines.generate_report as gr

    daily = pl.DataFrame({"ts_code": ["A.SZ", "B.SZ"],
                          "trade_date": [date(2024, 1, 1), date(2024, 1, 1)]})
    clean = pl.DataFrame({"ts_code": ["A.SZ"], "trade_date": [date(2024, 1, 1)],
                          "factor_clean": [0.1]})
    st_map = {date(2024, 1, 1): {"A.SZ"}}
    # build_is_st_by_date 在函数内 import，patch 源模块
    monkeypatch.setattr("factorzen.core.universe.build_is_st_by_date", lambda codes, dates: st_map)
    monkeypatch.setattr(gr, "build_backtest_strategies", lambda c: {"topn": object()})
    monkeypatch.setattr(gr, "build_runtime_backtest_config", lambda *a, **k: None)
    monkeypatch.setattr(gr, "build_cost_model", lambda *a, **k: None)
    monkeypatch.setattr(gr, "trim_backtest_to_first_trade", lambda r: r)
    monkeypatch.setattr(gr, "logger", SimpleNamespace(info=lambda *a, **k: None))

    captured: dict = {}

    def fake_bt(strategy, clean_df, dly, *, config, cost_model, factor_name, is_st_by_date=None):
        captured["is_st"] = is_st_by_date
        return SimpleNamespace(summary=lambda: "ok")

    monkeypatch.setattr(gr, "run_strategy_backtest", fake_bt)

    config = SimpleNamespace(
        backtest=SimpleNamespace(strategy_specs=[SimpleNamespace(name="topn")], primary="topn"))
    gr._run_backtest_strategies(config, clean, daily, factor_name="f", frequency="daily")
    assert captured["is_st"] == st_map, "回测应收到 is_st_by_date（ST PIT 涨跌停阈值）"

# ==== 来自 test_generate_report_persistence.py ====
def test_save_results_persists_quality_report_metadata(tmp_path, monkeypatch):
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult
    from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
    from factorzen.daily.evaluation.turnover import TurnoverResult

    # _save_results 已拆到 _report_persistence，路径构造函数在该模块命名空间解析
    from factorzen.pipelines import _report_persistence as mod

    monkeypatch.setattr(mod, "daily_factor_output_dir", lambda factor_name: tmp_path / "factors")
    monkeypatch.setattr(mod, "daily_result_output_dir", lambda factor_name: tmp_path / "results")

    clean_df = pl.DataFrame(
        {"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"], "factor_clean": [1.0]}
    )
    ic_result = ICAnalysisResult(
        factor_name="momentum_20d",
        ic_mean=0.01,
        ic_std=0.02,
        ir=0.5,
        ic_positive_ratio=1.0,
        n_periods=1,
        ic_series=pl.DataFrame({"trade_date": [date(2024, 1, 2)], "ic": [0.01]}),
    )
    returns = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "gross_return": [0.0],
            "cost": [0.0],
            "borrow_cost": [0.0],
            "net_return": [0.0],
            "nav": [1.0],
            "cash_weight": [1.0],
            "turnover": [0.0],
        }
    )
    bt_result = StrategyBacktestResult(
        factor_name="momentum_20d",
        strategy_name="quantile_long_short",
        n_groups=5,
        returns=returns,
        nav=returns.select(
            ["trade_date", "gross_return", "cost", "borrow_cost", "net_return", "nav", "cash_weight"]
        ),
        positions=pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "weight": pl.Float64,
                "market_value": pl.Float64,
            }
        ),
        trades=pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "prev_weight": pl.Float64,
                "target_weight": pl.Float64,
                "filled_delta_weight": pl.Float64,
                "turnover": pl.Float64,
                "cost": pl.Float64,
                "block_reason": pl.Utf8,
            }
        ),
        summary_stats={"portfolio": {"sharpe": 0.0}},
        config={"max_abs_weight": 0.1},
    )
    to_result = TurnoverResult(
        factor_name="momentum_20d",
        avg_turnover=0.1,
        daily_turnover=pl.DataFrame({"trade_date": [date(2024, 1, 2)], "turnover": [0.1]}),
        migration_matrix=pl.DataFrame({"from": [0], "to": [1], "count": [1]}),
    )

    mod._save_results(
        "momentum_20d",
        "20240101",
        "20240131",
        clean_df,
        ic_result,
        bt_result,
        to_result,
        quality_report={"status": "warning", "warnings": ["low coverage"]},
        quality_path=tmp_path / "quality.json",
        walk_forward_summary={
            "status": "ok",
            "n_folds": 2,
            "is_sharpe_mean": 1.1,
            "oos_sharpe_mean": 0.8,
            "oos_sharpe_std": 0.2,
            "oos_max_dd": -0.05,
            "stability_ratio": 0.72,
        },
    )

    meta = json.loads(
        (tmp_path / "results" / "momentum_20d_20240101_20240131_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["quality_status"] == "warning"
    assert meta["quality_warnings"] == ["low coverage"]
    assert meta["quality_report_path"] == str(tmp_path / "quality.json")
    assert meta["walk_forward_summary"] == {
        "status": "ok",
        "n_folds": 2,
        "is_sharpe_mean": 1.1,
        "oos_sharpe_mean": 0.8,
        "oos_sharpe_std": 0.2,
        "oos_max_dd": -0.05,
        "stability_ratio": 0.72,
    }

def test_backtest_direction_from_ic_suite():
    """test_negative_significant_ic_uses_reversed_backtest_direction；test_weak_negative_ic_keeps_normal_backtest_direction；test_reversed_backtest_direction_flips_factor_clean"""
    # -- 原 test_negative_significant_ic_uses_reversed_backtest_direction --
    def _section_0_test_negative_significant_ic_uses_reversed_backtest_direction():
        from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
        from factorzen.pipelines import generate_report as mod

        ic_result = ICAnalysisResult(
            factor_name="value",
            ic_mean=-0.03,
            ic_std=0.04,
            ir=-0.75,
            ic_positive_ratio=0.3,
            n_periods=60,
            ic_series=pl.DataFrame(),
            ic_tstat=-1.8,
            ic_pvalue=0.08,
        )

        decision = mod._decide_backtest_direction(ic_result)

        assert decision["direction"] == "reversed"
        assert decision["should_reverse"] is True

    _section_0_test_negative_significant_ic_uses_reversed_backtest_direction()

    # -- 原 test_weak_negative_ic_keeps_normal_backtest_direction --
    def _section_1_test_weak_negative_ic_keeps_normal_backtest_direction():
        from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
        from factorzen.pipelines import generate_report as mod

        ic_result = ICAnalysisResult(
            factor_name="noise",
            ic_mean=-0.005,
            ic_std=0.04,
            ir=-0.125,
            ic_positive_ratio=0.48,
            n_periods=60,
            ic_series=pl.DataFrame(),
            ic_tstat=-0.5,
            ic_pvalue=0.62,
            oos_ic={"train": -0.004, "test": 0.002},
        )

        decision = mod._decide_backtest_direction(ic_result)

        assert decision["direction"] == "normal"
        assert decision["should_reverse"] is False

    _section_1_test_weak_negative_ic_keeps_normal_backtest_direction()

    # -- 原 test_reversed_backtest_direction_flips_factor_clean --
    def _section_2_test_reversed_backtest_direction_flips_factor_clean():
        from factorzen.pipelines import generate_report as mod

        clean_df = pl.DataFrame(
            {"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"], "factor_clean": [2.0]}
        )

        out = mod._apply_backtest_direction(clean_df, {"direction": "reversed"})

        assert out["factor_clean"].to_list() == [-2.0]

    _section_2_test_reversed_backtest_direction_flips_factor_clean()


def test_merge_report_config_suite():
    """test_merge_report_config_args_uses_yaml_and_defaults_benchmark；test_merge_report_config_args_keeps_explicit_benchmark；双路径对齐：report 无 YAML 时必须与 daily_single 用同一份研究预设。；无 YAML 时 universe 兜底须与 fz factor run 研究预设一致（csi500）。"""
    # -- 原 test_merge_report_config_args_uses_yaml_and_defaults_benchmark --
    def _section_0_test_merge_report_config_args_uses_yaml_and_defaults_benchmark():
        from argparse import Namespace

        from factorzen.config.research import RunConfig
        from factorzen.pipelines import generate_report as mod

        args = Namespace(
            factor=None,
            start=None,
            end=None,
            universe=None,
            benchmark=None,
            frequency="daily",
            reuse=False,
            config=None,
        )
        cfg = RunConfig(
            factor="momentum_20d",
            start="20230101",
            end="20241231",
            universe="csi500",
            benchmark=None,
        )

        merged = mod._merge_report_config_args(args, cfg)

        assert merged.factor == "momentum_20d"
        assert merged.start == "20230101"
        assert merged.end == "20241231"
        assert merged.universe == "csi500"
        assert merged.benchmark == "000905.SH"
        for banned in ("ic_method", "neutralized_ic", "event_study", "llm_explain", "llm_refresh", "all"):
            assert banned not in vars(merged)

    _section_0_test_merge_report_config_args_uses_yaml_and_defaults_benchmark()

    # -- 原 test_merge_report_config_args_keeps_explicit_benchmark --
    def _section_1_test_merge_report_config_args_keeps_explicit_benchmark():
        from argparse import Namespace

        from factorzen.config.research import RunConfig
        from factorzen.pipelines import generate_report as mod

        args = Namespace(
            factor=None,
            start=None,
            end=None,
            universe="csi500",
            benchmark="000300.SH",
            frequency="daily",
            reuse=True,
            config=None,
        )
        cfg = RunConfig(
            factor="momentum_20d",
            start="20230101",
            end="20241231",
            universe="csi800",
            benchmark="000905.SH",
        )

        merged = mod._merge_report_config_args(args, cfg)

        assert merged.benchmark == "000300.SH"
        assert merged.reuse is True
        assert merged.universe == "csi500"

    _section_1_test_merge_report_config_args_keeps_explicit_benchmark()

    # -- 原 test_effective_report_config_without_yaml_matches_daily_single_preset --
    def _section_2_test_effective_report_config_without_yaml_matches_daily_single_preset():
        from argparse import Namespace

        from factorzen.config.research import build_default_daily_research_config
        from factorzen.pipelines import generate_report as mod

        args = Namespace(
            factor="momentum_20d",
            start="20240101",
            end="20240131",
            universe=None,
            benchmark=None,
            frequency="daily",
            reuse=False,
            config=None,
        )

        merged = mod._merge_report_config_args(args, None)
        cfg = mod._effective_report_config(merged, None)

        daily_preset = build_default_daily_research_config(
            factor="momentum_20d",
            start="20240101",
            end="20240131",
            universe=merged.universe,
            benchmark=merged.benchmark,
        )
        assert [spec.name for spec in cfg.backtest.strategy_specs] == [
            spec.name for spec in daily_preset.backtest.strategy_specs
        ] == ["quantile_ls_5"]
        assert cfg.backtest.primary == daily_preset.backtest.primary == "quantile_ls_5"
        assert cfg.preprocessing == daily_preset.preprocessing
        for banned in ("ic_method", "neutralized_ic", "event_study"):
            assert banned not in cfg.model_dump()

    _section_2_test_effective_report_config_without_yaml_matches_daily_single_preset()

    # -- 原 test_merge_report_config_args_default_universe_csi500 --
    def _section_3_test_merge_report_config_args_default_universe_csi500():
        from argparse import Namespace

        from factorzen.pipelines import generate_report as mod

        args = Namespace(
            factor="momentum_20d",
            start="20240101",
            end="20240131",
            universe=None,
            benchmark=None,
            frequency="daily",
            reuse=False,
            config=None,
        )

        merged = mod._merge_report_config_args(args, None)
        assert merged.universe == "csi500"
        assert merged.benchmark == "000905.SH"

    _section_3_test_merge_report_config_args_default_universe_csi500()


# ==== 来自 test_report_forward_returns.py ====
def test_attach_close_adj_suite():
    """test_attach_close_adj_derives_adjusted_close；test_attach_close_adj_empty_adj_returns_unchanged；除权日 close 跳空（送转 10→5）不应污染前向收益——复权后 close_adj 连续。"""
    # -- 原 test_attach_close_adj_derives_adjusted_close --
    def _section_0_test_attach_close_adj_derives_adjusted_close():
        from factorzen.pipelines.generate_report import _attach_close_adj

        daily = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ts_code": ["A.SZ", "A.SZ"],
                "close": [10.0, 11.0],
            }
        )
        adj = pl.DataFrame(
            {
                "ts_code": ["A.SZ", "A.SZ"],
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "adj_factor": [2.0, 2.0],
            }
        )
        out = _attach_close_adj(daily, adj)
        assert out.sort("trade_date")["close_adj"].to_list() == [20.0, 22.0]

    _section_0_test_attach_close_adj_derives_adjusted_close()

    # -- 原 test_attach_close_adj_empty_adj_returns_unchanged --
    def _section_1_test_attach_close_adj_empty_adj_returns_unchanged():
        from factorzen.pipelines.generate_report import _attach_close_adj

        daily = pl.DataFrame(
            {"trade_date": [date(2024, 1, 1)], "ts_code": ["A.SZ"], "close": [10.0]}
        )
        out = _attach_close_adj(daily, pl.DataFrame())
        assert "close_adj" not in out.columns  # adj 缺失 → 回退，_build_forward_return_frame 用 close

    _section_1_test_attach_close_adj_empty_adj_returns_unchanged()

    # -- 原 test_report_forward_returns_use_adjusted_close_no_ex_div_jump --
    def _section_2_test_report_forward_returns_use_adjusted_close_no_ex_div_jump():
        from factorzen.pipelines.daily_single import _build_forward_return_frame
        from factorzen.pipelines.generate_report import _attach_close_adj

        daily = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "ts_code": ["A.SZ", "A.SZ", "A.SZ"],
                "close": [10.0, 5.0, 5.0],  # d2 送转，未复权价腰斩
            }
        )
        adj = pl.DataFrame(
            {
                "ts_code": ["A.SZ", "A.SZ", "A.SZ"],
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "adj_factor": [1.0, 2.0, 2.0],  # 除权后翻倍 → close_adj = [10,10,10] 连续
            }
        )
        daily_adj = _attach_close_adj(daily, adj)
        ret_df = _build_forward_return_frame(daily_adj)
        fwd = ret_df.sort("trade_date")["fwd_ret_1d"].to_list()
        # 复权后 close_adj 连续 → d1→d2 的 fwd_ret_1d = 0，而非未复权的虚假 -50%
        assert abs(fwd[0] - 0.0) < 1e-9, f"复权后不应有除权跳空，实际 fwd_ret_1d={fwd[0]}"

    _section_2_test_report_forward_returns_use_adjusted_close_no_ex_div_jump()


# ==== 来自 test_pipeline_infra.py ====
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
@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
@pytest.mark.skipif(not _has_csi500_cache(), reason="需要本地 csi500 成分缓存")
def test_membership_vectorized_suite():
    """跨月窗口：逐日成分集合与旧实现完全一致。；含月初/月末边界与成分调整月（6 月调样窗口）。；2 年 csi500 membership ≤ 0.5s。；test_membership_csi800_union_equiv"""
    # -- 原 test_membership_vectorized_matches_daily_loop_cross_month --
    def _section_0_test_membership_vectorized_matches_daily_loop_cross_month():
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

    _section_0_test_membership_vectorized_matches_daily_loop_cross_month()

    # -- 原 test_membership_vectorized_month_boundary_and_rebalance --
    def _section_1_test_membership_vectorized_month_boundary_and_rebalance():
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

    _section_1_test_membership_vectorized_month_boundary_and_rebalance()

    # -- 原 test_membership_vectorized_two_year_perf --
    def _section_2_test_membership_vectorized_two_year_perf():
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

    _section_2_test_membership_vectorized_two_year_perf()

    # -- 原 test_membership_csi800_union_equiv --
    def _section_3_test_membership_csi800_union_equiv():
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

    _section_3_test_membership_csi800_union_equiv()


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

def test_walk_forward_summary_suite():
    """test_walk_forward_summary_marks_insufficient_data；test_walk_forward_summary_returns_oos_metrics_when_folds_exist；test_walk_forward_summary_uses_top_n_candidates_from_n_trials；test_walk_forward_summary_skips_runner_when_disabled；test_walk_forward_optimized_path_matches_sequential_search"""
    # -- 原 test_walk_forward_summary_marks_insufficient_data --
    def _section_0_test_walk_forward_summary_marks_insufficient_data():
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.walk_forward_summary import (
            run_quantile_walk_forward_summary,
        )

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

    _section_0_test_walk_forward_summary_marks_insufficient_data()

    # -- 原 test_walk_forward_summary_returns_oos_metrics_when_folds_exist --
    def _section_1_test_walk_forward_summary_returns_oos_metrics_when_folds_exist():
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.walk_forward_summary import (
            run_quantile_walk_forward_summary,
        )

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

    _section_1_test_walk_forward_summary_returns_oos_metrics_when_folds_exist()

    # -- 原 test_walk_forward_summary_uses_top_n_candidates_from_n_trials --
    def _section_2_test_walk_forward_summary_uses_top_n_candidates_from_n_trials(mp):
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.backtest import PrecomputedWeightsStrategy
        from factorzen.daily.evaluation.walk_forward import WalkForwardResult
        from factorzen.daily.evaluation.walk_forward_summary import (
            run_quantile_walk_forward_summary,
        )

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

        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_walk_forward_summary_uses_top_n_candidates_from_n_trials(mp)

    # -- 原 test_walk_forward_summary_skips_runner_when_disabled --
    def _section_3_test_walk_forward_summary_skips_runner_when_disabled(mp):
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.walk_forward_summary import (
            run_quantile_walk_forward_summary,
        )

        def unexpected_runner(**_kwargs):
            raise AssertionError("disabled walk-forward must not invoke the runner")

        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_walk_forward_summary_skips_runner_when_disabled(mp)

    # -- 原 test_walk_forward_optimized_path_matches_sequential_search --
    def _section_4_test_walk_forward_optimized_path_matches_sequential_search():
        from factorzen.daily.evaluation.backtest import BacktestConfig, TopNLongOnlyStrategy
        from factorzen.daily.evaluation.walk_forward import (
            WalkForwardSplitter,
            run_walk_forward_search,
        )

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

    _section_4_test_walk_forward_optimized_path_matches_sequential_search()


# ==== 来自 test_output_paths.py ====
def test_qlib_output_bucket_suite():
    """test_qlib_alpha158_outputs_go_to_qlib158_bucket；test_qlib_alpha360_outputs_go_to_qlib360_bucket；test_personal_factor_outputs_stay_in_daily_roots"""
    # -- 原 test_qlib_alpha158_outputs_go_to_qlib158_bucket --
    def _section_0_test_qlib_alpha158_outputs_go_to_qlib158_bucket():
        factor = "qlib_alpha158_kmid"

        assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS / "qlib158"
        assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS / "qlib158"
        assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS / "qlib158"

    _section_0_test_qlib_alpha158_outputs_go_to_qlib158_bucket()

    # -- 原 test_qlib_alpha360_outputs_go_to_qlib360_bucket --
    def _section_1_test_qlib_alpha360_outputs_go_to_qlib360_bucket():
        factor = "qlib_alpha360_close0"

        assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS / "qlib360"
        assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS / "qlib360"
        assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS / "qlib360"

    _section_1_test_qlib_alpha360_outputs_go_to_qlib360_bucket()

    # -- 原 test_personal_factor_outputs_stay_in_daily_roots --
    def _section_2_test_personal_factor_outputs_stay_in_daily_roots():
        factor = "momentum_20d"

        assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS
        assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS
        assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS

    _section_2_test_personal_factor_outputs_stay_in_daily_roots()


