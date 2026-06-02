"""Tests for generate_report result persistence metadata."""

from __future__ import annotations

import json
from datetime import date

import polars as pl


def test_save_results_persists_quality_report_metadata(tmp_path, monkeypatch):
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult
    from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
    from factorzen.daily.evaluation.turnover import TurnoverResult
    from factorzen.pipelines import generate_report as mod

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


def test_negative_significant_ic_uses_reversed_backtest_direction():
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


def test_weak_negative_ic_keeps_normal_backtest_direction():
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


def test_reversed_backtest_direction_flips_factor_clean():
    from factorzen.pipelines import generate_report as mod

    clean_df = pl.DataFrame(
        {"trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SZ"], "factor_clean": [2.0]}
    )

    out = mod._apply_backtest_direction(clean_df, {"direction": "reversed"})

    assert out["factor_clean"].to_list() == [-2.0]


def test_merge_report_config_args_all_enables_report_defaults():
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
        all=True,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = mod._merge_report_config_args(args, None)

    assert merged.universe == "csi300"
    assert merged.benchmark == "000300.SH"
    assert merged.reuse is False
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True
    assert merged.llm_explain is True
    assert merged.llm_refresh is False


def test_merge_report_config_args_all_uses_universe_matched_benchmark():
    from argparse import Namespace

    from factorzen.pipelines import generate_report as mod

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe="csi500",
        benchmark=None,
        frequency="daily",
        reuse=False,
        config=None,
        all=True,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = mod._merge_report_config_args(args, None)

    assert merged.benchmark == "000905.SH"


def test_merge_report_config_args_all_overrides_yaml_benchmark():
    from argparse import Namespace

    from factorzen.core.config_loader import RunConfig
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
        all=True,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        llm_explain=False,
        llm_refresh=False,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20241231",
        universe="csi500",
        benchmark="000300.SH",
    )

    merged = mod._merge_report_config_args(args, cfg)

    assert merged.benchmark == "000905.SH"
    assert merged.reuse is False
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True
    assert merged.llm_explain is True


def test_merge_report_config_args_all_keeps_explicit_deep_options():
    from argparse import Namespace

    from factorzen.core.config_loader import RunConfig
    from factorzen.pipelines import generate_report as mod

    args = Namespace(
        factor=None,
        start=None,
        end=None,
        universe=None,
        benchmark="000300.SH",
        frequency="daily",
        reuse=False,
        config=None,
        all=True,
        ic_method="pearson",
        neutralized_ic=False,
        event_study=False,
        llm_explain=False,
        llm_refresh=False,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20241231",
        universe="csi500",
        benchmark="000905.SH",
    )

    merged = mod._merge_report_config_args(args, cfg)

    assert merged.benchmark == "000300.SH"
    assert merged.ic_method == "pearson"
    assert merged.neutralized_ic is False
    assert merged.event_study is False


def test_effective_report_config_without_yaml_uses_default_strategy_suite():
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
        all=False,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = mod._merge_report_config_args(args, None)
    cfg = mod._effective_report_config(merged, None)

    assert [spec.name for spec in cfg.backtest.strategy_specs] == [
        "topn_50",
        "quantile_ls_5",
        "factor_weighted_ls",
        "optimizer_mv_long_only",
    ]
