"""Tests for generate_report result persistence metadata."""

from __future__ import annotations

import json
from datetime import date

import polars as pl


def test_save_results_persists_quality_report_metadata(tmp_path, monkeypatch):
    from daily.evaluation.backtest import StrategyBacktestResult
    from daily.evaluation.ic_analysis import ICAnalysisResult
    from daily.evaluation.turnover import TurnoverResult
    from scripts import generate_report as mod

    monkeypatch.setattr(mod, "OUTPUT_DAILY_FACTORS", tmp_path / "factors")
    monkeypatch.setattr(mod, "OUTPUT_DAILY_RESULTS", tmp_path / "results")

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
