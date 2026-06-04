"""_report_persistence / _report_direction 的离线单测：
覆盖 _save_results → _load_results 往返、walk-forward/direction 回读、
_existing_report_outputs 与 _save_quality_report。全部写入 tmp_path，不碰真实 workspace。
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from factorzen.pipelines import _report_direction as direction
from factorzen.pipelines import _report_persistence as persist


@pytest.fixture
def results():
    """构造一组最小但字段完整的评价结果对象。"""
    from factorzen.daily.evaluation.backtest import StrategyBacktestResult
    from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
    from factorzen.daily.evaluation.turnover import TurnoverResult

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
    return clean_df, ic_result, bt_result, to_result


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    """把 persist 的输出目录重定向到 tmp_path。"""
    monkeypatch.setattr(persist, "daily_factor_output_dir", lambda f: tmp_path / "factors")
    monkeypatch.setattr(persist, "daily_result_output_dir", lambda f: tmp_path / "results")
    monkeypatch.setattr(persist, "daily_report_output_dir", lambda f: tmp_path / "reports")
    return tmp_path


def _save(results, **kw):
    clean_df, ic_result, bt_result, to_result = results
    persist._save_results(
        "momentum_20d", "20240101", "20240131", clean_df, ic_result, bt_result, to_result, **kw
    )


# ── 往返：save → load ───────────────────────────────────────


def test_save_load_round_trip(tmp_dirs, results):
    _save(results)
    loaded = persist._load_results("momentum_20d", "20240101", "20240131")
    assert loaded is not None
    _clean, ic, bt, to = loaded
    assert ic.factor_name == "momentum_20d"
    assert ic.ic_mean == pytest.approx(0.01)
    assert ic.n_periods == 1
    assert bt.factor_name == "momentum_20d"
    assert bt.n_groups == 5
    assert to.avg_turnover == pytest.approx(0.1)


def test_load_results_missing_meta_returns_none(tmp_dirs):
    assert persist._load_results("momentum_20d", "20240101", "20240131") is None


def test_load_results_missing_parquet_returns_none(tmp_dirs, results):
    _save(results)
    # 删除其中一个必需 parquet → 退回重新计算（None）
    (tmp_dirs / "results" / "momentum_20d_20240101_20240131_ic.parquet").unlink()
    assert persist._load_results("momentum_20d", "20240101", "20240131") is None


# ── walk-forward / direction 回读 ───────────────────────────


def test_load_walk_forward_summary_round_trip(tmp_dirs, results):
    summary = {"status": "ok", "n_folds": 3, "oos_sharpe_mean": 0.7}
    _save(results, walk_forward_summary=summary)
    assert persist._load_walk_forward_summary("momentum_20d", "20240101", "20240131") == summary


def test_load_walk_forward_summary_missing_returns_none(tmp_dirs):
    assert persist._load_walk_forward_summary("x", "20240101", "20240131") is None


def test_load_backtest_direction_round_trip(tmp_dirs, results):
    decision = {"direction": "reversed", "should_reverse": True, "reason": "neg IC"}
    _save(results, backtest_direction=decision)
    loaded = direction._load_backtest_direction("momentum_20d", "20240101", "20240131")
    assert loaded["direction"] == "reversed"
    assert loaded["should_reverse"] is True


def test_load_backtest_direction_missing_returns_none(tmp_dirs):
    assert direction._load_backtest_direction("x", "20240101", "20240131") is None


# ── _existing_report_outputs / _save_quality_report ─────────


def test_existing_report_outputs_lists_present_files(tmp_dirs, results):
    _save(results)
    persist._save_quality_report("momentum_20d", "20240101", "20240131", {"status": "ok"})
    outputs = persist._existing_report_outputs("momentum_20d", "20240101", "20240131")
    assert "meta" in outputs
    assert "quality_report" in outputs
    assert outputs["meta"].endswith("_meta.json")


def test_existing_report_outputs_empty_when_nothing_saved(tmp_dirs):
    assert persist._existing_report_outputs("x", "20240101", "20240131") == {}


def test_save_quality_report_writes_json(tmp_dirs):
    import json

    path = persist._save_quality_report(
        "momentum_20d", "20240101", "20240131", {"status": "warning", "warnings": ["w"]}
    )
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "warning"
