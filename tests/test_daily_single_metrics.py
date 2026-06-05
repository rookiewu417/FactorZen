"""daily_single._write_run_metrics 单测：sweep 读取的 IC + 主策略回测指标 JSON。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from factorzen.pipelines.daily_single import _write_run_metrics


def _fake_ic_result():
    return SimpleNamespace(
        ic_mean=0.0397,
        ir=0.13,
        ic_tstat=1.95,
        ic_positive_ratio=0.55,
        n_periods=241,
    )


def _fake_bt_result(portfolio):
    return SimpleNamespace(summary_stats={"portfolio": portfolio, "long_short": portfolio})


def test_write_run_metrics_includes_ic_and_backtest(tmp_path):
    path = tmp_path / "metrics.json"
    _write_run_metrics(
        str(path),
        _fake_ic_result(),
        _fake_bt_result({"sharpe": -1.15, "ann_ret": -0.022, "avg_turnover": 0.51, "max_dd": -0.035}),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ic_mean"] == 0.0397
    assert data["ir"] == 0.13
    assert data["t"] == 1.95
    assert data["ic_pos"] == 0.55
    assert data["n"] == 241
    assert data["sharpe"] == -1.15
    assert data["ann_ret"] == -0.022
    assert data["avg_turnover"] == 0.51
    assert data["max_dd"] == -0.035


def test_write_run_metrics_tolerates_missing_portfolio(tmp_path):
    """回测 summary 缺 portfolio 键时回测指标置 None，不抛异常。"""
    path = tmp_path / "metrics.json"
    bad_bt = SimpleNamespace(summary_stats={})
    _write_run_metrics(str(path), _fake_ic_result(), bad_bt)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ic_mean"] == 0.0397
    assert data["sharpe"] is None
    assert data["ann_ret"] is None
