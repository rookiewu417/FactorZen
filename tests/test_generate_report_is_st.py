"""generate_report 回测须传 is_st_by_date（与 daily_single 一致，消除 ST 涨跌停双路径）。"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import polars as pl


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
