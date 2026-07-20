"""test_generate_report_is_st.py：generate_report 回测须传 is_st_by_date（与 daily_single 一致，消除 ST 涨跌停双路径）。
test_generate_report_persistence.py：Tests for generate_report result persistence metadata.
test_report_forward_returns.py：fz report build 前向收益/IC 标签须用复权收盘价，与 fz factor test 口径一致，
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import polars as pl


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

def test_merge_report_config_args_uses_yaml_and_defaults_benchmark():
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

def test_merge_report_config_args_keeps_explicit_benchmark():
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

def test_effective_report_config_without_yaml_matches_daily_single_preset():
    """双路径对齐：report 无 YAML 时必须与 daily_single 用同一份研究预设。"""
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

def test_merge_report_config_args_default_universe_csi500():
    """无 YAML 时 universe 兜底须与 fz factor run 研究预设一致（csi500）。"""
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

# ==== 来自 test_report_forward_returns.py ====
def test_attach_close_adj_derives_adjusted_close():
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

def test_attach_close_adj_empty_adj_returns_unchanged():
    from factorzen.pipelines.generate_report import _attach_close_adj

    daily = pl.DataFrame(
        {"trade_date": [date(2024, 1, 1)], "ts_code": ["A.SZ"], "close": [10.0]}
    )
    out = _attach_close_adj(daily, pl.DataFrame())
    assert "close_adj" not in out.columns  # adj 缺失 → 回退，_build_forward_return_frame 用 close

def test_report_forward_returns_use_adjusted_close_no_ex_div_jump():
    """除权日 close 跳空（送转 10→5）不应污染前向收益——复权后 close_adj 连续。"""
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

