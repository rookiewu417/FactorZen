"""回测方向判定：IC 对齐（负 IC 自动取反）与 factor run / report 路径共用逻辑。"""

from __future__ import annotations

import json
from datetime import date

import polars as pl

from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.pipelines._report_direction import (
    _apply_backtest_direction,
    _decide_backtest_direction,
)


def _ic(**kwargs) -> ICAnalysisResult:
    base = dict(
        factor_name="f",
        ic_mean=0.0,
        ic_std=0.1,
        ir=0.0,
        ic_positive_ratio=0.5,
        n_periods=100,
        ic_series=pl.DataFrame(),
        ic_tstat=0.0,
        ic_pvalue=1.0,
        oos_ic={},
    )
    base.update(kwargs)
    return ICAnalysisResult(**base)


def test_significant_negative_ic_reverses():
    d = _decide_backtest_direction(
        _ic(ic_mean=-0.03, ic_tstat=-3.0, ic_pvalue=0.003, ir=-0.3)
    )
    assert d["direction"] == "reversed"
    assert d["should_reverse"] is True
    assert "p 值" in d["reason"] or "负" in d["reason"]


def test_pvalue_zero_is_not_treated_as_missing():
    """``ic_pvalue=0.0`` 必须保留，不能被 ``x or 1.0`` 吃成不显著。"""
    d = _decide_backtest_direction(
        _ic(ic_mean=-0.03, ic_tstat=-13.0, ic_pvalue=0.0, ir=-0.3)
    )
    assert d["direction"] == "reversed"
    assert d["ic_pvalue"] == 0.0
    assert "p 值" in d["reason"]


def test_weak_negative_ic_keeps_normal():
    d = _decide_backtest_direction(
        _ic(
            ic_mean=-0.005,
            ic_tstat=-0.4,
            ic_pvalue=0.7,
            oos_ic={"train": -0.004, "test": 0.002},
        )
    )
    assert d["direction"] == "normal"
    assert d["should_reverse"] is False


def test_oos_both_negative_reverses_even_if_p_weak():
    """IS/OOS 两段 IC 均为负时，即使全样本 p 略高也对齐交易方向。"""
    d = _decide_backtest_direction(
        _ic(
            ic_mean=-0.02,
            ic_tstat=-1.5,
            ic_pvalue=0.14,  # > 0.10
            oos_ic={"train": -0.025, "test": -0.015},
        )
    )
    assert d["direction"] == "reversed"
    assert d["should_reverse"] is True


def test_positive_ic_keeps_normal():
    d = _decide_backtest_direction(
        _ic(ic_mean=0.04, ic_tstat=4.0, ic_pvalue=0.0001, ir=0.5)
    )
    assert d["direction"] == "normal"
    assert d["should_reverse"] is False


def test_apply_reversed_flips_factor_clean_only():
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_clean": [1.5, -0.5],
            "factor_value": [9.0, 8.0],
        }
    )
    out = _apply_backtest_direction(df, {"direction": "reversed", "should_reverse": True})
    assert out["factor_clean"].to_list() == [-1.5, 0.5]
    # 原始语义列不变
    assert out["factor_value"].to_list() == [9.0, 8.0]


def test_apply_normal_is_noop():
    df = pl.DataFrame({"factor_clean": [1.0, 2.0]})
    out = _apply_backtest_direction(df, {"direction": "normal", "should_reverse": False})
    assert out["factor_clean"].to_list() == [1.0, 2.0]
    assert _apply_backtest_direction(df, None)["factor_clean"].to_list() == [1.0, 2.0]


def test_daily_single_wires_direction_helpers():
    """factor run 主路径必须 import 并调用与 report 相同的方向工具。"""
    import inspect

    from factorzen.pipelines import daily_single as mod

    src = inspect.getsource(mod._run)
    assert "_decide_backtest_direction" in src
    assert "_apply_backtest_direction" in src
    assert "backtest_direction=backtest_direction" in src
    # IC 用 clean_df；回测/换手/walk-forward 用 backtest_df
    assert "compute_rank_ic(clean_df" in src or "compute_rank_ic(\n        clean_df" in src
    assert "_run_backtest_strategies(\n            effective_config,\n            backtest_df," in src or (
        "backtest_df" in src and "_run_backtest_strategies" in src
    )
    assert "compute_turnover(backtest_df" in src


def test_meta_path_records_backtest_direction(tmp_path, monkeypatch):
    """daily_single 写入的 meta 形状应可被 report --reuse 读取。"""
    from factorzen.pipelines import _report_direction as direction_mod
    from factorzen.pipelines import _report_persistence as persist_mod

    monkeypatch.setattr(
        persist_mod, "daily_result_output_dir", lambda _name: tmp_path / "results"
    )
    # _meta_path 在 persist 模块解析 daily_result_output_dir
    path = persist_mod._meta_path("hf_resiliency", "20200101", "20201231")
    decision = direction_mod._decide_backtest_direction(
        _ic(ic_mean=-0.03, ic_tstat=-5.0, ic_pvalue=0.001)
    )
    path.write_text(
        json.dumps({"backtest_direction": decision}, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = direction_mod._load_backtest_direction("hf_resiliency", "20200101", "20201231")
    # _load_backtest_direction 也走 _meta_path → 需同样 patch
    monkeypatch.setattr(
        direction_mod, "_meta_path", lambda *a, **k: path
    )
    loaded = direction_mod._load_backtest_direction("hf_resiliency", "20200101", "20201231")
    assert loaded is not None
    assert loaded["direction"] == "reversed"
    assert loaded["should_reverse"] is True
