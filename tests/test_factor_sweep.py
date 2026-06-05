"""factor_sweep 纯逻辑单测：网格展开、注入式编排排序、表格/CSV 渲染、失败容错。"""

from __future__ import annotations

import math

import pytest

from factorzen.pipelines.factor_sweep import (
    expand_grid,
    format_sweep_csv,
    format_sweep_table,
    run_sweep,
)


def test_expand_grid_cartesian_product():
    combos = expand_grid(["backtest.top_n=30,50", "preprocessing.normalizer=zscore,rank_normal"])
    assert combos == [
        ["backtest.top_n=30", "preprocessing.normalizer=zscore"],
        ["backtest.top_n=30", "preprocessing.normalizer=rank_normal"],
        ["backtest.top_n=50", "preprocessing.normalizer=zscore"],
        ["backtest.top_n=50", "preprocessing.normalizer=rank_normal"],
    ]


def test_expand_grid_single_dim():
    assert expand_grid(["backtest.top_n=30,50,100"]) == [
        ["backtest.top_n=30"],
        ["backtest.top_n=50"],
        ["backtest.top_n=100"],
    ]


def test_expand_grid_empty():
    assert expand_grid([]) == []


def test_expand_grid_strips_whitespace():
    assert expand_grid(["backtest.top_n = 30, 50 "]) == [["backtest.top_n=30"], ["backtest.top_n=50"]]


@pytest.mark.parametrize("token", ["backtest.top_n", "=30,50", "backtest.top_n="])
def test_expand_grid_rejects_bad_tokens(token):
    with pytest.raises(ValueError):
        expand_grid([token])


def test_run_sweep_collects_and_sorts_by_ir():
    metrics = {
        "backtest.top_n=30": {"ic_mean": 0.04, "ir": 0.18},
        "backtest.top_n=50": {"ic_mean": 0.05, "ir": 0.12},
        "backtest.top_n=100": {"ic_mean": 0.03, "ir": 0.20},
    }

    def fake_runner(overrides):
        return metrics[overrides[0]]

    rows = run_sweep(["backtest.top_n=30,50,100"], fake_runner, sort_by="ir")
    assert [r["overrides"][0] for r in rows] == [
        "backtest.top_n=100",  # ir 0.20
        "backtest.top_n=30",  # ir 0.18
        "backtest.top_n=50",  # ir 0.12
    ]


def test_run_sweep_sort_by_backtest_metric():
    """top_n 维度只影响回测：按 sharpe 排序才有区分度。"""
    metrics = {
        "backtest.top_n=20": {"ir": 0.1, "sharpe": -1.15},
        "backtest.top_n=50": {"ir": 0.1, "sharpe": -1.21},
    }

    def fake_runner(overrides):
        return metrics[overrides[0]]

    rows = run_sweep(["backtest.top_n=20,50"], fake_runner, sort_by="sharpe")
    assert [r["overrides"][0] for r in rows] == ["backtest.top_n=20", "backtest.top_n=50"]


def test_run_sweep_applies_extra_overrides():
    seen = []

    def fake_runner(overrides):
        seen.append(overrides)
        return {"ir": 1.0}

    run_sweep(["backtest.top_n=30"], fake_runner, extra_overrides=["preprocessing.neutralize=true"])
    assert seen == [["preprocessing.neutralize=true", "backtest.top_n=30"]]


def test_run_sweep_tolerates_runner_failure():
    def flaky_runner(overrides):
        if overrides[0].endswith("50"):
            raise RuntimeError("数据不足")
        return {"ir": 0.3}

    rows = run_sweep(["backtest.top_n=30,50"], flaky_runner, sort_by="ir")
    # 成功组排前，失败组（-inf）排后并带 error
    assert rows[0]["overrides"] == ["backtest.top_n=30"]
    assert rows[0]["ir"] == 0.3
    assert rows[1]["error"] == "数据不足"
    assert "ir" not in rows[1]


def test_run_sweep_nan_sorts_last():
    def runner(overrides):
        return {"ir": float("nan")} if overrides[0].endswith("50") else {"ir": 0.1}

    rows = run_sweep(["backtest.top_n=50,30"], runner, sort_by="ir")
    assert rows[0]["overrides"] == ["backtest.top_n=30"]
    assert math.isnan(rows[1]["ir"])


def test_format_sweep_table_has_headers_and_rows():
    rows = [
        {
            "overrides": ["backtest.top_n=30"],
            "ic_mean": 0.04,
            "ir": 0.18,
            "sharpe": -1.15,
            "ann_ret": -0.022,
            "avg_turnover": 0.51,
            "n": 24,
        },
    ]
    table = format_sweep_table(rows)
    assert "top_n" in table  # 短名表头
    assert "ir" in table
    assert "sharpe" in table  # 回测指标列
    assert "0.1800" in table  # ir 4 位小数
    assert "-1.1500" in table  # sharpe
    assert "30" in table


def test_format_sweep_table_empty():
    assert format_sweep_table([]) == "(空 sweep)"


def test_format_sweep_table_shows_error():
    rows = [{"overrides": ["backtest.top_n=50"], "error": "数据不足"}]
    assert "数据不足" in format_sweep_table(rows)


def test_format_sweep_csv_roundtrip():
    rows = [
        {
            "overrides": ["backtest.top_n=30"],
            "ic_mean": 0.04,
            "ir": 0.18,
            "sharpe": -1.15,
            "ann_ret": -0.022,
            "avg_turnover": 0.51,
            "n": 24,
        },
    ]
    csv_text = format_sweep_csv(rows)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("backtest.top_n,")
    assert "sharpe" in lines[0]
    assert "0.18" in lines[1]
    assert "30" in lines[1]


def test_format_sweep_csv_empty():
    assert format_sweep_csv([]) == ""
