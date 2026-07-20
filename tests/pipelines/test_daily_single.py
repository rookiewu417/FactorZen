"""test_daily_single_helpers.py：daily_single.py 纯 helper 的离线单测：日期归一、产物存在性、默认 YAML 配置查找。
test_daily_single_metrics.py：daily_single._write_run_metrics 单测：sweep 读取的 IC + 主策略回测指标 JSON。
test_daily_single_set_flag.py：daily_single 的 --set 接线测试：经 --dry-run 路径（纯配置流，无需数据）验证生效。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from factorzen.pipelines import daily_single
from factorzen.pipelines import daily_single as ds
from factorzen.pipelines.daily_single import _write_run_metrics


# ==== 来自 test_daily_single_helpers.py ====
def test_date_expr_parses_dash_format():
    df = pl.DataFrame({"trade_date": ["2024-01-02"]}).with_columns(ds._date_expr("trade_date"))
    assert df["trade_date"].item() == date(2024, 1, 2)

def test_date_expr_parses_plain_format():
    df = pl.DataFrame({"trade_date": ["20240102"]}).with_columns(ds._date_expr("trade_date"))
    assert df["trade_date"].item() == date(2024, 1, 2)

def test_date_expr_invalid_becomes_null():
    df = pl.DataFrame({"trade_date": ["garbage"]}).with_columns(ds._date_expr("trade_date"))
    assert df["trade_date"].item() is None

# ── _ensure_date_column ─────────────────────────────────────

def test_ensure_date_passthrough_when_already_date():
    df = pl.DataFrame({"trade_date": [date(2024, 1, 2)]})
    assert ds._ensure_date_column(df, "trade_date")["trade_date"].dtype == pl.Date

def test_ensure_date_from_datetime():
    df = pl.DataFrame({"trade_date": [datetime(2024, 1, 2, 9, 30)]})
    out = ds._ensure_date_column(df, "trade_date")
    assert out["trade_date"].dtype == pl.Date
    assert out["trade_date"].item() == date(2024, 1, 2)

def test_ensure_date_from_utf8():
    df = pl.DataFrame({"trade_date": ["20240102"]})
    out = ds._ensure_date_column(df, "trade_date")
    assert out["trade_date"].item() == date(2024, 1, 2)

def test_ensure_date_missing_column_passthrough():
    df = pl.DataFrame({"x": [1]})
    assert ds._ensure_date_column(df, "trade_date").equals(df)

# ── _existing_run_outputs
# ── _existing_run_outputs ───────────────────────────────────

def test_existing_run_outputs_lists_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "daily_factor_output_dir", lambda f: tmp_path / "factors")
    monkeypatch.setattr(ds, "daily_result_output_dir", lambda f: tmp_path / "results")
    monkeypatch.setattr(ds, "daily_report_output_dir", lambda f: tmp_path / "reports")
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / "mom_20240101_20240131_ic.parquet").write_text("x")

    out = ds._existing_run_outputs("mom", "20240101", "20240131")
    assert set(out) == {"ic"}
    assert out["ic"].endswith("_ic.parquet")

def test_existing_run_outputs_empty_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "daily_factor_output_dir", lambda f: tmp_path / "factors")
    monkeypatch.setattr(ds, "daily_result_output_dir", lambda f: tmp_path / "results")
    monkeypatch.setattr(ds, "daily_report_output_dir", lambda f: tmp_path / "reports")
    assert ds._existing_run_outputs("mom", "20240101", "20240131") == {}

# ── _find_default_run_config_path ───────────────────────────

def _write_yaml(path, factor):
    path.write_text(f"factor: {factor}\nstart: '20230101'\nend: '20231231'\n", encoding="utf-8")

def test_find_config_missing_dir_returns_none(tmp_path):
    assert ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path) is None

def test_find_config_no_match_returns_none(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "other.yaml", "value")
    assert ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path) is None

def test_find_config_exact_stem_match(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "mom.yaml", "mom")
    _write_yaml(d / "another_mom.yaml", "mom")
    result = ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)
    assert result.stem == "mom"

def test_find_config_single_factor_prefix_preferred(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "single_factor_mom.yaml", "mom")
    _write_yaml(d / "batch_mom.yaml", "mom")
    result = ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)
    assert result.name == "single_factor_mom.yaml"

def test_find_config_ambiguous_raises(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_yaml(d / "alpha_mom.yaml", "mom")
    _write_yaml(d / "beta_mom.yaml", "mom")
    with pytest.raises(ValueError, match="多个默认配置"):
        ds._find_default_run_config_path("mom", "daily", configs_root=tmp_path)

# ==== 来自 test_daily_single_metrics.py ====
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

# ==== 来自 test_daily_single_set_flag.py ====
def _run_dry(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["daily_single", *argv, "--dry-run"])
    daily_single.main()

def test_set_override_no_config_bakes_topn(monkeypatch, capsys):
    """无 YAML + --set backtest.top_n=30 → 单一 topn_30 策略（不套用 4 策略默认套件）。"""
    _run_dry(
        monkeypatch,
        ["--factor", "f", "--start", "20230101", "--end", "20231231", "--set", "backtest.top_n=30"],
    )
    bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
    assert bt["top_n"] == 30
    assert len(bt["strategies"]) == 1
    assert bt["strategies"][0]["name"] == "topn_30"
    assert bt["strategies"][0]["params"] == {"top_n": 30}

def test_set_override_preprocessing(monkeypatch, capsys):
    _run_dry(
        monkeypatch,
        [
            "--factor",
            "f",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--set",
            "preprocessing.neutralize=true",
            "--set",
            "preprocessing.normalizer=rank_normal",
        ],
    )
    pp = json.loads(capsys.readouterr().out)["config"]["preprocessing"]
    assert pp["neutralize"] is True
    assert pp["normalizer"] == "rank_normal"

def test_no_set_no_config_keeps_default_suite(monkeypatch, capsys):
    """对照：无 --set 无 --config 时维持研究预设（quantile_ls_5 单策略）。"""
    _run_dry(monkeypatch, ["--factor", "f", "--start", "20230101", "--end", "20231231"])
    bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
    assert [s["name"] for s in bt["strategies"]] == ["quantile_ls_5"]
    assert bt["primary"] == "quantile_ls_5"

def test_set_override_with_config(monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "factor: f\nstart: '20230101'\nend: '20231231'\nbacktest:\n  top_n: 50\n",
        encoding="utf-8",
    )
    _run_dry(monkeypatch, ["--config", str(cfg), "--set", "backtest.top_n=20"])
    bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
    assert bt["top_n"] == 20
    assert bt["strategies"][0]["name"] == "topn_20"

def test_set_override_invalid_value_exits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily_single",
            "--factor",
            "f",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--set",
            "preprocessing.normalizer=bogus",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        daily_single.main()
    assert exc.value.code == 2

def test_set_override_malformed_exits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily_single",
            "--factor",
            "f",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--set",
            "backtest.top_n",  # 缺 '='
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        daily_single.main()
    assert exc.value.code == 2

