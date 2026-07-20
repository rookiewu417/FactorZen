"""合并自: test_daily_single.py, test_daily_single_config.py
目标: test_daily_single.py

--- 来源 test_daily_single.py ---
test_daily_single_helpers.py：daily_single.py 纯 helper 的离线单测：日期归一、产物存在性、默认 YAML 配置查找。
test_daily_single_metrics.py：daily_single._write_run_metrics 单测：sweep 读取的 IC + 主策略回测指标 JSON。
test_daily_single_set_flag.py：daily_single 的 --set 接线测试：经 --dry-run 路径（纯配置流，无需数据）验证生效。

--- 来源 test_daily_single_config.py ---
Tests for run_daily_single configuration merging.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from factorzen.pipelines import daily_single
from factorzen.pipelines import daily_single as ds
from factorzen.pipelines.daily_single import _write_run_metrics


# ==== 来自 test_daily_single.py ====
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


# ==== 来自 test_daily_single_config.py ====
def _ns(**kw):
    base = dict(
        factor=None,
        start=None,
        end=None,
        universe=None,
        benchmark=None,
        seed=None,
    )
    base.update(kw)
    return Namespace(**base)

def test_build_forward_return_frame_prefers_adjusted_close():
    from factorzen.pipelines.daily_single import _build_forward_return_frame

    daily = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 5.0],
            "close_adj": [10.0, 10.0],
        }
    )

    ret_df = _build_forward_return_frame(daily)

    assert ret_df["ret"][1] == pytest.approx(0.0)
    assert ret_df["fwd_ret_1d"][0] == pytest.approx(0.0)

def test_build_forward_return_frame_falls_back_to_close_without_adjusted_close():
    from factorzen.pipelines.daily_single import _build_forward_return_frame

    daily = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 5.0],
        }
    )

    ret_df = _build_forward_return_frame(daily)

    assert ret_df["ret"][1] == pytest.approx(-0.5)
    assert ret_df["fwd_ret_1d"][0] == pytest.approx(-0.5)

def test_build_forward_return_frame_falls_back_per_stock_for_partial_adjusted_close():
    from factorzen.pipelines.daily_single import _build_forward_return_frame

    daily = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "ts_code": ["000002.SZ", "000001.SZ", "000002.SZ", "000001.SZ"],
            "close": [50.0, 5.0, 100.0, 10.0],
            "close_adj": [220.0, None, 200.0, 10.0],
        }
    )

    ret_df = _build_forward_return_frame(daily)
    stock_a = ret_df.filter(pl.col("ts_code") == "000001.SZ").sort("trade_date")
    stock_b = ret_df.filter(pl.col("ts_code") == "000002.SZ").sort("trade_date")

    assert stock_a["ret"][1] == pytest.approx(-0.5)
    assert stock_a["fwd_ret_1d"][0] == pytest.approx(-0.5)
    assert stock_b["ret"][1] == pytest.approx(0.1)
    assert stock_b["fwd_ret_1d"][0] == pytest.approx(0.1)

def test_run_backtest_strategies_runs_each_configured_strategy(monkeypatch):
    from types import SimpleNamespace

    from factorzen.config.research import RunConfig
    from factorzen.pipelines import daily_single as mod

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20230131",
        backtest={
            "primary": "topn_5",
            "strategies": [
                {"name": "topn_5", "type": "topn_long_only", "params": {"top_n": 5}},
                {
                    "name": "quantile_ls_4",
                    "type": "quantile_long_short",
                    "params": {"quantiles": 4},
                },
            ],
        },
    )
    calls = []

    def fake_run_strategy_backtest(strategy, *_args, **_kwargs):
        calls.append(strategy.name)
        return SimpleNamespace(strategy_name=strategy.name)

    monkeypatch.setattr(mod, "run_strategy_backtest", fake_run_strategy_backtest)
    monkeypatch.setattr(mod, "trim_backtest_to_first_trade", lambda result: result)

    primary, results = mod._run_backtest_strategies(
        cfg,
        pl.DataFrame(),
        pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date(2023, 1, 3)]}),
        factor_name="x",
        frequency="daily",
    )

    assert calls == ["topn_5", "quantile_ls_4"]
    assert primary.strategy_name == "topn_5"
    assert list(results) == ["topn_5", "quantile_ls_4"]

def test_run_backtest_strategies_passes_is_st_by_date_to_backtest(monkeypatch):
    """ST涨跌停容差接线：_run_backtest_strategies 应基于 daily 的
    codes/trade_dates 构建 is_st_by_date 并传给 run_strategy_backtest。
    """
    from types import SimpleNamespace

    from factorzen.config.research import RunConfig
    from factorzen.pipelines import daily_single as mod

    cfg = RunConfig(
        factor="x",
        start="20230101",
        end="20230131",
        backtest={
            "primary": "topn_5",
            "strategies": [
                {"name": "topn_5", "type": "topn_long_only", "params": {"top_n": 5}},
            ],
        },
    )
    captured: dict = {}

    def fake_run_strategy_backtest(strategy, *_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(strategy_name=strategy.name)

    sentinel = {date(2023, 1, 3): {"000001.SZ"}}
    monkeypatch.setattr(mod, "run_strategy_backtest", fake_run_strategy_backtest)
    monkeypatch.setattr(mod, "trim_backtest_to_first_trade", lambda result: result)
    monkeypatch.setattr(mod, "build_is_st_by_date", lambda codes, dates: sentinel)

    daily = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date(2023, 1, 3)]})
    mod._run_backtest_strategies(cfg, pl.DataFrame(), daily, factor_name="x", frequency="daily")

    assert captured.get("is_st_by_date") == sentinel, (
        "run_strategy_backtest 应收到由 build_is_st_by_date 构建的 is_st_by_date，"
        f"实际收到: {captured.get('is_st_by_date')!r}"
    )

def test_merge_run_config_args_uses_yaml_for_missing_cli_values():
    from factorzen.config.research import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = _ns()
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20241231",
        universe="csi500",
        benchmark=None,
        seed=42,
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.factor == "momentum_20d"
    assert merged.start == "20230101"
    assert merged.end == "20241231"
    assert merged.universe == "csi500"
    assert merged.benchmark == "000905.SH"
    assert merged.seed == 42

def test_merge_run_config_args_keeps_explicit_cli_values():
    from factorzen.config.research import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = _ns(
        factor="reversal_5d",
        start="20240101",
        end="20241231",
        universe="csi300",
        benchmark=None,
        seed=7,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20231231",
        universe="csi500",
        benchmark="000905.SH",
        seed=42,
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.factor == "reversal_5d"
    assert merged.start == "20240101"
    assert merged.end == "20241231"
    assert merged.universe == "csi300"
    assert merged.benchmark == "000905.SH"
    assert merged.seed == 7

def test_merge_run_config_args_keeps_explicit_cli_benchmark():
    from factorzen.config.research import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = _ns(benchmark="000852.SH")
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20231231",
        universe="csi500",
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.benchmark == "000852.SH"

def test_dry_run_payload_includes_effective_config_and_output_dir():
    from factorzen.config.research import RunConfig
    from factorzen.pipelines.daily_single import _build_dry_run_payload

    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20231231",
        universe="csi500",
        benchmark="000905.SH",
        backtest={"top_n": 25},
        walk_forward={"n_trials": 3},
    )

    payload = _build_dry_run_payload(cfg)

    assert payload["config"]["benchmark"] == "000905.SH"
    assert payload["config"]["backtest"]["top_n"] == 25
    assert payload["config"]["walk_forward"]["n_trials"] == 3
    assert payload["output_dir"].endswith("workspace/factor_evaluations/<run_id>")
    assert "execution" not in payload
    for banned in ("ic_method", "neutralized_ic", "event_study", "llm_explain", "llm_refresh"):
        assert banned not in payload["config"]

def test_merge_run_config_args_and_dry_run_drop_deep_eval_keys():
    """防回归：合并后的 namespace / dry-run payload 不再含深度评估键。"""
    from factorzen.pipelines.daily_single import (
        _build_dry_run_payload,
        _effective_run_config,
        _merge_run_config_args,
    )

    args = _ns(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
    )
    # 旧 YAML 字段应被 ignore，不应出现在 args 上
    merged = _merge_run_config_args(args, None)
    for banned in ("ic_method", "neutralized_ic", "event_study", "llm_explain", "llm_refresh", "all"):
        assert not hasattr(merged, banned) or getattr(merged, banned, None) is None
        # 更严格：合并逻辑不应写入这些属性
        assert banned not in vars(merged)

    cfg = _effective_run_config(merged, None)
    dumped = cfg.model_dump()
    for banned in ("ic_method", "neutralized_ic", "event_study"):
        assert banned not in dumped

    payload = _build_dry_run_payload(cfg, args=merged)
    assert "execution" not in payload
    for banned in ("llm_explain", "llm_refresh", "ic_method", "neutralized_ic", "event_study"):
        assert banned not in payload.get("execution", {})
        assert banned not in payload["config"]

def test_effective_run_config_without_yaml_uses_quantile_ls_5():
    from factorzen.pipelines.daily_single import _effective_run_config, _merge_run_config_args

    args = _ns(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
    )

    merged = _merge_run_config_args(args, None)
    cfg = _effective_run_config(merged, None)

    assert cfg.universe == "csi500"
    assert cfg.benchmark == "000905.SH"
    assert cfg.seed == 42
    assert cfg.preprocessing.neutralize is True
    assert cfg.preprocessing.neutralize_by == "industry+size"
    assert cfg.backtest.primary == "quantile_ls_5"
    assert [spec.name for spec in cfg.backtest.strategy_specs] == ["quantile_ls_5"]
    assert cfg.backtest.strategy_specs[0].type == "quantile_long_short"
    assert cfg.backtest.strategy_specs[0].params == {"quantiles": 5}

def test_preprocess_with_industry_neutralization_uses_universe_industry():
    from factorzen.config.research import RunConfig
    from factorzen.pipelines.daily_single import _preprocess_factor

    rows = []
    universe_rows = []
    for i in range(40):
        code = f"{i:06d}.SZ"
        industry = "银行" if i < 20 else "医药"
        value = 1.0 if industry == "银行" else -1.0
        rows.append({"trade_date": "2024-01-02", "ts_code": code, "factor_value": value})
        universe_rows.append({"ts_code": code, "industry": industry})

    cfg = RunConfig(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        preprocessing={"neutralize": True, "neutralize_by": "industry"},
    )

    clean = _preprocess_factor(
        pl.DataFrame(rows),
        cfg,
        universe=pl.DataFrame(universe_rows),
        daily_basic=None,
    )

    by_industry = (
        clean.join(pl.DataFrame(universe_rows), on="ts_code")
        .group_by("industry")
        .agg(pl.col("factor_clean").mean().alias("mean_factor"))
    )
    assert by_industry["mean_factor"].abs().max() < 1e-10

def test_load_daily_basic_for_neutralization_reads_ensured_cache(monkeypatch):
    import factorzen.pipelines.daily_single as run_mod

    expected = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "ts_code": ["000001.SZ"],
            "total_mv": [100.0],
        }
    )
    calls: list[tuple[str, str, str]] = []

    class _LazyFrameStub:
        def collect(self):
            return expected

    def fake_load_parquet(data_type: str, *, start: str, end: str):
        calls.append((data_type, start, end))
        return _LazyFrameStub()

    monkeypatch.setattr(run_mod, "load_parquet", fake_load_parquet)

    result = run_mod._load_daily_basic_for_neutralization("20240102", "20240103")

    assert result.equals(expected)
    assert calls == [("daily_basic", "20240102", "20240103")]

def test_run_ensures_required_data_before_loading_universe(monkeypatch):
    import factorzen.pipelines.daily_single as run_mod
    from factorzen.config.research import RunConfig

    calls: list[str] = []

    class DummyFactor:
        name = "dummy_factor"
        description = "dummy"
        required_data = ["daily"]
        lookback_days = 20

    def fake_ensure_data_for_daily_run(**kwargs):
        calls.append("ensure")
        assert kwargs["required_data"] == ["daily"]
        assert kwargs["start"] == "20240102"
        assert kwargs["end"] == "20240103"

    def fake_load_pit_membership(*args, **kwargs):
        calls.append("universe")
        raise RuntimeError("stop after data ensure")

    monkeypatch.setattr(run_mod, "get_factor", lambda name: DummyFactor)
    monkeypatch.setattr(run_mod, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    monkeypatch.setattr(run_mod, "ensure_data_for_daily_run", fake_ensure_data_for_daily_run)
    monkeypatch.setattr(run_mod, "load_pit_membership", fake_load_pit_membership)

    args = Namespace(
        factor="dummy_factor",
        start="20240102",
        end="20240103",
        universe="csi300",
        frequency="daily",
        benchmark=None,
        seed=None,
    )

    with pytest.raises(RuntimeError, match="stop after data ensure"):
        run_mod._run(args, RunConfig(factor="dummy_factor", start="20240102", end="20240103"))

    assert calls == ["ensure", "universe"]

