"""Tests for run_daily_single configuration merging."""

from __future__ import annotations

from argparse import Namespace
from datetime import date

import polars as pl
import pytest


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


def test_merge_run_config_args_without_yaml_fills_benchmark_and_seed():
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = _ns(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
    )

    merged = _merge_run_config_args(args, None)

    assert merged.universe == "csi500"
    assert merged.benchmark == "000905.SH"
    assert merged.seed == 42


def test_find_default_run_config_path_matches_factor_field(tmp_path):
    from factorzen.pipelines.daily_single import _find_default_run_config_path

    config_dir = tmp_path / "daily"
    config_dir.mkdir()
    (config_dir / "momentum_20d.yaml").write_text(
        "\n".join(
            [
                "factor: momentum_20d",
                "universe: csi500",
                'start: "20230101"',
                'end: "20241231"',
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "other.yaml").write_text(
        "\n".join(
            [
                "factor: reversal_5d",
                "universe: csi300",
                'start: "20230101"',
                'end: "20241231"',
            ]
        ),
        encoding="utf-8",
    )

    path = _find_default_run_config_path("momentum_20d", "daily", configs_root=tmp_path)

    assert path == config_dir / "momentum_20d.yaml"


def test_find_default_run_config_path_errors_on_multiple_matches(tmp_path):
    from factorzen.pipelines.daily_single import _find_default_run_config_path

    config_dir = tmp_path / "daily"
    config_dir.mkdir()
    for name in ("a.yaml", "b.yaml"):
        (config_dir / name).write_text(
            "\n".join(
                [
                    "factor: momentum_20d",
                    "universe: csi500",
                    'start: "20230101"',
                    'end: "20241231"',
                ]
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="找到多个默认配置"):
        _find_default_run_config_path("momentum_20d", "daily", configs_root=tmp_path)


def test_find_default_run_config_path_prefers_factor_named_config(tmp_path):
    from factorzen.pipelines.daily_single import _find_default_run_config_path

    config_dir = tmp_path / "daily"
    config_dir.mkdir()
    for name in ("momentum_20d.yaml", "momentum_20d_walk_forward.yaml"):
        (config_dir / name).write_text(
            "\n".join(
                [
                    "factor: momentum_20d",
                    "universe: csi500",
                    'start: "20230101"',
                    'end: "20241231"',
                ]
            ),
            encoding="utf-8",
        )

    path = _find_default_run_config_path("momentum_20d", "daily", configs_root=tmp_path)

    assert path == config_dir / "momentum_20d.yaml"



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
