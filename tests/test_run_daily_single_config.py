"""Tests for run_daily_single configuration merging."""

from __future__ import annotations

from argparse import Namespace
from datetime import date

import polars as pl
import pytest


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


def test_build_advanced_results_includes_sector_and_size_breakdowns():
    from factorzen.pipelines.daily_single import _build_advanced_results

    rows = []
    ret_rows = []
    universe_rows = []
    basic_rows = []
    for i in range(12):
        code = f"{i:06d}.SZ"
        industry = "Bank" if i < 6 else "Tech"
        universe_rows.append({"ts_code": code, "industry": industry})
        for date_str, offset in [("2024-01-02", 0), ("2024-01-03", 1)]:
            rows.append(
                {
                    "trade_date": date_str,
                    "ts_code": code,
                    "factor_clean": float(i + offset),
                }
            )
            ret_rows.append(
                {
                    "trade_date": date_str,
                    "ts_code": code,
                    "fwd_ret_1d": float(i - offset) / 100.0,
                }
            )
            basic_rows.append(
                {
                    "trade_date": date_str,
                    "ts_code": code,
                    "total_mv": float(100 + i),
                }
            )

    advanced = _build_advanced_results(
        pl.DataFrame(rows),
        pl.DataFrame(ret_rows),
        universe=pl.DataFrame(universe_rows),
        daily_basic=pl.DataFrame(basic_rows),
    )

    assert advanced is not None
    assert advanced["sector"].sector_ic_df.height > 0
    assert advanced["size"].buckets


def test_build_attribution_result_uses_long_book_for_brinson():
    from types import SimpleNamespace

    from factorzen.daily.evaluation.attribution import BrinsonResult
    from factorzen.pipelines.daily_single import _build_attribution_result

    positions = pl.DataFrame(
        [
            {"trade_date": "2024-01-03", "ts_code": "000001.SZ", "weight": 0.6},
            {"trade_date": "2024-01-03", "ts_code": "000002.SZ", "weight": 0.4},
            {"trade_date": "2024-01-03", "ts_code": "000003.SZ", "weight": -0.5},
        ]
    )
    daily = pl.DataFrame(
        [
            {"trade_date": "2024-01-02", "ts_code": "000001.SZ", "close": 10.0},
            {"trade_date": "2024-01-02", "ts_code": "000002.SZ", "close": 20.0},
            {"trade_date": "2024-01-02", "ts_code": "000003.SZ", "close": 30.0},
            {"trade_date": "2024-01-03", "ts_code": "000001.SZ", "close": 11.0},
            {"trade_date": "2024-01-03", "ts_code": "000002.SZ", "close": 19.0},
            {"trade_date": "2024-01-03", "ts_code": "000003.SZ", "close": 33.0},
        ]
    )
    universe = pl.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry": "Bank"},
            {"ts_code": "000002.SZ", "industry": "Tech"},
            {"ts_code": "000003.SZ", "industry": "Tech"},
        ]
    )

    attribution = _build_attribution_result(SimpleNamespace(positions=positions), daily, universe)

    assert attribution is not None
    assert isinstance(attribution["brinson"], BrinsonResult)
    assert set(attribution["brinson"].sector_df["sector"].to_list()) == {"Bank", "Tech"}


def test_run_backtest_strategies_runs_each_configured_strategy(monkeypatch):
    from types import SimpleNamespace

    from factorzen.core.config_loader import RunConfig
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

    from factorzen.core.config_loader import RunConfig
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
    from factorzen.core.config_loader import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor=None,
        start=None,
        end=None,
        universe=None,
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=False,
        llm_explain=False,
        llm_refresh=False,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20241231",
        universe="csi500",
        benchmark=None,
        seed=42,
        ic_method="both",
        neutralized_ic=True,
        event_study=True,
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.factor == "momentum_20d"
    assert merged.start == "20230101"
    assert merged.end == "20241231"
    assert merged.universe == "csi500"
    assert merged.benchmark == "000905.SH"
    assert merged.seed == 42
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True


def test_merge_run_config_args_keeps_explicit_cli_values():
    from factorzen.core.config_loader import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor="reversal_5d",
        start="20240101",
        end="20241231",
        universe="csi300",
        benchmark=None,
        seed=7,
        ic_method="pearson",
        neutralized_ic=False,
        event_study=False,
        all=False,
        llm_explain=False,
        llm_refresh=False,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20231231",
        universe="csi500",
        benchmark="000905.SH",
        seed=42,
        ic_method="both",
        neutralized_ic=True,
        event_study=True,
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.factor == "reversal_5d"
    assert merged.start == "20240101"
    assert merged.end == "20241231"
    assert merged.universe == "csi300"
    assert merged.benchmark == "000905.SH"
    assert merged.seed == 7
    assert merged.ic_method == "pearson"
    assert merged.neutralized_ic is False
    assert merged.event_study is False


def test_merge_run_config_args_keeps_explicit_cli_benchmark():
    from factorzen.core.config_loader import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor=None,
        start=None,
        end=None,
        universe=None,
        benchmark="000852.SH",
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=False,
        llm_explain=False,
        llm_refresh=False,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20231231",
        universe="csi500",
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.benchmark == "000852.SH"


def test_merge_run_config_args_all_enables_single_factor_defaults():
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe=None,
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=True,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = _merge_run_config_args(args, None)

    assert merged.universe == "csi500"
    assert merged.benchmark == "000905.SH"
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True
    assert merged.llm_explain is True
    assert merged.llm_refresh is False


def test_merge_run_config_args_all_uses_universe_matched_benchmark():
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe="csi500",
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=True,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = _merge_run_config_args(args, None)

    assert merged.benchmark == "000905.SH"


def test_dry_run_payload_includes_effective_config_and_output_dir():
    from factorzen.core.config_loader import RunConfig
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


def test_dry_run_payload_includes_execution_options():
    from factorzen.core.config_loader import RunConfig
    from factorzen.pipelines.daily_single import _build_dry_run_payload

    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20231231",
    )
    args = Namespace(llm_explain=True, llm_refresh=False)

    payload = _build_dry_run_payload(cfg, args=args)

    assert payload["execution"]["llm_explain"] is True
    assert payload["execution"]["llm_refresh"] is False


def test_merge_run_config_args_all_overrides_yaml_defaults():
    from factorzen.core.config_loader import RunConfig
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor=None,
        start=None,
        end=None,
        universe=None,
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=True,
        llm_explain=False,
        llm_refresh=False,
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20241231",
        universe="csi500",
        benchmark="000300.SH",
        ic_method="rank",
        neutralized_ic=False,
        event_study=False,
    )

    merged = _merge_run_config_args(args, cfg)

    assert merged.benchmark == "000905.SH"
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True


def test_effective_run_config_without_yaml_uses_default_strategy_suite():
    from factorzen.pipelines.daily_single import _effective_run_config, _merge_run_config_args

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe=None,
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=False,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = _merge_run_config_args(args, None)
    cfg = _effective_run_config(merged, None)

    assert cfg.universe == "csi500"
    assert cfg.benchmark == "000905.SH"
    assert cfg.seed == 42
    assert cfg.preprocessing.neutralize is True
    assert cfg.preprocessing.neutralize_by == "industry+size"
    assert cfg.ic_method == "both"
    assert cfg.neutralized_ic is True
    assert cfg.event_study is True
    assert [spec.name for spec in cfg.backtest.strategy_specs] == [
        "topn_50",
        "quantile_ls_5",
        "factor_weighted_ls",
        "optimizer_mv_long_only",
    ]


def test_merge_run_config_args_without_yaml_enables_comprehensive_defaults():
    from factorzen.pipelines.daily_single import _merge_run_config_args

    args = Namespace(
        factor="momentum_20d",
        start="20240101",
        end="20240131",
        universe=None,
        benchmark=None,
        seed=None,
        ic_method=None,
        neutralized_ic=None,
        event_study=None,
        all=False,
        llm_explain=False,
        llm_refresh=False,
    )

    merged = _merge_run_config_args(args, None)

    assert merged.universe == "csi500"
    assert merged.benchmark == "000905.SH"
    assert merged.seed == 42
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True
    assert merged.llm_explain is True


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


def test_build_neutralized_ic_frame_includes_industry_and_market_cap():
    from factorzen.pipelines.daily_single import _build_neutralized_ic_frame

    clean_df = pl.DataFrame(
        [
            {"trade_date": "2024-01-02", "ts_code": "000001.SZ", "factor_clean": 1.0},
            {"trade_date": "2024-01-02", "ts_code": "000002.SZ", "factor_clean": -1.0},
        ]
    )
    ret_df = pl.DataFrame(
        [
            {"trade_date": "2024-01-02", "ts_code": "000001.SZ", "fwd_ret_1d": 0.01},
            {"trade_date": "2024-01-02", "ts_code": "000002.SZ", "fwd_ret_1d": -0.01},
        ]
    )
    universe = pl.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry": "Bank"},
            {"ts_code": "000002.SZ", "industry": "Tech"},
        ]
    )
    daily_basic = pl.DataFrame(
        [
            {"trade_date": "2024-01-02", "ts_code": "000001.SZ", "total_mv": 100.0},
            {"trade_date": "2024-01-02", "ts_code": "000002.SZ", "total_mv": 200.0},
        ]
    )

    frame = _build_neutralized_ic_frame(
        clean_df,
        ret_df,
        universe=universe,
        daily_basic=daily_basic,
    )

    assert set(["factor_clean", "ret_1d", "industry", "total_mv"]).issubset(frame.columns)
    assert frame.select(pl.col("industry").null_count()).item() == 0
    assert frame.select(pl.col("total_mv").null_count()).item() == 0


def test_preprocess_with_industry_neutralization_uses_universe_industry():
    from factorzen.core.config_loader import RunConfig
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
    from factorzen.core.config_loader import RunConfig

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

    def fake_get_universe(*args, **kwargs):
        calls.append("universe")
        raise RuntimeError("stop after data ensure")

    monkeypatch.setattr(run_mod, "get_factor", lambda name: DummyFactor)
    monkeypatch.setattr(run_mod, "get_trade_dates", lambda start, end: [date(2024, 1, 2)])
    monkeypatch.setattr(run_mod, "ensure_data_for_daily_run", fake_ensure_data_for_daily_run)
    monkeypatch.setattr(run_mod, "get_universe", fake_get_universe)

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
