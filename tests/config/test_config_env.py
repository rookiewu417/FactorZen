"""
test_config_loader.py：common.config_loader 模块测试
test_config_overrides.py：config_loader 的 --set 覆盖层单测
test_dotenv_loader.py：tushare_config._load_dotenv 的单测(BOM/CRLF/引号/注释)
test_tushare_config.py：tushare 配置相关测试
test_tushare_lake_downloader.py：tushare lake downloader 测试
"""

from __future__ import annotations

import json

import pytest

from factorzen.config import tushare_config
from factorzen.config.research import (
    apply_overrides,
    build_run_config_from_dict,
    load_run_config,
)
from factorzen.config.settings import ROOT
from factorzen.config.tushare_config import _load_dotenv
from tools import download_tushare_lake as dl


# ==== 来自 test_config_loader.py ====
def test_run_config_load_validate_suite(tmp_path):
    """test_load_valid_config；test_load_config_with_seed；test_invalid_outlier_method；test_default_preprocessing；test_walk_forward_is_disabled_by_default_and_can_be_enabled"""
    # -- 原 test_load_valid_config --
    def _section_0_test_load_valid_config(tmp_path):
        from factorzen.config.research import load_run_config

        yaml_content = "factor: momentum_20d\nstart: '20230101'\nend: '20241231'\n"
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        config = load_run_config(p)
        assert config.factor == "momentum_20d"
        assert config.benchmark is None
        assert config.seed is None  # optional default

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_load_valid_config(_tp0)

    # -- 原 test_load_config_with_seed --
    def _section_1_test_load_config_with_seed(tmp_path):
        from factorzen.config.research import load_run_config

        yaml_content = "factor: reversal\nstart: '20230101'\nend: '20241231'\nseed: 99\n"
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        config = load_run_config(p)
        assert config.seed == 99

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_load_config_with_seed(_tp1)

    # -- 原 test_invalid_outlier_method --
    def _section_2_test_invalid_outlier_method(tmp_path):
        import pydantic

        from factorzen.config.research import load_run_config

        yaml_content = (
            "factor: x\nstart: '20230101'\nend: '20241231'\npreprocessing:\n  outlier: invalid_method\n"
        )
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        with pytest.raises(pydantic.ValidationError):
            load_run_config(p)

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_invalid_outlier_method(_tp2)

    # -- 原 test_default_preprocessing --
    def _section_3_test_default_preprocessing():
        from factorzen.config.research import RunConfig

        cfg = RunConfig(factor="x", start="20230101", end="20241231")
        assert cfg.preprocessing.outlier == "mad"
        assert cfg.preprocessing.normalizer == "zscore"

    _section_3_test_default_preprocessing()

    # -- 原 test_walk_forward_is_disabled_by_default_and_can_be_enabled --
    def _section_4_test_walk_forward_is_disabled_by_default_and_can_be_enabled():
        from factorzen.config.research import RunConfig

        default_cfg = RunConfig(factor="x", start="20230101", end="20241231")
        enabled_cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            walk_forward={"enabled": True},
        )

        assert default_cfg.walk_forward.enabled is False
        assert enabled_cfg.walk_forward.enabled is True

    _section_4_test_walk_forward_is_disabled_by_default_and_can_be_enabled()


def test_build_runtime_from_run_config_suite():
    """test_build_preprocessing_pipeline_from_run_config；test_build_runtime_backtest_config_from_run_config；test_build_cost_model_from_run_config；test_default_benchmark_is_derived_from_universe；test_build_backtest_strategy_uses_top_n；test_backtest_config_supports_multiple_named_strategies；test_strategy_spec_can_override_runtime_backtest_settings；test_legacy_backtest_config_exposes_default_strategy_spec；test_build_backtest_strategies_returns_named_runtime_strategies；test_build_top_n_candidate_params_uses_n_trials；test_build_top_n_candidate_params_skips_weights_above_cap；test_default_daily_research_config_top_n_override_updates_primary_strategy"""
    # -- 原 test_build_preprocessing_pipeline_from_run_config --
    def _section_0_test_build_preprocessing_pipeline_from_run_config():
        from factorzen.config.research import RunConfig
        from factorzen.daily.runtime import build_preprocessing_pipeline

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            preprocessing={
                "outlier": "winsorize",
                "normalizer": "rank_uniform",
                "neutralize": True,
            },
        )

        pipeline = build_preprocessing_pipeline(cfg)

        assert pipeline.outlier_method == "winsorize"
        assert pipeline.normalizer_method == "rank_uniform"
        assert pipeline.neutralize is True

    _section_0_test_build_preprocessing_pipeline_from_run_config()

    # -- 原 test_build_runtime_backtest_config_from_run_config --
    def _section_1_test_build_runtime_backtest_config_from_run_config():
        from factorzen.config.research import RunConfig
        from factorzen.daily.runtime import build_runtime_backtest_config

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={
                "quantiles": 7,
                "max_abs_weight": 0.2,
                "rebalance_threshold": 0.15,
            },
        )

        runtime = build_runtime_backtest_config(cfg, factor_col="factor_clean", frequency="weekly")

        assert runtime.factor_col == "factor_clean"
        assert runtime.frequency == "weekly"
        assert runtime.max_abs_weight == 0.2
        assert runtime.rebalance_threshold == 0.15

    _section_1_test_build_runtime_backtest_config_from_run_config()

    # -- 原 test_build_cost_model_from_run_config --
    def _section_2_test_build_cost_model_from_run_config():
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.cost_models import (
            LinearCostModel,
            SquareRootImpactCostModel,
        )
        from factorzen.daily.runtime import build_cost_model

        linear_cfg = RunConfig(factor="x", start="20230101", end="20241231")
        assert isinstance(build_cost_model(linear_cfg), LinearCostModel)

        impact_cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={"cost_model": "square_root_impact"},
        )
        assert isinstance(build_cost_model(impact_cfg), SquareRootImpactCostModel)

    _section_2_test_build_cost_model_from_run_config()

    # -- 原 test_default_benchmark_is_derived_from_universe --
    def _section_3_test_default_benchmark_is_derived_from_universe():
        from factorzen.config.research import default_benchmark_for_universe

        assert default_benchmark_for_universe("csi300") == "000300.SH"
        assert default_benchmark_for_universe("csi500") == "000905.SH"
        assert default_benchmark_for_universe("csi800") == "000906.SH"
        assert default_benchmark_for_universe("unknown") == "000300.SH"

    _section_3_test_default_benchmark_is_derived_from_universe()

    # -- 原 test_build_backtest_strategy_uses_top_n --
    def _section_4_test_build_backtest_strategy_uses_top_n():
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.backtest import TopNLongOnlyStrategy
        from factorzen.daily.runtime import build_backtest_strategy

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={"top_n": 17},
        )

        strategy = build_backtest_strategy(cfg)

        assert isinstance(strategy, TopNLongOnlyStrategy)
        assert strategy.n == 17

    _section_4_test_build_backtest_strategy_uses_top_n()

    # -- 原 test_backtest_config_supports_multiple_named_strategies --
    def _section_5_test_backtest_config_supports_multiple_named_strategies():
        from factorzen.config.research import RunConfig

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={
                "primary": "topn_50",
                "strategies": [
                    {"name": "topn_50", "type": "topn_long_only", "params": {"top_n": 50}},
                    {
                        "name": "quantile_ls_5",
                        "type": "quantile_long_short",
                        "params": {"quantiles": 5},
                    },
                ],
            },
        )

        assert cfg.backtest.primary == "topn_50"
        assert [strategy.name for strategy in cfg.backtest.strategies] == [
            "topn_50",
            "quantile_ls_5",
        ]
        assert cfg.backtest.strategies[1].params == {"quantiles": 5}

    _section_5_test_backtest_config_supports_multiple_named_strategies()

    # -- 原 test_strategy_spec_can_override_runtime_backtest_settings --
    def _section_6_test_strategy_spec_can_override_runtime_backtest_settings():
        from factorzen.config.research import RunConfig
        from factorzen.daily.runtime import build_runtime_backtest_config

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={
                "strategies": [
                    {
                        "name": "topn_20_tight",
                        "type": "topn_long_only",
                        "params": {"top_n": 20},
                        "max_abs_weight": 0.03,
                        "rebalance_threshold": 0.2,
                        "cost_model": "square_root_impact",
                    }
                ],
            },
        )

        spec = cfg.backtest.strategy_specs[0]
        assert spec.max_abs_weight == 0.03
        assert spec.rebalance_threshold == 0.2
        assert spec.cost_model == "square_root_impact"

        runtime = build_runtime_backtest_config(cfg, strategy_spec=spec)
        assert runtime.strategy_type == "topn_long_only"
        assert runtime.strategy_params == {"top_n": 20}

    _section_6_test_strategy_spec_can_override_runtime_backtest_settings()

    # -- 原 test_legacy_backtest_config_exposes_default_strategy_spec --
    def _section_7_test_legacy_backtest_config_exposes_default_strategy_spec():
        from factorzen.config.research import RunConfig

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={"top_n": 17, "quantiles": 7},
        )

        assert cfg.backtest.strategy_specs[0].name == "topn_17"
        assert cfg.backtest.strategy_specs[0].type == "topn_long_only"
        assert cfg.backtest.strategy_specs[0].params == {"top_n": 17}
        assert cfg.backtest.primary == "topn_17"

    _section_7_test_legacy_backtest_config_exposes_default_strategy_spec()

    # -- 原 test_build_backtest_strategies_returns_named_runtime_strategies --
    def _section_8_test_build_backtest_strategies_returns_named_runtime_strategies():
        from factorzen.config.research import RunConfig
        from factorzen.daily.evaluation.backtest import (
            QuantileLongShortStrategy,
            TopNLongOnlyStrategy,
        )
        from factorzen.daily.runtime import build_backtest_strategies

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={
                "strategies": [
                    {"name": "topn_12", "type": "topn_long_only", "params": {"top_n": 12}},
                    {"name": "ls_4", "type": "quantile_long_short", "params": {"quantiles": 4}},
                ],
            },
        )

        strategies = build_backtest_strategies(cfg)

        assert list(strategies) == ["topn_12", "ls_4"]
        assert isinstance(strategies["topn_12"], TopNLongOnlyStrategy)
        assert strategies["topn_12"].n == 12
        assert isinstance(strategies["ls_4"], QuantileLongShortStrategy)
        assert strategies["ls_4"].n_groups == 4

    _section_8_test_build_backtest_strategies_returns_named_runtime_strategies()

    # -- 原 test_build_top_n_candidate_params_uses_n_trials --
    def _section_9_test_build_top_n_candidate_params_uses_n_trials():
        from factorzen.config.research import RunConfig, build_top_n_candidate_params

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={"top_n": 10},
            walk_forward={"n_trials": 4},
        )

        assert build_top_n_candidate_params(cfg) == [
            {"top_n": 10},
        ]

    _section_9_test_build_top_n_candidate_params_uses_n_trials()

    # -- 原 test_build_top_n_candidate_params_skips_weights_above_cap --
    def _section_10_test_build_top_n_candidate_params_skips_weights_above_cap():
        from factorzen.config.research import RunConfig, build_top_n_candidate_params

        cfg = RunConfig(
            factor="x",
            start="20230101",
            end="20241231",
            backtest={"top_n": 50, "max_abs_weight": 0.1},
            walk_forward={"n_trials": 5},
        )

        assert build_top_n_candidate_params(cfg) == [
            {"top_n": 10},
            {"top_n": 20},
            {"top_n": 30},
            {"top_n": 40},
            {"top_n": 50},
        ]

    _section_10_test_build_top_n_candidate_params_skips_weights_above_cap()

    # -- 原 test_default_daily_research_config_top_n_override_updates_primary_strategy --
    def _section_11_test_default_daily_research_config_top_n_override_updates_primary_strategy():
        from factorzen.config.research import (
            build_default_daily_research_config,
            build_run_config_from_dict,
        )

        base = build_default_daily_research_config(
            factor="x",
            start="20230101",
            end="20241231",
        ).model_dump()

        cfg = build_run_config_from_dict(base, overrides=["backtest.top_n=30"])

        # 默认主策略为 quantile_ls_5，top_n 覆盖不会改写 quantile 规格
        assert cfg.backtest.primary == "quantile_ls_5"
        assert cfg.backtest.top_n == 30
        assert cfg.backtest.strategy_specs[0].name == "quantile_ls_5"
        assert cfg.backtest.strategy_specs[0].params == {"quantiles": 5}

    _section_11_test_default_daily_research_config_top_n_override_updates_primary_strategy()


# ==== 来自 test_config_overrides.py ====
def test_apply_overrides_suite():
    """test_apply_overrides_coerces_types；test_apply_overrides_creates_nested_dicts；test_apply_overrides_merges_into_existing_branch；test_apply_overrides_empty_is_noop；test_apply_overrides_rejects_missing_equals；test_apply_overrides_rejects_empty_key；test_apply_overrides_rejects_non_mapping_path；test_apply_overrides_value_with_equals_sign"""
    # -- 原 test_apply_overrides_coerces_types --
    def _section_0_test_apply_overrides_coerces_types():
        data: dict = {}
        apply_overrides(
            data,
            [
                "backtest.top_n=30",  # int
                "preprocessing.neutralize=true",  # bool
                "backtest.alpha=0.2",  # float
                "preprocessing.normalizer=rank_normal",  # str
                "backtest.rebalance_threshold=null",  # None
            ],
        )
        assert data == {
            "backtest": {"top_n": 30, "alpha": 0.2, "rebalance_threshold": None},
            "preprocessing": {"neutralize": True, "normalizer": "rank_normal"},
        }

    _section_0_test_apply_overrides_coerces_types()

    # -- 原 test_apply_overrides_creates_nested_dicts --
    def _section_1_test_apply_overrides_creates_nested_dicts():
        data: dict = {"factor": "x"}
        apply_overrides(data, ["walk_forward.train_days=252"])
        assert data == {"factor": "x", "walk_forward": {"train_days": 252}}

    _section_1_test_apply_overrides_creates_nested_dicts()

    # -- 原 test_apply_overrides_merges_into_existing_branch --
    def _section_2_test_apply_overrides_merges_into_existing_branch():
        data: dict = {"backtest": {"cost_model": "linear"}}
        apply_overrides(data, ["backtest.top_n=20"])
        assert data["backtest"] == {"cost_model": "linear", "top_n": 20}

    _section_2_test_apply_overrides_merges_into_existing_branch()

    # -- 原 test_apply_overrides_empty_is_noop --
    def _section_3_test_apply_overrides_empty_is_noop():
        data: dict = {"a": 1}
        assert apply_overrides(data, []) == {"a": 1}

    _section_3_test_apply_overrides_empty_is_noop()

    # -- 原 test_apply_overrides_rejects_missing_equals --
    def _section_4_test_apply_overrides_rejects_missing_equals():
        with pytest.raises(ValueError, match="key=value"):
            apply_overrides({}, ["backtest.top_n"])

    _section_4_test_apply_overrides_rejects_missing_equals()

    # -- 原 test_apply_overrides_rejects_empty_key --
    def _section_5_test_apply_overrides_rejects_empty_key():
        with pytest.raises(ValueError, match="键名非法"):
            apply_overrides({}, ["=30"])

    _section_5_test_apply_overrides_rejects_empty_key()

    # -- 原 test_apply_overrides_rejects_non_mapping_path --
    def _section_6_test_apply_overrides_rejects_non_mapping_path():
        with pytest.raises(ValueError, match="不是映射"):
            apply_overrides({"backtest": 5}, ["backtest.top_n=30"])

    _section_6_test_apply_overrides_rejects_non_mapping_path()

    # -- 原 test_apply_overrides_value_with_equals_sign --
    def _section_7_test_apply_overrides_value_with_equals_sign():
        data: dict = {}
        apply_overrides(data, ["benchmark=000300.SH"])
        assert data == {"benchmark": "000300.SH"}

    _section_7_test_apply_overrides_value_with_equals_sign()


def test_build_from_dict_and_load_run_config_suite(tmp_path):
    """无显式 strategies 时，top_n 覆盖应通过 model_validator 生成对应 topn 策略。；test_build_from_dict_invalid_value_raises；test_load_run_config_applies_overrides；test_load_run_config_without_overrides_unchanged"""
    # -- 原 test_build_from_dict_bakes_legacy_top_n_into_strategy --
    def _section_0_test_build_from_dict_bakes_legacy_top_n_into_strategy():
        config = build_run_config_from_dict(
            {"factor": "f", "start": "20230101", "end": "20231231"},
            overrides=["backtest.top_n=30"],
        )
        assert config.backtest.top_n == 30
        assert len(config.backtest.strategies) == 1
        spec = config.backtest.strategies[0]
        assert spec.type == "topn_long_only"
        assert spec.name == "topn_30"
        assert spec.params == {"top_n": 30}

    _section_0_test_build_from_dict_bakes_legacy_top_n_into_strategy()

    # -- 原 test_build_from_dict_invalid_value_raises --
    def _section_1_test_build_from_dict_invalid_value_raises():
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            build_run_config_from_dict(
                {"factor": "f", "start": "20230101", "end": "20231231"},
                overrides=["preprocessing.normalizer=not_a_real_method"],
            )

    _section_1_test_build_from_dict_invalid_value_raises()

    # -- 原 test_load_run_config_applies_overrides --
    def _section_2_test_load_run_config_applies_overrides(tmp_path):
        cfg = tmp_path / "base.yaml"
        cfg.write_text(
            "factor: momentum_20d\nstart: '20230101'\nend: '20231231'\n"
            "preprocessing:\n  normalizer: zscore\n",
            encoding="utf-8",
        )
        config = load_run_config(
            cfg, overrides=["preprocessing.normalizer=rank_normal", "backtest.top_n=20"]
        )
        assert config.preprocessing.normalizer == "rank_normal"
        assert config.backtest.top_n == 20
        assert config.backtest.strategies[0].name == "topn_20"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_load_run_config_applies_overrides(_tp2)

    # -- 原 test_load_run_config_without_overrides_unchanged --
    def _section_3_test_load_run_config_without_overrides_unchanged(tmp_path):
        cfg = tmp_path / "base.yaml"
        cfg.write_text(
            "factor: f\nstart: '20230101'\nend: '20231231'\nbacktest:\n  top_n: 50\n",
            encoding="utf-8",
        )
        config = load_run_config(cfg)
        assert config.backtest.top_n == 50

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_load_run_config_without_overrides_unchanged(_tp3)


# ==== 来自 test_dotenv_loader.py ====
def test_dotenv_load_suite(tmp_path):
    """test_load_plain；带 UTF-8 BOM 的首行键不应被污染成 \\ufeffKEY。；test_load_handles_crlf；test_load_strips_quotes；test_load_skips_comments_and_blanks；test_load_does_not_override_existing；test_load_missing_file_is_noop；值内含 '=' 时按首个 '=' 切分，保留其余。；`KEY=val # 说明` 的行内注释应被剥离（dotenv 常见写法）。"""
    # -- 原 test_load_plain --
    def _section_0_test_load_plain(tmp_path):
        f = tmp_path / ".env"
        f.write_text("TUSHARE_TOKEN=abc123\n", encoding="utf-8")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env["TUSHARE_TOKEN"] == "abc123"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_load_plain(_tp0)

    # -- 原 test_load_strips_bom --
    def _section_1_test_load_strips_bom(tmp_path):
        f = tmp_path / ".env"
        f.write_bytes(b"\xef\xbb\xbfTUSHARE_TOKEN=abc123\n")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env["TUSHARE_TOKEN"] == "abc123"
        assert "﻿TUSHARE_TOKEN" not in env

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_load_strips_bom(_tp1)

    # -- 原 test_load_handles_crlf --
    def _section_2_test_load_handles_crlf(tmp_path):
        f = tmp_path / ".env"
        f.write_bytes(b"TUSHARE_TOKEN=abc123\r\nFOO=bar\r\n")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env["TUSHARE_TOKEN"] == "abc123"
        assert env["FOO"] == "bar"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_load_handles_crlf(_tp2)

    # -- 原 test_load_strips_quotes --
    def _section_3_test_load_strips_quotes(tmp_path):
        f = tmp_path / ".env"
        f.write_text('A="dq"\nB=\'sq\'\n', encoding="utf-8")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env["A"] == "dq"
        assert env["B"] == "sq"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_load_strips_quotes(_tp3)

    # -- 原 test_load_skips_comments_and_blanks --
    def _section_4_test_load_skips_comments_and_blanks(tmp_path):
        f = tmp_path / ".env"
        f.write_text("# comment\n\nKEY=val\n   \n", encoding="utf-8")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env == {"KEY": "val"}

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_load_skips_comments_and_blanks(_tp4)

    # -- 原 test_load_does_not_override_existing --
    def _section_5_test_load_does_not_override_existing(tmp_path):
        f = tmp_path / ".env"
        f.write_text("TUSHARE_TOKEN=from_file\n", encoding="utf-8")
        env = {"TUSHARE_TOKEN": "from_env"}
        _load_dotenv(f, env)
        assert env["TUSHARE_TOKEN"] == "from_env"  # 已有键不被覆盖

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_load_does_not_override_existing(_tp5)

    # -- 原 test_load_missing_file_is_noop --
    def _section_6_test_load_missing_file_is_noop(tmp_path):
        env: dict[str, str] = {}
        _load_dotenv(tmp_path / "nope.env", env)
        assert env == {}

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_load_missing_file_is_noop(_tp6)

    # -- 原 test_load_value_with_equals_sign --
    def _section_7_test_load_value_with_equals_sign(tmp_path):
        f = tmp_path / ".env"
        f.write_text("URL=https://x/y?a=1&b=2\n", encoding="utf-8")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env["URL"] == "https://x/y?a=1&b=2"

    _tp7 = tmp_path / "_s7"
    _tp7.mkdir(exist_ok=True)
    _section_7_test_load_value_with_equals_sign(_tp7)

    # -- 原 test_load_strips_inline_comment --
    def _section_8_test_load_strips_inline_comment(tmp_path):
        f = tmp_path / ".env"
        f.write_text("TUSHARE_MAX_RPS=5 # 限流说明\nKEY=val#nospace\n", encoding="utf-8")
        env: dict[str, str] = {}
        _load_dotenv(f, env)
        assert env["TUSHARE_MAX_RPS"] == "5"
        assert env["KEY"] == "val#nospace"  # 无空格的 # 不当注释（如 URL fragment）

    _tp8 = tmp_path / "_s8"
    _tp8.mkdir(exist_ok=True)
    _section_8_test_load_strips_inline_comment(_tp8)


# ==== 来自 test_tushare_config.py ====
def test_tushare_env_and_lake_suite(tmp_path):
    """test_tushare_env_file_is_project_root_env；行内注释/非数字值不应让 import 期 int() 崩溃：剥注释、失败回退默认。；test_to_df_preserves_numeric_columns_and_fills_short_rows；test_lake_ledger_and_atomic_parquet_are_resumable；test_api_call_uses_tushare_wire_contract；test_daily_quota_response_fails_closed_without_retry"""
    # -- 原 test_tushare_env_file_is_project_root_env --
    def _section_0_test_tushare_env_file_is_project_root_env():
        assert tushare_config._env_file == ROOT / ".env"

    _section_0_test_tushare_env_file_is_project_root_env()

    # -- 原 test_int_env_strips_inline_comment_and_falls_back --
    def _section_1_test_int_env_strips_inline_comment_and_falls_back(mp):
        from factorzen.config.tushare_config import _int_env

        mp.setenv("X_TEST_INT", "2000 # 注释")
        assert _int_env("X_TEST_INT", "100") == 2000
        mp.setenv("X_TEST_INT", "notanumber")
        assert _int_env("X_TEST_INT", "100") == 100

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_int_env_strips_inline_comment_and_falls_back(mp)

    # -- 原 test_to_df_preserves_numeric_columns_and_fills_short_rows --
    def _section_2_test_to_df_preserves_numeric_columns_and_fills_short_rows():
        frame = dl._to_df(["ts_code", "close"], [["000001.SZ", 10], ["000002.SZ"]])

        assert frame.columns == ["ts_code", "close"]
        assert frame["ts_code"].to_list() == ["000001.SZ", "000002.SZ"]
        assert frame["close"].to_list() == [10, None]

    _section_2_test_to_df_preserves_numeric_columns_and_fills_short_rows()

    # -- 原 test_lake_ledger_and_atomic_parquet_are_resumable --
    def _section_3_test_lake_ledger_and_atomic_parquet_are_resumable(tmp_path):
        lake = dl.Lake(tmp_path)
        lake.mark("minute", "000001.SZ")
        rows = lake.write_parquet(
            "minute/1min/000001.SZ.parquet",
            ["ts_code", "trade_time", "close"],
            [["000001.SZ", "2024-01-02 09:31:00", 10.0]],
        )

        reloaded = dl.Lake(tmp_path)
        assert rows == 1
        assert reloaded.done_set("minute") == {"000001.SZ"}
        assert (tmp_path / "minute/1min/000001.SZ.parquet").is_file()
        assert not list(tmp_path.rglob("*.tmp"))

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_lake_ledger_and_atomic_parquet_are_resumable(_tp3)

    # -- 原 test_api_call_uses_tushare_wire_contract --
    def _section_4_test_api_call_uses_tushare_wire_contract(mp):
        captured: dict = {}

        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"code": 0, "data": {"fields": ["x"], "items": [[1]]}}

        def fake_post(url, **kwargs):
            captured.update({"url": url, **kwargs})
            return Response()

        mp.setattr(dl.RL, "wait", lambda: None)
        mp.setattr(dl.requests, "post", fake_post)
        mp.setattr(dl, "TOKEN", "test-token")

        assert dl.api_call("trade_cal", {"exchange": "SSE"}) == (["x"], [[1]])
        assert captured["json"] == {
            "api_name": "trade_cal",
            "token": "test-token",
            "params": {"exchange": "SSE"},
            "fields": "",
        }

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_api_call_uses_tushare_wire_contract(mp)

    # -- 原 test_daily_quota_response_fails_closed_without_retry --
    def _section_5_test_daily_quota_response_fails_closed_without_retry(mp):
        class Response:
            status_code = 200
            text = json.dumps({"code": -2001})

            @staticmethod
            def json():
                return {"code": -2001, "msg": "今日调用已达上限，明日再试"}

        mp.setattr(dl.RL, "wait", lambda: None)
        mp.setattr(dl.requests, "post", lambda *args, **kwargs: Response())

        with pytest.raises(dl.DailyCap):
            dl.api_call("stk_mins", {}, max_tries=6)

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_daily_quota_response_fails_closed_without_retry(mp)


# ==== 来自 test_tushare_lake_downloader.py ====


