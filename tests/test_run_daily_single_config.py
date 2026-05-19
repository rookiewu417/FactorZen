"""Tests for run_daily_single configuration merging."""

from __future__ import annotations

from argparse import Namespace


def test_merge_run_config_args_uses_yaml_for_missing_cli_values():
    from common.config_loader import RunConfig
    from scripts.run_daily_single import _merge_run_config_args

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
    )
    cfg = RunConfig(
        factor="momentum_20d",
        start="20230101",
        end="20241231",
        universe="csi500",
        benchmark="000300.SH",
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
    assert merged.benchmark == "000300.SH"
    assert merged.seed == 42
    assert merged.ic_method == "both"
    assert merged.neutralized_ic is True
    assert merged.event_study is True


def test_merge_run_config_args_keeps_explicit_cli_values():
    from common.config_loader import RunConfig
    from scripts.run_daily_single import _merge_run_config_args

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
