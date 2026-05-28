"""Tests for run_daily_single configuration merging."""

from __future__ import annotations

from argparse import Namespace
from datetime import date

import polars as pl
import pytest


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


def test_preprocess_with_industry_neutralization_uses_universe_industry():
    from common.config_loader import RunConfig
    from scripts.run_daily_single import _preprocess_factor

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


def test_run_ensures_required_data_before_loading_universe(monkeypatch):
    import scripts.run_daily_single as run_mod
    from common.config_loader import RunConfig

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
