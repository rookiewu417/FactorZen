"""config_loader 的 --set 覆盖层单测：值类型推断、嵌套路径、legacy top_n bake、非法输入。"""

from __future__ import annotations

import pytest

from factorzen.config.research import (
    apply_overrides,
    build_run_config_from_dict,
    load_run_config,
)


def test_apply_overrides_coerces_types():
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


def test_apply_overrides_creates_nested_dicts():
    data: dict = {"factor": "x"}
    apply_overrides(data, ["walk_forward.train_days=252"])
    assert data == {"factor": "x", "walk_forward": {"train_days": 252}}


def test_apply_overrides_merges_into_existing_branch():
    data: dict = {"backtest": {"cost_model": "linear"}}
    apply_overrides(data, ["backtest.top_n=20"])
    assert data["backtest"] == {"cost_model": "linear", "top_n": 20}


def test_apply_overrides_empty_is_noop():
    data: dict = {"a": 1}
    assert apply_overrides(data, []) == {"a": 1}


def test_apply_overrides_rejects_missing_equals():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["backtest.top_n"])


def test_apply_overrides_rejects_empty_key():
    with pytest.raises(ValueError, match="键名非法"):
        apply_overrides({}, ["=30"])


def test_apply_overrides_rejects_non_mapping_path():
    with pytest.raises(ValueError, match="不是映射"):
        apply_overrides({"backtest": 5}, ["backtest.top_n=30"])


def test_apply_overrides_value_with_equals_sign():
    data: dict = {}
    apply_overrides(data, ["benchmark=000300.SH"])
    assert data == {"benchmark": "000300.SH"}


def test_build_from_dict_bakes_legacy_top_n_into_strategy():
    """无显式 strategies 时，top_n 覆盖应通过 model_validator 生成对应 topn 策略。"""
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


def test_build_from_dict_invalid_value_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build_run_config_from_dict(
            {"factor": "f", "start": "20230101", "end": "20231231"},
            overrides=["preprocessing.normalizer=not_a_real_method"],
        )


def test_load_run_config_applies_overrides(tmp_path):
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


def test_load_run_config_without_overrides_unchanged(tmp_path):
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "factor: f\nstart: '20230101'\nend: '20231231'\nbacktest:\n  top_n: 50\n",
        encoding="utf-8",
    )
    config = load_run_config(cfg)
    assert config.backtest.top_n == 50
