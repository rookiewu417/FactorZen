"""Tests for common.experiment module."""
from __future__ import annotations

import json

import pytest


def test_experiment_success(tmp_path, monkeypatch):
    from common import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from common.config_loader import RunConfig

    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="test_run") as exp_dir:
        pass  # success

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["config"]["factor"] == "momentum_20d"
    assert manifest["end_ts"] is not None
    assert "git_sha" in manifest


def test_experiment_failure(tmp_path, monkeypatch):
    from common import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from common.config_loader import RunConfig

    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with pytest.raises(ValueError), exp_mod.run_experiment(cfg, run_id="fail_run") as exp_dir:
        raise ValueError("test error")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["status"] == "failure"
    assert "test error" in manifest["error"]
