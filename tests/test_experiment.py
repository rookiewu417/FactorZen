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


def test_experiment_records_reproducibility_metadata(tmp_path, monkeypatch):
    from common import experiment as exp_mod
    from common.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: True)
    monkeypatch.setattr(exp_mod, "_get_pixi_lock_hash", lambda: "abc123")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(
        cfg,
        run_id="metadata_run",
        command=["python", "scripts/run_daily_single.py"],
    ) as exp_dir:
        exp_mod.record_experiment_output(exp_dir, "quality_report", "quality.json")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["git_dirty"] is True
    assert manifest["pixi_lock_sha256"] == "abc123"
    assert manifest["command"] == ["python", "scripts/run_daily_single.py"]
    assert manifest["outputs"]["quality_report"] == "quality.json"
