"""Tests for script-level experiment manifest wrappers."""

from __future__ import annotations

import json
import sys

import pytest


def _single_manifest(experiments_dir):
    manifests = list(experiments_dir.glob("*/manifest.json"))
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8"))


def test_generate_report_failure_manifest_records_partial_outputs(tmp_path, monkeypatch):
    from common import experiment as exp_mod
    from scripts import generate_report as mod

    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(mod, "OUTPUT_DAILY_RESULTS", tmp_path / "results")
    monkeypatch.setattr(mod, "OUTPUT_DAILY_REPORTS", tmp_path / "reports")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_report.py",
            "--factor",
            "momentum_20d",
            "--start",
            "20240101",
            "--end",
            "20240131",
        ],
    )

    def fail_after_meta(args, effective_config):
        meta_path = mod._meta_path(args.factor, args.start, args.end)
        meta_path.write_text("{}", encoding="utf-8")
        raise RuntimeError("report boom")

    monkeypatch.setattr(mod, "_run", fail_after_meta)

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 1
    manifest = _single_manifest(experiments_dir)
    assert manifest["status"] == "failure"
    assert manifest["error"] == "report boom"
    assert manifest["config"]["factor"] == "momentum_20d"
    assert manifest["outputs"]["meta"] == str(
        tmp_path / "results" / "momentum_20d_20240101_20240131_meta.json"
    )


def test_run_daily_failure_manifest_records_partial_outputs(tmp_path, monkeypatch):
    from common import experiment as exp_mod
    from scripts import run_daily_single as mod

    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(mod, "OUTPUT_DAILY_FACTORS", tmp_path / "factors")
    monkeypatch.setattr(mod, "OUTPUT_DAILY_RESULTS", tmp_path / "results")
    monkeypatch.setattr(mod, "OUTPUT_DAILY_REPORTS", tmp_path / "reports")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_daily_single.py",
            "--factor",
            "momentum_20d",
            "--start",
            "20240101",
            "--end",
            "20240131",
        ],
    )

    def fail_after_quality(args, effective_config):
        quality_path = mod.OUTPUT_DAILY_RESULTS / (
            f"{args.factor}_{args.start}_{args.end}_quality.json"
        )
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text("{}", encoding="utf-8")
        raise RuntimeError("daily boom")

    monkeypatch.setattr(mod, "_run", fail_after_quality)

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 1
    manifest = _single_manifest(experiments_dir)
    assert manifest["status"] == "failure"
    assert manifest["error"] == "daily boom"
    assert manifest["config"]["factor"] == "momentum_20d"
    assert manifest["outputs"]["quality_report"] == str(
        tmp_path / "results" / "momentum_20d_20240101_20240131_quality.json"
    )
