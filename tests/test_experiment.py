"""Tests for common.experiment module."""
from __future__ import annotations

import json

import pytest


def test_experiment_success(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from factorzen.core.config_loader import RunConfig

    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="test_run") as exp_dir:
        pass  # success

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["config"]["factor"] == "momentum_20d"
    assert manifest["end_ts"] is not None
    assert "git_sha" in manifest


def test_auto_run_id_includes_factor_name(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="momentum_12_1", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg) as exp_dir:
        pass

    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exp_dir.name.startswith("momentum_12_1_")
    assert manifest["run_id"] == exp_dir.name


def test_experiment_failure(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from factorzen.core.config_loader import RunConfig

    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with pytest.raises(ValueError), exp_mod.run_experiment(cfg, run_id="fail_run") as exp_dir:
        raise ValueError("test error")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["status"] == "failure"
    assert "test error" in manifest["error"]


def test_experiment_index_created_on_success(tmp_path, monkeypatch):
    """成功 run 后，experiment_index.jsonl 被创建并含正确字段。"""
    import json as _json

    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="reversal_5d", start="20240101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="idx_run") as _exp_dir:
        pass

    index_path = tmp_path / "experiments" / "experiment_index.jsonl"
    assert index_path.exists(), "experiment_index.jsonl 应被创建"
    entry = _json.loads(index_path.read_text(encoding="utf-8").strip())
    assert entry["run_id"] == "idx_run"
    assert entry["factor"] == "reversal_5d"
    assert entry["status"] == "success"
    assert "manifest_path" in entry


def test_experiment_index_appends_multiple_runs(tmp_path, monkeypatch):
    """两次 run 各 append 一行，JSONL 共两行。"""
    import json as _json

    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")

    for run_id, factor in [("run_a", "momentum_20d"), ("run_b", "reversal_5d")]:
        cfg = RunConfig(factor=factor, start="20240101", end="20241231")
        with exp_mod.run_experiment(cfg, run_id=run_id):
            pass

    index_path = tmp_path / "experiments" / "experiment_index.jsonl"
    lines = [_json.loads(ln) for ln in index_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    factors = {e["factor"] for e in lines}
    assert factors == {"momentum_20d", "reversal_5d"}


def test_experiment_index_records_failure_status(tmp_path, monkeypatch):
    """失败 run 的状态也被记录到索引。"""
    import json as _json

    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="x", start="20240101", end="20241231")

    with pytest.raises(RuntimeError), exp_mod.run_experiment(cfg, run_id="fail_idx"):
        raise RuntimeError("boom")

    index_path = tmp_path / "experiments" / "experiment_index.jsonl"
    entry = _json.loads(index_path.read_text(encoding="utf-8").strip())
    assert entry["status"] == "failure"


def test_experiment_records_reproducibility_metadata(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: True)
    monkeypatch.setattr(exp_mod, "_get_pixi_lock_hash", lambda: "abc123")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(
        cfg,
        run_id="metadata_run",
        command=["python", "factorzen.pipelines.daily_single"],
    ) as exp_dir:
        exp_mod.record_experiment_output(exp_dir, "quality_report", "quality.json")

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["git_dirty"] is True
    assert manifest["pixi_lock_sha256"] == "abc123"
    assert manifest["command"] == ["python", "factorzen.pipelines.daily_single"]
    assert manifest["outputs"]["quality_report"] == "quality.json"


def test_record_experiment_metadata_survives_run_finalization(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="meta_run") as exp_dir:
        exp_mod.record_experiment_metadata(exp_dir, "stage_timings", {"ic": 1.2, "backtest": 3.4})

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    # 运行期写入的顶层元数据不应被 run_experiment 的 finally 覆盖丢失
    assert manifest["stage_timings"] == {"ic": 1.2, "backtest": 3.4}
    assert manifest["status"] == "success"
    assert manifest["end_ts"] is not None


def test_manifest_records_duration_seconds(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="dur_run"):
        pass

    manifest = json.loads((tmp_path / "experiments" / "dur_run" / "manifest.json").read_text())
    assert "duration_seconds" in manifest
    assert isinstance(manifest["duration_seconds"], (int, float))
    assert manifest["duration_seconds"] >= 0


def test_experiment_warns_when_git_dirty(tmp_path, monkeypatch, caplog):
    import logging

    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: True)
    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with caplog.at_level(logging.WARNING), exp_mod.run_experiment(cfg, run_id="dirty_warn_run"):
        pass

    assert any("git_dirty" in r.getMessage() for r in caplog.records)


def test_experiment_does_not_warn_when_clean(tmp_path, monkeypatch, caplog):
    import logging

    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: False)
    cfg = RunConfig(factor="x", start="20230101", end="20241231")

    with caplog.at_level(logging.WARNING), exp_mod.run_experiment(cfg, run_id="clean_run"):
        pass

    assert not any("git_dirty" in r.getMessage() for r in caplog.records)


def test_build_manifest_base_returns_reproducibility_fields(monkeypatch):
    """build_manifest_base() 是可被其它 pipeline（risk_build/portfolio_build）复用的基础字段构造器。"""
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "get_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(exp_mod, "_get_git_dirty", lambda: False)
    monkeypatch.setattr(exp_mod, "_get_pixi_lock_hash", lambda: "lockhash123")

    base = exp_mod.build_manifest_base(
        ["python", "-m", "factorzen.cli.main", "risk", "build"],
        {"start": "20230101", "end": "20231231"},
    )

    assert base["schema_version"] == "1"
    assert base["git_sha"] == "deadbeef"
    assert base["git_dirty"] is False
    assert base["pixi_lock_sha256"] == "lockhash123"
    assert base["command"] == ["python", "-m", "factorzen.cli.main", "risk", "build"]
    assert base["config"] == {"start": "20230101", "end": "20231231"}
    assert base.get("start_ts")


def test_build_manifest_base_accepts_plain_dict_config(monkeypatch):
    """非 RunConfig 调用方（如 risk_build/portfolio_build）可直接传 dict 作为 config。"""
    from factorzen.core import experiment as exp_mod

    base = exp_mod.build_manifest_base(None, {"cov_half_life": 90, "nw_lags": 2})

    assert base["command"] is None
    assert base["config"] == {"cov_half_life": 90, "nw_lags": 2}


def test_build_manifest_base_used_by_run_experiment_unchanged(tmp_path, monkeypatch):
    """run_experiment() 重构为复用 build_manifest_base 后，对外行为（字段集合/取值）保持不变。"""
    from factorzen.core import experiment as exp_mod
    from factorzen.core.config_loader import RunConfig

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20241231")

    with exp_mod.run_experiment(cfg, run_id="base_reuse_run", command=["fz", "daily-single"]) as exp_dir:
        pass

    manifest = json.loads((exp_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "1"
    assert manifest["run_id"] == "base_reuse_run"
    assert manifest["command"] == ["fz", "daily-single"]
    assert manifest["config"]["factor"] == "momentum_20d"
    assert isinstance(manifest["git_dirty"], bool)
    assert isinstance(manifest["pixi_lock_sha256"], str)
