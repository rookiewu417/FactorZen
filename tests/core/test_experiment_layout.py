"""run_experiment 新落点 + 全局 experiment_index。"""

from __future__ import annotations

import json


def test_run_experiment_with_factor_nested_layout(tmp_path, monkeypatch):
    from factorzen.config.research import RunConfig
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "factors")
    cfg = RunConfig(factor="momentum_20d", start="20230101", end="20231231")

    with exp_mod.run_experiment(cfg, run_id="momentum_20d_20260101_120000") as exp_dir:
        assert exp_dir == (
            tmp_path
            / "factors"
            / "ashare"
            / "momentum_20d"
            / "evaluations"
            / "momentum_20d_20260101_120000"
        )
        assert (exp_dir / "manifest.json").exists()

    index = tmp_path / "factors" / "experiment_index.jsonl"
    assert index.exists()
    entry = json.loads(index.read_text(encoding="utf-8").strip())
    assert entry["run_id"] == "momentum_20d_20260101_120000"
    assert entry["factor"] == "momentum_20d"
    assert "evaluations" in entry["manifest_path"]
    # index 在 factors 根，不在 evaluations 旁
    assert index.parent == tmp_path / "factors"


def test_run_experiment_without_factor_uses_runs(tmp_path, monkeypatch):
    from factorzen.core import experiment as exp_mod

    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path / "factors")
    with exp_mod.run_experiment({"note": "sweep"}, run_id="20260101_120000") as exp_dir:
        assert exp_dir == tmp_path / "factors" / "_runs" / "20260101_120000"

    index = tmp_path / "factors" / "experiment_index.jsonl"
    assert index.exists()
    entry = json.loads(index.read_text(encoding="utf-8").strip())
    assert entry["run_id"] == "20260101_120000"
    assert entry["factor"] is None


def test_find_run_dir_and_run_dir_peel(tmp_path, monkeypatch):
    from factorzen.experiments import run_paths as rp

    store = tmp_path / "factors"
    monkeypatch.setattr(rp, "FACTOR_STORE_DIR", store)

    nested = store / "ashare" / "alpha_x" / "evaluations" / "alpha_x_20260101_010101"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text("{}", encoding="utf-8")

    plain = store / "_runs" / "no_factor_run"
    plain.mkdir(parents=True)

    # reports 不应被当 run
    (store / "reports" / "daily").mkdir(parents=True)

    assert rp.find_run_dir("alpha_x_20260101_010101") == nested
    assert rp.find_run_dir("no_factor_run") == plain
    assert rp.find_run_dir("../evil") is None
    assert rp.find_run_dir("missing") is None

    assert rp.run_dir("alpha_x_20260101_010101") == nested
    assert rp.run_dir("plain", factor=None) == store / "_runs" / "plain"
