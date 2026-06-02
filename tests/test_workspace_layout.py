from pathlib import Path


def test_run_artifacts_are_copied_with_stable_names(tmp_path):
    from factorzen.experiments.run_paths import copy_outputs_to_run_dir

    report = tmp_path / "momentum_20d_20240101_20240131.html"
    report.write_text("<html></html>", encoding="utf-8")
    quality = tmp_path / "momentum_20d_20240101_20240131_quality.json"
    quality.write_text("{}", encoding="utf-8")

    copied = copy_outputs_to_run_dir(
        {"report": str(report), "quality_report": str(quality)},
        tmp_path / "run",
    )

    assert Path(copied["run_report"]).name == "report.html"
    assert Path(copied["run_quality_report"]).name == "quality.json"
    assert (tmp_path / "run" / "report.html").read_text(encoding="utf-8") == "<html></html>"
    assert (tmp_path / "run" / "quality.json").read_text(encoding="utf-8") == "{}"


def test_run_dir_uses_factor_evaluations_folder():
    from factorzen.config.settings import WORKSPACE_DIR
    from factorzen.experiments.run_paths import run_dir

    assert run_dir("momentum_12_1_20260530_031234") == (
        WORKSPACE_DIR / "factor_evaluations" / "momentum_12_1_20260530_031234"
    )


def test_fz_factor_new_writes_to_workspace(tmp_path, monkeypatch):
    from factorzen.cli import main as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)

    assert cli.main(["factor", "new", "my_alpha", "--freq", "daily"]) == 0

    factor_path = tmp_path / "workspace" / "factors" / "daily" / "my_alpha.py"
    assert factor_path.exists()
    text = factor_path.read_text(encoding="utf-8")
    assert 'name = "my_alpha"' in text
    assert "class MyAlphaFactor" in text


def test_fz_factor_new_accepts_frequency_alias(tmp_path, monkeypatch):
    from factorzen.cli import main as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)

    assert cli.main(["factor", "new", "my_weekly_alpha", "--frequency", "weekly"]) == 0

    assert (tmp_path / "workspace" / "factors" / "weekly" / "my_weekly_alpha.py").exists()
