from __future__ import annotations

import json
import sys


def test_factor_run_forwards_to_daily_pipeline(monkeypatch):
    from factorzen.cli import main as cli

    captured: list[str] = []

    def fake_main():
        captured.extend(sys.argv)

    monkeypatch.setattr("factorzen.pipelines.daily_single.main", fake_main)

    assert (
        cli.main(
            [
                "factor",
                "run",
                "momentum_20d",
                "--start",
                "20250101",
                "--end",
                "20260513",
                "--universe",
                "csi500",
                "--frequency",
                "weekly",
                "--config",
                "workspace/configs/daily/daily_factor_template.yaml",
                "--seed",
                "42",
                "--dry-run",
            ]
        )
        == 0
    )

    assert captured == [
        "fz factor run",
        "--factor",
        "momentum_20d",
        "--start",
        "20250101",
        "--end",
        "20260513",
        "--universe",
        "csi500",
        "--frequency",
        "weekly",
        "--config",
        "workspace/configs/daily/daily_factor_template.yaml",
        "--seed",
        "42",
        "--dry-run",
    ]


def test_report_build_forwards_to_report_pipeline(monkeypatch):
    from factorzen.cli import main as cli

    captured: list[str] = []

    def fake_main():
        captured.extend(sys.argv)

    monkeypatch.setattr("factorzen.pipelines.generate_report.main", fake_main)

    assert (
        cli.main(
            [
                "report",
                "build",
                "momentum_20d",
                "--start",
                "20250101",
                "--end",
                "20260513",
                "--universe",
                "csi300",
                "--frequency",
                "monthly",
                "--benchmark",
                "000300.SH",
                "--config",
                "workspace/configs/daily/daily_factor_template.yaml",
                "--reuse",
            ]
        )
        == 0
    )

    assert captured == [
        "fz report build",
        "--factor",
        "momentum_20d",
        "--start",
        "20250101",
        "--end",
        "20260513",
        "--universe",
        "csi300",
        "--frequency",
        "monthly",
        "--benchmark",
        "000300.SH",
        "--config",
        "workspace/configs/daily/daily_factor_template.yaml",
        "--reuse",
    ]


def test_report_path_prints_stable_run_report_path(tmp_path, monkeypatch, capsys):
    from factorzen.cli import main as cli

    run_dir = tmp_path / "workspace" / "factor_evaluations" / "run-1"
    run_dir.mkdir(parents=True)
    report = run_dir / "report.html"
    report.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(
        "factorzen.experiments.run_paths.FACTOR_EVALUATIONS_DIR",
        tmp_path / "workspace" / "factor_evaluations",
    )

    assert cli.main(["report", "path", "run-1"]) == 0

    assert capsys.readouterr().out.strip() == str(report)


def test_config_validate_prints_effective_config_and_output_dir(tmp_path, monkeypatch, capsys):
    from factorzen.cli import main as cli

    config = tmp_path / "run.yaml"
    config.write_text(
        "\n".join(
            [
                "factor: momentum_20d",
                "universe: csi500",
                'start: "20230101"',
                'end: "20231231"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    assert cli.main(["config", "validate", str(config)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["benchmark"] == "000905.SH"
    assert payload["output_dir"].endswith("workspace/factor_evaluations/<run_id>")


def test_runs_list_reads_experiment_index(tmp_path, monkeypatch, capsys):
    from factorzen.cli import main as cli

    root = tmp_path / "workspace" / "factor_evaluations"
    root.mkdir(parents=True)
    (root / "experiment_index.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "timestamp": "2026-05-30T01:02:03",
                "factor": "momentum_20d",
                "universe": "csi500",
                "status": "success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "FACTOR_EVALUATIONS_DIR", root)

    assert cli.main(["runs", "list"]) == 0

    out = capsys.readouterr().out
    assert "run-1" in out
    assert "momentum_20d" in out
    assert "success" in out


def test_runs_show_reads_manifest(tmp_path, monkeypatch, capsys):
    from factorzen.cli import main as cli

    root = tmp_path / "workspace" / "factor_evaluations"
    run_dir = root / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "success",
                "config": {"factor": "momentum_20d", "benchmark": "000905.SH"},
                "outputs": {"run_report": str(run_dir / "report.html")},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "FACTOR_EVALUATIONS_DIR", root)

    assert cli.main(["runs", "show", "run-1"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-1"
    assert payload["config"]["benchmark"] == "000905.SH"


def test_data_fetch_daily_and_daily_basic(monkeypatch):
    from factorzen.cli import main as cli

    calls: list[tuple[str, str, str]] = []

    def fake_fetch_daily(start: str, end: str):
        calls.append(("daily", start, end))
        return []

    def fake_fetch_daily_basic(start: str, end: str):
        calls.append(("daily-basic", start, end))
        return []

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", fake_fetch_daily_basic)

    assert cli.main(["data", "fetch", "daily", "--start", "20250101", "--end", "20250131"]) == 0
    assert (
        cli.main(
            ["data", "fetch", "daily-basic", "--start", "20250101", "--end", "20250131"]
        )
        == 0
    )

    assert calls == [
        ("daily", "20250101", "20250131"),
        ("daily-basic", "20250101", "20250131"),
    ]


def test_data_fetch_margin_detail(monkeypatch):
    from factorzen.cli import main as cli

    calls: list[tuple[str, str]] = []

    def fake_fetch_margin(start: str, end: str):
        calls.append((start, end))
        return [1, 2, 3]

    monkeypatch.setattr("factorzen.core.loader.fetch_margin_detail", fake_fetch_margin)
    assert (
        cli.main(
            ["data", "fetch", "margin_detail", "--start", "20240101", "--end", "20240131"]
        )
        == 0
    )
    assert calls == [("20240101", "20240131")]
