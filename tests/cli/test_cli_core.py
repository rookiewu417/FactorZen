"""test_cli.py：CLI 主入口转发（factor/mine/portfolio/sim/validate）冒烟。
test_cli_market.py：MC1 T7: fz mine search/export-alpha 的 --market 参数（默认 ashare 不变）。
test_workspace_layout.py：workspace 布局：run_dir / fz factor new 写路径与 frequency 别名。
test_wave6_crash_guards.py：Wave6 crash-P2：sim 跳过半成品目录（无 manifest）+ validate overfit 缺参友好报错。
"""



from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.cli.main import (
    _cmd_mine_export_alpha,
    _cmd_mine_search,
    _cmd_portfolio_build,
    _cmd_sim_run,
    _cmd_validate_overfit,
    build_parser,
)

# ==== 来自 test_cli.py ====

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


def test_every_subparser_help_renders():
    """argparse 对 help 串做 % 格式化——未转义的 % 会让 --help 直接崩。

    遍历全部 subparser 递归 format_help()，任一节点抛异常即失败。
    """
    import argparse

    from factorzen.cli.main import build_parser

    def walk(parser: argparse.ArgumentParser, path: str) -> list[str]:
        failures: list[str] = []
        try:
            parser.format_help()
        except Exception as exc:  # 汇总全部失败点再报，不在首个失败处中断
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    failures.extend(walk(sub, f"{path} {name}"))
        return failures

    failures = walk(build_parser(), "fz")
    assert not failures, "以下命令的 --help 无法渲染:\n" + "\n".join(failures)


# ==== 来自 test_cli_market.py ====

def test_mine_search_market_default_ashare():
    p = build_parser()
    args = p.parse_args(["mine", "search", "--start", "20240101", "--end", "20240201"])
    assert args.market == "ashare"
    assert args.func is _cmd_mine_search


def test_mine_search_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "mine", "search", "--start", "20240101", "--end", "20240201",
        "--market", "crypto", "--top-n", "30",
    ])
    assert args.market == "crypto"
    assert args.top_n == 30
    assert args.func is _cmd_mine_search


def test_export_alpha_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "mine", "export-alpha", "--session", "s", "--date", "20240201",
        "--out", "o.parquet", "--market", "crypto",
    ])
    assert args.market == "crypto"
    assert args.func is _cmd_mine_export_alpha


def test_validate_overfit_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "validate", "overfit", "--start", "20240101", "--end", "20240201",
        "--market", "crypto", "--expression", "ts_mean(ret_1d, 5)",
    ])
    assert args.market == "crypto"
    assert args.expression == "ts_mean(ret_1d, 5)"
    assert args.factor is None  # crypto 不用 positional factor
    assert args.func is _cmd_validate_overfit


def test_validate_overfit_ashare_positional_unchanged():
    p = build_parser()
    args = p.parse_args(["validate", "overfit", "momentum_12_1",
                         "--start", "20230101", "--end", "20240101"])
    assert args.market == "ashare"
    assert args.factor == "momentum_12_1"


def test_portfolio_build_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "portfolio", "build", "--start", "20240101", "--end", "20240224",
        "--alpha-file", "a.parquet", "--market", "crypto", "--gross-limit", "1.5",
    ])
    assert args.market == "crypto"
    assert args.gross_limit == 1.5
    assert args.func is _cmd_portfolio_build


def test_portfolio_build_default_ashare():
    p = build_parser()
    args = p.parse_args([
        "portfolio", "build", "--start", "20240101", "--end", "20240224",
        "--alpha-file", "a.parquet",
    ])
    assert args.market == "ashare"


def test_sim_run_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "sim", "run", "--portfolio-dir", "d", "--start", "20240201", "--end", "20240224",
        "--market", "crypto",
    ])
    assert args.market == "crypto"
    assert args.func is _cmd_sim_run


def test_sim_run_default_ashare():
    p = build_parser()
    args = p.parse_args([
        "sim", "run", "--portfolio-dir", "d", "--start", "20240201", "--end", "20240224",
    ])
    assert args.market == "ashare"


def test_freq_parsed_for_crypto_and_defaults_daily():
    p = build_parser()
    a = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502",
                      "--market", "crypto", "--freq", "15m"])
    assert a.freq == "15m"
    b = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502"])
    assert b.freq == "daily"  # 默认 daily,ashare 零回归


def test_data_crypto_backfill_parser():
    from factorzen.cli.main import _cmd_data_crypto_backfill
    p = build_parser()
    a = p.parse_args(["data", "crypto", "backfill", "--start", "20260501", "--end", "20260502",
                      "--symbols", "BTCUSDT,ETHUSDT", "--lake-root", "/tmp/lk"])
    assert a.func is _cmd_data_crypto_backfill
    assert a.symbols == "BTCUSDT,ETHUSDT" and a.start == "20260501"


def test_ashare_rejects_intraday_freq(capsys):
    from factorzen.cli.main import _cmd_mine_search
    p = build_parser()
    a = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502",
                      "--freq", "15m"])  # market 默认 ashare
    assert _cmd_mine_search(a) == 2
    assert "仅 crypto" in capsys.readouterr().err


# ==== 来自 test_workspace_layout.py ====

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


# ==== 来自 test_wave6_crash_guards.py ====

def test_load_weights_skips_dir_without_manifest(tmp_path: Path):
    """含 weights.parquet 无 manifest.json 的半成品目录应被跳过，不 FileNotFoundError。"""
    from factorzen.sim.engine import _load_weights_by_date

    good = tmp_path / "20240102"
    good.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [1.0]}).write_parquet(good / "weights.parquet")
    import json
    (good / "manifest.json").write_text(json.dumps({"signal_date": "2024-01-02", "status": "optimal"}))

    half = tmp_path / "20240103"  # 半成品：只有 weights，无 manifest
    half.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [1.0]}).write_parquet(half / "weights.parquet")

    out = _load_weights_by_date([str(good), str(half)])  # 不应抛异常
    assert date(2024, 1, 2) in out
    assert len(out) == 1  # 半成品目录被跳过


def test_validate_overfit_missing_factor_friendly_error(capsys):
    """fz validate overfit 不给 factor → 返回 2 + 友好提示，而非裸 KeyError traceback。"""
    from factorzen.cli.main import main

    rc = main(["validate", "overfit", "--start", "20240101", "--end", "20241231"])
    assert rc == 2
    assert "缺少因子名" in capsys.readouterr().err
