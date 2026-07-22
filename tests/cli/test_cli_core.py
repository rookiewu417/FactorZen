"""合并自: test_cli_core.py, test_validation_report_cli.py
目标: test_cli_core.py

--- 来源 test_cli_core.py ---
test_cli.py：CLI 主入口转发（factor/mine/portfolio/sim/validate）冒烟。
test_cli_market.py：MC1 T7: fz mine search/export-alpha 的 --market 参数（默认 ashare 不变）。
test_workspace_layout.py：workspace 布局：run_dir / fz factor new 写路径与 frequency 别名。
test_wave6_crash_guards.py：Wave6 crash-P2：sim 跳过半成品目录（无 manifest）+ validate overfit 缺参友好报错。

--- 来源 test_validation_report_cli.py ---
test_validation_cli.py：Tests for `fz validate overfit` CLI command.
test_report_cli.py：Tests for `fz report portfolio` CLI parser (Task 5).
test_ops_cli_smoke.py：fz ops daily/status CLI 冒烟(dispatch/日期解析/返回码/状态打印)。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from factorzen.cli.main import (
    _cmd_mine_export_alpha,
    _cmd_mine_search,
    _cmd_portfolio_build,
    _cmd_sim_run,
    _cmd_validate_overfit,
    build_parser,
    main,
)
from factorzen.ops.state import OpsState

# ==== 来自 test_cli_core.py ====
# ==== 来自 test_cli.py ====

def test_pipeline_argv_forward_suite():
    """test_factor_eval_forwards_to_daily_pipeline；test_factor_backtest_forwards_to_daily_pipeline；test_report_build_forwards_to_report_pipeline；test_data_fetch_daily_and_daily_basic；test_data_fetch_margin_detail"""
    # -- factor eval 转发 --
    def _section_0_test_factor_eval_forwards_to_daily_pipeline(mp):
        from factorzen.cli import main as cli

        captured: list[str] = []
        tracks: list[str] = []

        def fake_main(*, track="backtest"):
            captured.extend(sys.argv)
            tracks.append(track)

        mp.setattr("factorzen.pipelines.daily_single.main", fake_main)

        assert (
            cli.main(
                [
                    "factor",
                    "eval",
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
            "fz factor eval",
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
            "--exec-lag",
            "1",
            "--exec-price-col",
            "open_adj",
            # 信号轨专属旋钮:eval 子命令独有,必须真的转发下去
            # (曾漏接线——CLI 层收了参数但拼 argv 时没带,只看 --help 发现不了)
            "--n-groups",
            "5",
            # 信号轨不该有成本参数:它是纯毛口径,成本走 fz factor backtest
        ]
        assert tracks == ["eval"]

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_factor_eval_forwards_to_daily_pipeline(mp)

    # -- factor backtest 转发 --
    def _section_0b_test_factor_backtest_forwards_to_daily_pipeline(mp):
        from factorzen.cli import main as cli

        captured: list[str] = []
        tracks: list[str] = []

        def fake_main(*, track="backtest"):
            captured.extend(sys.argv)
            tracks.append(track)

        mp.setattr("factorzen.pipelines.daily_single.main", fake_main)

        assert (
            cli.main(
                [
                    "factor",
                    "backtest",
                    "momentum_20d",
                    "--start",
                    "20250101",
                    "--end",
                    "20260513",
                    "--universe",
                    "csi500",
                    "--dry-run",
                ]
            )
            == 0
        )

        assert captured[0] == "fz factor backtest"
        assert tracks == ["backtest"]

    with pytest.MonkeyPatch.context() as mp:
        _section_0b_test_factor_backtest_forwards_to_daily_pipeline(mp)

    # -- 旧子命令 run 已删除 --
    def _section_0c_test_legacy_run_subcommand_removed():
        from factorzen.cli import main as cli

        with pytest.raises(SystemExit):
            cli.main(["factor", "run", "momentum_20d", "--start", "20250101", "--end", "20260513"])

    _section_0c_test_legacy_run_subcommand_removed()

    # -- 原 test_report_build_forwards_to_report_pipeline --
    def _section_1_test_report_build_forwards_to_report_pipeline(mp):
        from factorzen.cli import main as cli

        captured: list[str] = []

        def fake_main():
            captured.extend(sys.argv)

        mp.setattr("factorzen.pipelines.generate_report.main", fake_main)

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

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_report_build_forwards_to_report_pipeline(mp)

    # -- 原 test_data_fetch_daily_and_daily_basic --
    def _section_2_test_data_fetch_daily_and_daily_basic(mp):
        from factorzen.cli import main as cli

        calls: list[tuple[str, str, str]] = []

        def fake_fetch_daily(start: str, end: str):
            calls.append(("daily", start, end))
            return []

        def fake_fetch_daily_basic(start: str, end: str):
            calls.append(("daily-basic", start, end))
            return []

        mp.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", fake_fetch_daily_basic)

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

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_data_fetch_daily_and_daily_basic(mp)

    # -- 原 test_data_fetch_margin_detail --
    def _section_3_test_data_fetch_margin_detail(mp):
        from factorzen.cli import main as cli

        calls: list[tuple[str, str]] = []

        def fake_fetch_margin(start: str, end: str):
            calls.append((start, end))
            return [1, 2, 3]

        mp.setattr("factorzen.core.loader.fetch_margin_detail", fake_fetch_margin)
        assert (
            cli.main(
                ["data", "fetch", "margin_detail", "--start", "20240101", "--end", "20240131"]
            )
            == 0
        )
        assert calls == [("20240101", "20240131")]

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_data_fetch_margin_detail(mp)


def test_runs_config_report_path_suite(tmp_path, capsys):
    """test_report_path_prints_stable_run_report_path；test_ops_validate_config_prints_effective_config；test_runs_list_reads_experiment_index"""
    # -- 原 test_report_path_prints_stable_run_report_path --
    def _section_0_test_report_path_prints_stable_run_report_path(tmp_path, mp, capsys):
        from factorzen.cli import main as cli

        run_dir = tmp_path / "workspace" / "factor_evaluations" / "run-1"
        run_dir.mkdir(parents=True)
        report = run_dir / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        mp.setattr(
            "factorzen.experiments.run_paths.FACTOR_EVALUATIONS_DIR",
            tmp_path / "workspace" / "factor_evaluations",
        )

        assert cli.main(["report", "path", "run-1"]) == 0

        assert capsys.readouterr().out.strip() == str(report)

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_report_path_prints_stable_run_report_path(_tp0, mp, capsys)

    # -- 原 test_config_validate → ops validate-config --
    def _section_1_test_ops_validate_config_prints_effective_config(tmp_path, mp, capsys):
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
        mp.setattr(cli, "ROOT", tmp_path)

        assert cli.main(["ops", "validate-config", str(config)]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["config"]["benchmark"] == "000905.SH"
        assert payload["output_dir"].endswith("workspace/factor_evaluations/<run_id>")

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_ops_validate_config_prints_effective_config(_tp1, mp, capsys)

    # -- 原 test_runs_list_reads_experiment_index --
    def _section_2_test_runs_list_reads_experiment_index(tmp_path, mp, capsys):
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
        mp.setattr(cli, "FACTOR_EVALUATIONS_DIR", root)

        assert cli.main(["runs", "list"]) == 0

        out = capsys.readouterr().out
        assert "run-1" in out
        assert "momentum_20d" in out
        assert "success" in out

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_runs_list_reads_experiment_index(_tp2, mp, capsys)

    # runs show 已删（B1 CLI prune）


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

def test_cli_market_parser_suite(capsys):
    """test_mine_search_market_default_ashare；test_mine_search_market_crypto；test_export_alpha_market_crypto；test_validate_overfit_market_crypto；test_validate_overfit_ashare_positional_unchanged；test_portfolio_build_market_crypto；test_portfolio_build_default_ashare；test_sim_run_market_crypto；test_sim_run_default_ashare；test_freq_parsed_for_crypto_and_defaults_daily；test_data_crypto_backfill_parser；test_ashare_rejects_intraday_freq"""
    # -- 原 test_mine_search_market_default_ashare --
    def _section_0_test_mine_search_market_default_ashare():
        p = build_parser()
        args = p.parse_args(["mine", "search", "--start", "20240101", "--end", "20240201"])
        assert args.market == "ashare"
        assert args.func is _cmd_mine_search

    _section_0_test_mine_search_market_default_ashare()

    # -- 原 test_mine_search_market_crypto --
    def _section_1_test_mine_search_market_crypto():
        p = build_parser()
        args = p.parse_args([
            "mine", "search", "--start", "20240101", "--end", "20240201",
            "--market", "crypto", "--top-n", "30",
        ])
        assert args.market == "crypto"
        assert args.top_n == 30
        assert args.func is _cmd_mine_search

    _section_1_test_mine_search_market_crypto()

    # -- 原 test_export_alpha_market_crypto --
    def _section_2_test_export_alpha_market_crypto():
        p = build_parser()
        args = p.parse_args([
            "mine", "export-alpha", "--session", "s", "--date", "20240201",
            "--out", "o.parquet", "--market", "crypto",
        ])
        assert args.market == "crypto"
        assert args.func is _cmd_mine_export_alpha

    _section_2_test_export_alpha_market_crypto()

    # -- 原 test_validate_overfit_market_crypto --
    def _section_3_test_validate_overfit_market_crypto():
        p = build_parser()
        args = p.parse_args([
            "validate", "overfit", "--start", "20240101", "--end", "20240201",
            "--market", "crypto", "--expression", "ts_mean(ret_1d, 5)",
        ])
        assert args.market == "crypto"
        assert args.expression == "ts_mean(ret_1d, 5)"
        assert args.factor is None  # crypto 不用 positional factor
        assert args.func is _cmd_validate_overfit

    _section_3_test_validate_overfit_market_crypto()

    # -- 原 test_validate_overfit_ashare_positional_unchanged --
    def _section_4_test_validate_overfit_ashare_positional_unchanged():
        p = build_parser()
        args = p.parse_args(["validate", "overfit", "momentum_12_1",
                             "--start", "20230101", "--end", "20240101"])
        assert args.market == "ashare"
        assert args.factor == "momentum_12_1"

    _section_4_test_validate_overfit_ashare_positional_unchanged()

    # -- 原 test_portfolio_build_market_crypto --
    def _section_5_test_portfolio_build_market_crypto():
        p = build_parser()
        args = p.parse_args([
            "portfolio", "build", "--start", "20240101", "--end", "20240224",
            "--alpha-file", "a.parquet", "--market", "crypto", "--gross-limit", "1.5",
        ])
        assert args.market == "crypto"
        assert args.gross_limit == 1.5
        assert args.func is _cmd_portfolio_build

    _section_5_test_portfolio_build_market_crypto()

    # -- 原 test_portfolio_build_default_ashare --
    def _section_6_test_portfolio_build_default_ashare():
        p = build_parser()
        args = p.parse_args([
            "portfolio", "build", "--start", "20240101", "--end", "20240224",
            "--alpha-file", "a.parquet",
        ])
        assert args.market == "ashare"

    _section_6_test_portfolio_build_default_ashare()

    # -- 原 test_sim_run_market_crypto --
    def _section_7_test_sim_run_market_crypto():
        p = build_parser()
        args = p.parse_args([
            "sim", "run", "--portfolio-dir", "d", "--start", "20240201", "--end", "20240224",
            "--market", "crypto",
        ])
        assert args.market == "crypto"
        assert args.func is _cmd_sim_run

    _section_7_test_sim_run_market_crypto()

    # -- 原 test_sim_run_default_ashare --
    def _section_8_test_sim_run_default_ashare():
        p = build_parser()
        args = p.parse_args([
            "sim", "run", "--portfolio-dir", "d", "--start", "20240201", "--end", "20240224",
        ])
        assert args.market == "ashare"

    _section_8_test_sim_run_default_ashare()

    # -- 原 test_freq_parsed_for_crypto_and_defaults_daily --
    def _section_9_test_freq_parsed_for_crypto_and_defaults_daily():
        p = build_parser()
        a = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502",
                          "--market", "crypto", "--freq", "15m"])
        assert a.freq == "15m"
        b = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502"])
        assert b.freq == "daily"  # 默认 daily,ashare 零回归

    _section_9_test_freq_parsed_for_crypto_and_defaults_daily()

    # -- 原 test_data_crypto_backfill_parser --
    def _section_10_test_data_crypto_backfill_parser():
        from factorzen.cli.main import _cmd_data_crypto_backfill
        p = build_parser()
        a = p.parse_args(["data", "crypto", "backfill", "--start", "20260501", "--end", "20260502",
                          "--symbols", "BTCUSDT,ETHUSDT", "--lake-root", "/tmp/lk"])
        assert a.func is _cmd_data_crypto_backfill
        assert a.symbols == "BTCUSDT,ETHUSDT" and a.start == "20260501"

    _section_10_test_data_crypto_backfill_parser()

    # -- 原 test_ashare_rejects_intraday_freq --
    def _section_11_test_ashare_rejects_intraday_freq(capsys):
        from factorzen.cli.main import _cmd_mine_search
        p = build_parser()
        a = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502",
                          "--freq", "15m"])  # market 默认 ashare
        assert _cmd_mine_search(a) == 2
        assert "仅 crypto" in capsys.readouterr().err

    _section_11_test_ashare_rejects_intraday_freq(capsys)


# ==== 来自 test_workspace_layout.py ====

def test_workspace_factor_new_suite(tmp_path):
    """test_run_artifacts_are_copied_with_stable_names；test_run_dir_uses_factor_evaluations_folder；test_fz_factor_new_writes_to_workspace；test_fz_factor_new_accepts_frequency_alias"""
    # -- 原 test_run_artifacts_are_copied_with_stable_names --
    def _section_0_test_run_artifacts_are_copied_with_stable_names(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_run_artifacts_are_copied_with_stable_names(_tp0)

    # -- 原 test_run_dir_uses_factor_evaluations_folder --
    def _section_1_test_run_dir_uses_factor_evaluations_folder():
        from factorzen.config.settings import WORKSPACE_DIR
        from factorzen.experiments.run_paths import run_dir

        assert run_dir("momentum_12_1_20260530_031234") == (
            WORKSPACE_DIR / "factor_evaluations" / "momentum_12_1_20260530_031234"
        )

    _section_1_test_run_dir_uses_factor_evaluations_folder()

    # -- 原 test_fz_factor_new_writes_to_workspace --
    def _section_2_test_fz_factor_new_writes_to_workspace(tmp_path, mp):
        from factorzen.cli import main as cli

        mp.setattr(cli, "ROOT", tmp_path)

        assert cli.main(["factor", "new", "my_alpha", "--freq", "daily"]) == 0

        asset = tmp_path / "workspace" / "factor_store" / "ashare" / "my_alpha"
        factor_path = asset / "factor.py"
        assert factor_path.exists()
        text = factor_path.read_text(encoding="utf-8")
        assert 'name = "my_alpha"' in text
        assert "class MyAlphaFactor" in text
        assert (asset / "meta.json").exists()

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_fz_factor_new_writes_to_workspace(_tp2, mp)

    # -- 原 test_fz_factor_new_accepts_frequency_alias --
    def _section_3_test_fz_factor_new_accepts_frequency_alias(tmp_path, mp):
        from factorzen.cli import main as cli

        mp.setattr(cli, "ROOT", tmp_path)

        assert cli.main(["factor", "new", "my_weekly_alpha", "--frequency", "weekly"]) == 0

        asset = tmp_path / "workspace" / "factor_store" / "ashare" / "my_weekly_alpha"
        assert (asset / "factor.py").exists()
        meta = (asset / "meta.json").read_text(encoding="utf-8")
        assert '"frequency": "weekly"' in meta

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_fz_factor_new_accepts_frequency_alias(_tp3, mp)


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


# ==== 来自 test_validation_report_cli.py ====
# ==== 来自 test_validation_cli.py ====


def _install_fake_overfit_pipeline(monkeypatch):
    """monkeypatch `_cmd_validate_overfit` 依赖的每一步，返回按依赖名分组的调用记录。

    `_cmd_validate_overfit` 自己串起 get_factor → FactorDataContext → factor.compute →
    cross_sectional_zscore → DataBundle.build → compute_rank_ic → block_bootstrap_ic_ci →
    deflated_sharpe 这条链（没有单一 pipeline 入口可 monkeypatch），所以逐个依赖打桩，
    只让 rename/select 这类真实 polars 操作跑真的，其余全部替身、离线可跑。
    """
    import polars as pl

    calls: dict[str, list] = {
        "get_factor": [],
        "context": [],
        "get_universe": [],
        "zscore": [],
        "bundle_build": [],
        "compute_rank_ic": [],
        "bootstrap": [],
        "deflated_sharpe": [],
    }

    class FakeFactor:
        lookback_days = 45

        def compute(self, ctx):
            return "FAKE_FACTOR_DF"

    def fake_get_factor(name):
        calls["get_factor"].append(name)
        return FakeFactor

    def fake_get_universe(date_str, universe_name):
        calls["get_universe"].append((date_str, universe_name))
        return pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})

    class FakeDaily:
        def collect(self):
            return "FAKE_COLLECTED_DAILY"

    class FakeContext:
        def __init__(self, *, start, end, required_data, lookback_days, universe):
            calls["context"].append(
                {
                    "start": start,
                    "end": end,
                    "required_data": required_data,
                    "lookback_days": lookback_days,
                    "universe": universe,
                }
            )
            self.daily = FakeDaily()

    def fake_cross_sectional_zscore(fdf, col):
        calls["zscore"].append((fdf, col))
        return pl.DataFrame(
            {
                "trade_date": ["20230101", "20230102"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value_z": [0.1, -0.1],
            }
        )

    class FakeBundle:
        fwd_returns = "FAKE_FWD_RETURNS"

    class FakeDataBundle:
        @staticmethod
        def build(daily, train_ratio):
            calls["bundle_build"].append((daily, train_ratio))
            return FakeBundle()

    class FakeIcResult:
        ic_mean = 0.056
        ir = 1.234
        ic_series = pl.DataFrame({"ic": [0.05, 0.06, 0.07]})

    def fake_compute_rank_ic(factor_df, fwd_returns, *, factor_col, frequency):
        calls["compute_rank_ic"].append(
            {
                "factor_df": factor_df,
                "fwd_returns": fwd_returns,
                "factor_col": factor_col,
                "frequency": frequency,
            }
        )
        return FakeIcResult()

    def fake_block_bootstrap_ic_ci(ic_vals):
        calls["bootstrap"].append(list(ic_vals))
        return (-0.01, 0.09)

    def fake_deflated_sharpe(ir, n_trials, n_obs, skew=0.0, kurt=3.0, sharpe_variance=None):
        calls["deflated_sharpe"].append({"ir": ir, "n_trials": n_trials, "n_obs": n_obs})
        return (2.5, 0.012)

    monkeypatch.setattr("factorzen.daily.factors.registry.get_factor", fake_get_factor)
    monkeypatch.setattr("factorzen.daily.data.context.FactorDataContext", FakeContext)
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    # 合并后 validate 走 scoring.ic_overfit_report(crypto 共用重构),它在 scoring 顶层
    # import 了这两个函数,故须 patch scoring 命名空间(patch 源模块对已绑定名字无效)。
    monkeypatch.setattr(
        "factorzen.discovery.scoring.cross_sectional_zscore",
        fake_cross_sectional_zscore,
    )
    monkeypatch.setattr("factorzen.discovery.scoring.DataBundle", FakeDataBundle)
    monkeypatch.setattr(
        "factorzen.discovery.scoring.compute_rank_ic", fake_compute_rank_ic
    )
    monkeypatch.setattr(
        "factorzen.validation.bootstrap.block_bootstrap_ic_ci", fake_block_bootstrap_ic_ci
    )
    # ic_overfit_report 经 guardrails.deflated_pvalue 调用 deflated_sharpe（两条挖掘路径的
    # DSR 唯一入口，见 test_deflation_recipe_parity 的架构守卫）。guardrails 顶层已绑定该名字，
    # 故须 patch guardrails 命名空间——patch 源模块对已绑定名字无效（此前碰巧生效只因
    # guardrails 尚未被导入，依赖导入时序，脆弱）。
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.deflated_sharpe", fake_deflated_sharpe
    )
    return calls


def test_validate_overfit_cmd_suite(capsys):
    """`fz validate overfit`（无 --universe）应把 factor/start/end 转发到底层每一步。；`fz validate overfit --universe` 应查询 get_universe 并把股票列表转发进 context。；fz validate overfit 不给 factor → 返回 2 + 友好提示，而非裸 KeyError traceback。"""
    # -- 原 test_cmd_validate_overfit_forwards_args_and_prints_metrics --
    def _section_0_test_cmd_validate_overfit_forwards_args_and_prints_metrics(mp, capsys):
        from factorzen.cli import main as cli

        calls = _install_fake_overfit_pipeline(mp)

        rc = cli.main(
            ["validate", "overfit", "momentum_12_1", "--start", "20230101", "--end", "20230601"]
        )

        assert rc == 0
        assert calls["get_factor"] == ["momentum_12_1"]
        assert calls["get_universe"] == []  # 未传 --universe，不应查询股票池
        assert calls["context"] == [
            {
                "start": "20230101",
                "end": "20230601",
                "required_data": ["daily", "daily_basic"],
                "lookback_days": 45,  # 取自 FakeFactor.lookback_days，而非硬编码的 60
                "universe": None,
            }
        ]
        assert calls["bundle_build"] == [("FAKE_COLLECTED_DAILY", 1.0)]
        assert calls["zscore"] == [("FAKE_FACTOR_DF", "factor_value")]
        assert len(calls["compute_rank_ic"]) == 1
        ic_call = calls["compute_rank_ic"][0]
        assert ic_call["fwd_returns"] == "FAKE_FWD_RETURNS"
        assert ic_call["factor_col"] == "factor_clean"
        assert ic_call["frequency"] == "daily"
        assert ic_call["factor_df"].columns == ["trade_date", "ts_code", "factor_clean"]
        assert calls["bootstrap"] == [[0.05, 0.06, 0.07]]
        assert calls["deflated_sharpe"] == [{"ir": 1.234, "n_trials": 1, "n_obs": 3}]

        out = capsys.readouterr().out.splitlines()
        assert out[0] == (
            "[validate] momentum_12_1: IC=0.0560 IR=1.2340 "
            "DSR_p=0.0120 IC_95%CI=[-0.0100,0.0900]"
        )
        assert out[1] == "[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。"

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cmd_validate_overfit_forwards_args_and_prints_metrics(mp, capsys)

    # -- 原 test_cmd_validate_overfit_resolves_universe_into_context --
    def _section_1_test_cmd_validate_overfit_resolves_universe_into_context(mp):
        from factorzen.cli import main as cli

        calls = _install_fake_overfit_pipeline(mp)

        rc = cli.main(
            [
                "validate",
                "overfit",
                "momentum_12_1",
                "--start",
                "20230101",
                "--end",
                "20230601",
                "--universe",
                "csi500",
            ]
        )

        assert rc == 0
        assert calls["get_universe"] == [("20230601", "csi500")]
        assert calls["context"][0]["universe"] == ["000001.SZ", "000002.SZ"]

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_validate_overfit_resolves_universe_into_context(mp)

    # -- 原 test_validate_overfit_missing_factor_friendly_error --
    def _section_2_test_validate_overfit_missing_factor_friendly_error(capsys):
        from factorzen.cli.main import main

        rc = main(["validate", "overfit", "--start", "20240101", "--end", "20241231"])
        assert rc == 2
        assert "缺少因子名" in capsys.readouterr().err

    _section_2_test_validate_overfit_missing_factor_friendly_error(capsys)


# ==== 来自 test_report_cli.py ====


def test_parser_report_portfolio_suite():
    """--sim-dir 正确映射到 args.sim_dir。；--portfolio-dir 正确映射到 args.portfolio_dir（可选）。；--out 未指定时 args.out 为 None（handler 负责生成默认路径）。；--out 明确指定时 args.out 等于该值。"""
    # -- 原 test_parser_report_portfolio_sim_dir --
    def _section_0_test_parser_report_portfolio_sim_dir():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "report",
                "portfolio",
                "--sim-dir",
                "workspace/sim/myrun",
            ]
        )
        assert args.sim_dir == "workspace/sim/myrun"

    _section_0_test_parser_report_portfolio_sim_dir()

    # -- 原 test_parser_report_portfolio_portfolio_dir --
    def _section_1_test_parser_report_portfolio_portfolio_dir():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "report",
                "portfolio",
                "--sim-dir",
                "workspace/sim/run-001",
                "--portfolio-dir",
                "workspace/portfolios/run-001",
            ]
        )
        assert args.portfolio_dir == "workspace/portfolios/run-001"

    _section_1_test_parser_report_portfolio_portfolio_dir()

    # -- 原 test_parser_report_portfolio_out_default_is_none --
    def _section_2_test_parser_report_portfolio_out_default_is_none():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "report",
                "portfolio",
                "--sim-dir",
                "workspace/sim/run-001",
            ]
        )
        assert args.out is None

    _section_2_test_parser_report_portfolio_out_default_is_none()

    # -- 原 test_parser_report_portfolio_out_explicit --
    def _section_3_test_parser_report_portfolio_out_explicit():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args(
            [
                "report",
                "portfolio",
                "--sim-dir",
                "workspace/sim/run-001",
                "--out",
                "workspace/reports/my_report.html",
            ]
        )
        assert args.out == "workspace/reports/my_report.html"

    _section_3_test_parser_report_portfolio_out_explicit()


def test_cmd_report_portfolio_renders_nav_chart_from_parquet(tmp_path: Path) -> None:
    """Fix 3: sim_dir 含 nav.parquet 时，_cmd_report_portfolio 应在 HTML 中渲染净值图。"""
    import polars as pl

    from factorzen.cli.main import _cmd_report_portfolio

    # 建 sim_dir / nav.parquet
    sim_dir = tmp_path / "sim_out"
    sim_dir.mkdir()
    nav_df = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "nav": [1.0, 1.01, 1.02],
            "gross_return": [0.0, 0.01, 0.01],
            "cost": [0.0, 0.0, 0.0],
            "borrow_cost": [0.0, 0.0, 0.0],
            "net_return": [0.0, 0.01, 0.01],
            "cash_weight": [1.0, 0.0, 0.0],
        }
    )
    nav_df.write_parquet(sim_dir / "nav.parquet")
    (sim_dir / "metrics.json").write_text(
        json.dumps({
            "ann_ret": 0.1,
            "ann_vol": 0.15,
            "sharpe": 0.8,
            "max_dd": -0.05,
            "ann_turnover": 2.0,
            "total_cost": 0.005,
        }),
        encoding="utf-8",
    )

    out_html = tmp_path / "portfolio_out.html"
    args = argparse.Namespace(
        sim_dir=str(sim_dir),
        portfolio_dir=None,
        out=str(out_html),
    )
    ret = _cmd_report_portfolio(args)
    assert ret == 0
    html = out_html.read_text(encoding="utf-8")
    assert "data:image/png;base64" in html, (
        "sim_dir 含 nav.parquet 时，report 应渲染净值图；未找到 base64 图表"
    )

    # 修复4：_cmd_report_portfolio 重建的 sim_result 此前只设置 .nav、未设置
    # .returns，导致 _make_monthly_return_heatmap（只读 .returns）在生产路径下
    # 恒返回 None，月度收益热力图永不渲染（死代码）。同一份 nav_df 已含
    # net_return 列，足够同时驱动两张图。
    assert "月度收益热力图" in html, (
        "月度收益热力图应被渲染进 HTML；sim_result.returns 未设置会导致该图永远是"
        "死代码（'暂无数据'而非真正接通）"
    )
    assert html.count("data:image/png;base64") >= 2, (
        f"应同时渲染净值曲线图与月度收益热力图两张 base64 图表，实际出现"
        f" {html.count('data:image/png;base64')} 次"
    )


# ==== 来自 test_ops_cli_smoke.py ====

def _write_cfg(tmp_path, state_dir=None):
    p = tmp_path / "ops.yaml"
    sd = state_dir or (tmp_path / "state")
    p.write_text(
        f"session_dir: s\nportfolio_run_dirs_glob: g\nstate_dir: {sd}\n",
        encoding="utf-8",
    )
    return p


def test_ops_cli_suite(tmp_path, capsys):
    """test_fz_ops_daily_dispatches_with_date；test_fz_ops_daily_defaults_to_today；test_fz_ops_daily_propagates_return_code；test_fz_ops_status_prints_summary"""
    # -- 原 test_fz_ops_daily_dispatches_with_date --
    def _section_0_test_fz_ops_daily_dispatches_with_date(mp, tmp_path):
        p = _write_cfg(tmp_path)
        seen: dict = {}

        def fake_run(cfg, as_of, notifier=None):
            seen["as_of"] = as_of
            seen["session_dir"] = cfg.session_dir
            return 0

        mp.setattr("factorzen.ops.runner.run_ops_daily", fake_run)
        rc = main(["ops", "daily", "--config", str(p), "--date", "20260720"])
        assert rc == 0
        assert seen["as_of"] == date(2026, 7, 20)
        assert seen["session_dir"] == "s"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_fz_ops_daily_dispatches_with_date(mp, _tp0)

    # -- 原 test_fz_ops_daily_defaults_to_today --
    def _section_1_test_fz_ops_daily_defaults_to_today(mp, tmp_path):
        p = _write_cfg(tmp_path)
        seen: dict = {}

        def fake_run(cfg, as_of, notifier=None):
            seen["as_of"] = as_of
            return 0

        mp.setattr("factorzen.ops.runner.run_ops_daily", fake_run)
        rc = main(["ops", "daily", "--config", str(p)])
        assert rc == 0
        assert seen["as_of"] == date.today()

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_fz_ops_daily_defaults_to_today(mp, _tp1)

    # -- 原 test_fz_ops_daily_propagates_return_code --
    def _section_2_test_fz_ops_daily_propagates_return_code(mp, tmp_path):
        p = _write_cfg(tmp_path)
        mp.setattr(
            "factorzen.ops.runner.run_ops_daily", lambda cfg, as_of, notifier=None: 1
        )
        rc = main(["ops", "daily", "--config", str(p), "--date", "20260720"])
        assert rc == 1

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_fz_ops_daily_propagates_return_code(mp, _tp2)

    # -- 原 test_fz_ops_status_prints_summary --
    def _section_3_test_fz_ops_status_prints_summary(tmp_path, capsys):
        sd = tmp_path / "state"
        p = _write_cfg(tmp_path, state_dir=sd)
        OpsState(sd, date(2026, 7, 20)).mark_done("guard", detail="ok")
        rc = main(["ops", "status", "--config", str(p), "--date", "20260720"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "guard" in out and "done" in out

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_fz_ops_status_prints_summary(_tp3, capsys)


