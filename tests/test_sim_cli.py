"""Tests for `fz sim run / sim show` CLI parser (Task 3)."""

from __future__ import annotations


def test_parser_has_sim_run():
    """sim run subcommand is registered with correct dest attrs and a callable handler."""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "sim",
            "run",
            "--portfolio-dir",
            "workspace/portfolios",
            "--start",
            "20230101",
            "--end",
            "20241231",
        ]
    )
    assert args.command == "sim"
    assert args.sim_command == "run"
    assert callable(args.func)


def test_parser_sim_run_optional_run_id():
    """sim run accepts optional --run-id; defaults to None when omitted."""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "sim",
            "run",
            "--portfolio-dir",
            "workspace/portfolios",
            "--start",
            "20230101",
            "--end",
            "20241231",
            "--run-id",
            "test-run-001",
        ]
    )
    assert args.run_id == "test-run-001"

    args_no_id = p.parse_args(
        [
            "sim",
            "run",
            "--portfolio-dir",
            "workspace/portfolios",
            "--start",
            "20230101",
            "--end",
            "20241231",
        ]
    )
    assert args_no_id.run_id is None


def test_parser_has_sim_show():
    """sim show subcommand is registered with --sim-dir and a callable handler."""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "sim",
            "show",
            "--sim-dir",
            "workspace/sim/run-001",
        ]
    )
    assert args.command == "sim"
    assert args.sim_command == "show"
    assert callable(args.func)
    assert args.sim_dir == "workspace/sim/run-001"
