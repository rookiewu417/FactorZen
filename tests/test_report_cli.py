"""Tests for `fz report portfolio` CLI parser (Task 5)."""

from __future__ import annotations


def test_parser_has_report_portfolio():
    """report portfolio 子命令已注册，attrs: command=report / report_command=portfolio / callable func。"""
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
    assert args.command == "report"
    assert args.report_command == "portfolio"
    assert callable(args.func)


def test_parser_report_portfolio_sim_dir():
    """--sim-dir 正确映射到 args.sim_dir。"""
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


def test_parser_report_portfolio_portfolio_dir():
    """--portfolio-dir 正确映射到 args.portfolio_dir（可选）。"""
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


def test_parser_report_portfolio_out_default_is_none():
    """--out 未指定时 args.out 为 None（handler 负责生成默认路径）。"""
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


def test_parser_report_portfolio_out_explicit():
    """--out 明确指定时 args.out 等于该值。"""
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
