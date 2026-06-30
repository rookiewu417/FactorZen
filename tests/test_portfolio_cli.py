"""Tests for `fz portfolio build` CLI parser (Task 6)."""

from __future__ import annotations


def test_parser_has_portfolio_build():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "portfolio",
            "build",
            "--start",
            "20230101",
            "--end",
            "20241231",
            "--universe",
            "csi500",
            "--alpha-file",
            "a.parquet",
            "--lam",
            "2.0",
        ]
    )
    assert args.command == "portfolio"
    assert args.portfolio_command == "build"
    assert args.alpha_file == "a.parquet"
    assert args.lam == 2.0
    assert callable(args.func)
