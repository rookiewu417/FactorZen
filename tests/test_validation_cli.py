# tests/test_validation_cli.py
"""Tests for `fz validate overfit` CLI command."""


def test_parser_has_validate_overfit():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["validate", "overfit", "momentum_12_1", "--start", "20230101", "--end", "20240101"]
    )
    assert args.command == "validate"
    assert args.validate_command == "overfit"
    assert args.factor == "momentum_12_1"
    assert callable(args.func)
