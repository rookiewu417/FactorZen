# tests/test_agent_cli.py
"""Parser smoke tests for `fz mine agent` CLI subcommand."""
from __future__ import annotations


def test_parser_has_mine_agent():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231",
         "--iterations", "5", "--seed", "42"]
    )
    assert args.command == "mine"
    assert args.mine_command == "agent"
    assert args.start == "20220101"
    assert args.end == "20231231"
    assert args.iterations == 5
    assert args.seed == 42
    assert callable(args.func)


def test_parser_mine_agent_defaults():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231"]
    )
    assert args.iterations == 5
    assert args.top_k == 5
    assert args.seed == 42
    assert args.human_review is False
    assert args.universe is None


def test_parser_mine_agent_human_review():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231", "--human-review"]
    )
    assert args.human_review is True
