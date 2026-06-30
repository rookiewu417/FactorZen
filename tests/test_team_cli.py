"""Parser smoke tests for `fz mine team` CLI subcommand."""

from __future__ import annotations


def test_parser_has_mine_team() -> None:
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "team", "--start", "20220101", "--end", "20231231",
         "--iterations", "5", "--seed", "42"]
    )
    assert args.command == "mine"
    assert args.mine_command == "team"
    assert args.start == "20220101"
    assert args.iterations == 5
    assert args.seed == 42
    assert callable(args.func)


def test_parser_mine_team_index_default() -> None:
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["mine", "team", "--start", "20220101", "--end", "20231231"]
    )
    assert args.index_path.endswith(".jsonl")  # 长期记忆默认路径
