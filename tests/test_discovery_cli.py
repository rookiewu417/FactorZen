from __future__ import annotations


def test_parser_has_mine_search():
    from factorzen.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mine", "search", "--start", "20240101", "--end", "20240601"])
    assert args.command == "mine"
    assert args.mine_command == "search"
    assert args.start == "20240101"
    assert callable(args.func)


def test_parser_has_mine_leaderboard():
    from factorzen.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mine", "leaderboard", "some/dir"])
    assert args.mine_command == "leaderboard"
    assert args.session_dir == "some/dir"


def test_leaderboard_prints_csv(tmp_path, capsys):
    import argparse

    from factorzen.cli.main import _cmd_mine_leaderboard
    (tmp_path / "candidates.csv").write_text("rank,expression\n1,rank(close)\n")
    rc = _cmd_mine_leaderboard(argparse.Namespace(session_dir=str(tmp_path)))
    assert rc == 0
    assert "rank(close)" in capsys.readouterr().out
