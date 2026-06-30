"""Parser smoke test for `fz risk build` CLI."""


def test_parser_has_risk_build():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["risk", "build", "--start", "20230101", "--end", "20241231", "--universe", "csi500"]
    )
    assert args.command == "risk"
    assert args.risk_command == "build"
    assert args.start == "20230101"
    assert args.end == "20241231"
    assert args.universe == "csi500"
    assert callable(args.func)
