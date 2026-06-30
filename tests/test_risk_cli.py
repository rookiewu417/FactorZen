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
    # 默认值断言（dest 别名 cov_half_life/nw_lags + type=int 转换的易错点）
    assert args.cov_half_life == 90
    assert args.nw_lags == 2
