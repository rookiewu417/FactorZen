"""MC1 T7: fz mine search/export-alpha 的 --market 参数（默认 ashare 不变）。"""
from __future__ import annotations

from factorzen.cli.main import (
    _cmd_mine_export_alpha,
    _cmd_mine_search,
    _cmd_portfolio_build,
    _cmd_validate_overfit,
    build_parser,
)


def test_mine_search_market_default_ashare():
    p = build_parser()
    args = p.parse_args(["mine", "search", "--start", "20240101", "--end", "20240201"])
    assert args.market == "ashare"
    assert args.func is _cmd_mine_search


def test_mine_search_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "mine", "search", "--start", "20240101", "--end", "20240201",
        "--market", "crypto", "--top-n", "30",
    ])
    assert args.market == "crypto"
    assert args.top_n == 30
    assert args.func is _cmd_mine_search


def test_export_alpha_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "mine", "export-alpha", "--session", "s", "--date", "20240201",
        "--out", "o.parquet", "--market", "crypto",
    ])
    assert args.market == "crypto"
    assert args.func is _cmd_mine_export_alpha


def test_validate_overfit_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "validate", "overfit", "--start", "20240101", "--end", "20240201",
        "--market", "crypto", "--expression", "ts_mean(ret_1d, 5)",
    ])
    assert args.market == "crypto"
    assert args.expression == "ts_mean(ret_1d, 5)"
    assert args.factor is None  # crypto 不用 positional factor
    assert args.func is _cmd_validate_overfit


def test_validate_overfit_ashare_positional_unchanged():
    p = build_parser()
    args = p.parse_args(["validate", "overfit", "momentum_12_1",
                         "--start", "20230101", "--end", "20240101"])
    assert args.market == "ashare"
    assert args.factor == "momentum_12_1"


def test_portfolio_build_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "portfolio", "build", "--start", "20240101", "--end", "20240224",
        "--alpha-file", "a.parquet", "--market", "crypto", "--gross-limit", "1.5",
    ])
    assert args.market == "crypto"
    assert args.gross_limit == 1.5
    assert args.func is _cmd_portfolio_build


def test_portfolio_build_default_ashare():
    p = build_parser()
    args = p.parse_args([
        "portfolio", "build", "--start", "20240101", "--end", "20240224",
        "--alpha-file", "a.parquet",
    ])
    assert args.market == "ashare"
