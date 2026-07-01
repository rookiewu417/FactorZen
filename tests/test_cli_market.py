"""MC1 T7: fz mine search/export-alpha 的 --market 参数（默认 ashare 不变）。"""
from __future__ import annotations

from factorzen.cli.main import (
    _cmd_mine_export_alpha,
    _cmd_mine_search,
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
