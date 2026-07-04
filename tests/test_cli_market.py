"""MC1 T7: fz mine search/export-alpha 的 --market 参数（默认 ashare 不变）。"""
from __future__ import annotations

from factorzen.cli.main import (
    _cmd_mine_export_alpha,
    _cmd_mine_search,
    _cmd_portfolio_build,
    _cmd_sim_run,
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


def test_sim_run_market_crypto():
    p = build_parser()
    args = p.parse_args([
        "sim", "run", "--portfolio-dir", "d", "--start", "20240201", "--end", "20240224",
        "--market", "crypto",
    ])
    assert args.market == "crypto"
    assert args.func is _cmd_sim_run


def test_sim_run_default_ashare():
    p = build_parser()
    args = p.parse_args([
        "sim", "run", "--portfolio-dir", "d", "--start", "20240201", "--end", "20240224",
    ])
    assert args.market == "ashare"


def test_freq_parsed_for_crypto_and_defaults_daily():
    p = build_parser()
    a = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502",
                      "--market", "crypto", "--freq", "15m"])
    assert a.freq == "15m"
    b = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502"])
    assert b.freq == "daily"  # 默认 daily,ashare 零回归


def test_data_crypto_backfill_parser():
    from factorzen.cli.main import _cmd_data_crypto_backfill
    p = build_parser()
    a = p.parse_args(["data", "crypto", "backfill", "--start", "20260501", "--end", "20260502",
                      "--symbols", "BTCUSDT,ETHUSDT", "--lake-root", "/tmp/lk"])
    assert a.func is _cmd_data_crypto_backfill
    assert a.symbols == "BTCUSDT,ETHUSDT" and a.start == "20260501"


def test_ashare_rejects_intraday_freq(capsys):
    from factorzen.cli.main import _cmd_mine_search
    p = build_parser()
    a = p.parse_args(["mine", "search", "--start", "20260501", "--end", "20260502",
                      "--freq", "15m"])  # market 默认 ashare
    assert _cmd_mine_search(a) == 2
    assert "仅 crypto" in capsys.readouterr().err
