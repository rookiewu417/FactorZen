"""``fz strategies run``：parser 形状 + handler 最小闭环（mock loader）。"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest


def test_parser_strategies_run_suite():
    """help 可打印；旗标 / 子命令 / name choices 存在。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    # help 不抛
    help_txt = p.format_help()
    assert "strategies" in help_txt

    args = p.parse_args(
        [
            "strategies",
            "run",
            "trend_timing",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--universe",
            "csi300",
            "--out-dir",
            "/tmp/strat_out",
            "--run-id",
            "tt1",
            "--set",
            "ma_window=60",
            "--set",
            "top_n=30",
        ]
    )
    assert args.command == "strategies"
    assert args.strategies_command == "run"
    assert args.name == "trend_timing"
    assert args.start == "20230101"
    assert args.end == "20231231"
    assert args.universe == "csi300"
    assert args.out_dir == "/tmp/strat_out"
    assert args.run_id == "tt1"
    assert args.set_overrides == ["ma_window=60", "top_n=30"]
    assert callable(args.func)

    # name 合法 choices
    for name in ("trend_timing", "momentum_rotation", "sleeve", "quantile_group"):
        a = p.parse_args(
            ["strategies", "run", name, "--start", "20230101", "--end", "20230131"]
        )
        assert a.name == name

    # 未知 name 应失败
    with pytest.raises(SystemExit):
        p.parse_args(
            ["strategies", "run", "unknown_strat", "--start", "20230101", "--end", "20230131"]
        )


def test_cmd_strategies_run_trend_timing_mock(tmp_path, monkeypatch, capsys):
    """handler 最小闭环：mock loader + 真实 generate/sim 走合成面板。"""
    from factorzen.cli import main as cli

    n = 25
    start_d = date(2023, 1, 3)
    dates = [start_d + timedelta(days=i) for i in range(n)]
    codes = ["A.SZ", "B.SZ", "C.SZ"]

    daily_rows = []
    for c in codes:
        px = 10.0
        for i, d in enumerate(dates):
            close = 10.0 + i * 0.05
            daily_rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "open": px,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "pre_close": px,
                    "change": close - px,
                    "pct_chg": 0.5,
                    "vol": 1e6,
                    "amount": 1e9,
                }
            )
            px = close
    daily = pl.DataFrame(daily_rows)
    idx = pl.DataFrame(
        {"trade_date": dates, "close": [10.0 + i * 0.5 for i in range(n)]}
    )

    monkeypatch.setattr(
        "factorzen.core.loader.fetch_daily",
        lambda s, e: daily,
    )
    monkeypatch.setattr(
        "factorzen.core.loader.fetch_index_daily",
        lambda code, s, e: idx,
    )
    # 默认 members 会打 Tushare；注入 generate 侧
    monkeypatch.setattr(
        "factorzen.strategies.trend_timing._default_members",
        lambda code, ds: codes,
    )

    out = tmp_path / "strat"
    ret = cli.main(
        [
            "strategies",
            "run",
            "trend_timing",
            "--start",
            "20230103",
            "--end",
            "20230127",
            "--out-dir",
            str(out),
            "--run-id",
            "tt_mock",
            "--set",
            "ma_window=5",
            "--set",
            "top_n=2",
            "--set",
            "rebalance=weekly",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr().out
    assert "[strategies]" in captured
    assert "run_dir=" in captured
    sim_dir = out / "tt_mock" / "sim"
    assert (sim_dir / "nav.parquet").exists()
    assert (sim_dir / "metrics.json").exists()
    # 至少有一期产物
    products = list((out / "tt_mock" / "products").iterdir())
    assert products
