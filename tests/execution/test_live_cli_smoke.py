"""fz live init/step/status/report：CLI 路由 + 离线端到端 smoke。

- test_init_step_report_pipeline：底层 driver/attribution 直接跑通(init→多日
  step→report)，断言 attribution.json 落盘 + 关键字段。
- test_live_cli_parser_routes_new_subcommands：确认新增 4 子命令能被
  build_parser() 解析并挂到正确的 func，且不干扰既有 replay/其他顶层命令。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.attribution import build_attribution_report
from factorzen.execution.drivers import run_daily_step
from factorzen.execution.store import SessionStore


def test_init_step_report_pipeline(tmp_path: Path) -> None:
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    daily = pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": "A.SZ",
                "open": 10.1,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            }
            for d in dates
        ]
    )
    pf = tmp_path / "pf"
    pf.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [0.5]}).write_parquet(
        pf / "weights.parquet"
    )
    (pf / "manifest.json").write_text(
        json.dumps({"signal_date": dates[0].isoformat(), "status": "optimal"})
    )
    cfg = {"initial_cash": 1_000_000.0, "slippage_bps": 0.0}
    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", **cfg})
    for d in dates:
        run_daily_step(sess, d, [str(pf)], daily, config=cfg)
    rep = build_attribution_report(sess, [str(pf)], daily, initial_cash=1_000_000.0)
    assert (sess / "attribution.json").exists()
    assert rep["n_days"] == 3
    assert "cost_bps" in rep and "residual_bps" in rep


def test_live_cli_parser_routes_new_subcommands() -> None:
    from factorzen.cli.main import (
        _cmd_live_init,
        _cmd_live_report,
        _cmd_live_status,
        _cmd_live_step,
        build_parser,
    )

    parser = build_parser()

    init_args = parser.parse_args(
        ["live", "init", "--session-dir", "workspace/execution/sess1"]
    )
    assert init_args.func is _cmd_live_init
    assert init_args.session_dir == "workspace/execution/sess1"
    assert init_args.initial_cash == 1_000_000.0
    assert init_args.broker == "paper"

    step_args = parser.parse_args(
        [
            "live",
            "step",
            "--session-dir",
            "workspace/execution/sess1",
            "--date",
            "20260105",
            "--portfolio-run-dir",
            "workspace/portfolios/run1",
            "--start",
            "20251201",
            "--end",
            "20260105",
        ]
    )
    assert step_args.func is _cmd_live_step
    assert step_args.portfolio_run_dirs == ["workspace/portfolios/run1"]

    status_args = parser.parse_args(
        ["live", "status", "--session-dir", "workspace/execution/sess1"]
    )
    assert status_args.func is _cmd_live_status

    report_args = parser.parse_args(
        [
            "live",
            "report",
            "--session-dir",
            "workspace/execution/sess1",
            "--portfolio-run-dir",
            "workspace/portfolios/run1",
            "--start",
            "20251201",
            "--end",
            "20260105",
        ]
    )
    assert report_args.func is _cmd_live_report

    # replay(M1)与其余顶层命令不受影响，仍可正常解析。
    replay_args = parser.parse_args(
        [
            "live",
            "replay",
            "--session-dir",
            "workspace/execution/sess1",
            "--portfolio-run-dir",
            "workspace/portfolios/run1",
            "--start",
            "20251201",
            "--end",
            "20260105",
        ]
    )
    assert replay_args.broker == "paper"

    sim_show_args = parser.parse_args(["sim", "show", "--sim-dir", "workspace/sim/run1"])
    assert sim_show_args.sim_dir == "workspace/sim/run1"
