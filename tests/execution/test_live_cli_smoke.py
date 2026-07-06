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


def test_live_status_handles_resumable_state_shape(tmp_path: Path, capsys) -> None:
    # run_daily_step 落的是"可续跑态" broker.state() = {cash: float, pos, order_seq}。
    from factorzen.cli.main import _cmd_live_status, build_parser

    sess = tmp_path / "sess"
    cfg = {"initial_cash": 1_000_000.0}
    SessionStore(sess).init({"broker": "paper", **cfg})
    d0 = date(2026, 1, 5)
    daily = pl.DataFrame(
        [
            {
                "trade_date": d0,
                "ts_code": "A.SZ",
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": 1e9,
            }
        ]
    )
    pf = tmp_path / "pf"
    pf.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [0.5]}).write_parquet(
        pf / "weights.parquet"
    )
    (pf / "manifest.json").write_text(
        json.dumps({"signal_date": d0.isoformat(), "status": "optimal"})
    )
    run_daily_step(sess, d0, [str(pf)], daily, config=cfg)

    args = build_parser().parse_args(["live", "status", "--session-dir", str(sess)])
    rc = _cmd_live_status(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "持仓数=1" in out
    # 现金应是数字（可续跑态 cash 直接是 float），不能把 dict 原样打出来
    cash_field = out.split("现金=")[1].split(" ")[0]
    assert "{" not in cash_field
    float(cash_field)  # 不抛异常即说明是个可解析的数字


def test_live_status_handles_legacy_display_view_state(tmp_path: Path, capsys) -> None:
    # legacy 兼容：旧 session 的 state.json 可能是 step() 的"显示视图"
    # {positions: {...}, cash: {available,total_asset,market_value}}——旧版 run_replay
    # 曾落这种格式（现已改落可续跑态，见 test_replay_state_resumable_by_daily_step），
    # 但历史 session 仍需能读。_cmd_live_status 须解析它、不误报：cash 从 dict 取数值、
    # 持仓从 "positions" 键取（而非可续跑态的 "pos"）。直接构造显示视图以锁死该兼容分支。
    from factorzen.cli.main import _cmd_live_status, build_parser

    sess = tmp_path / "sess"
    SessionStore(sess).init({"broker": "paper", "initial_cash": 1_000_000.0})
    (sess / "state.json").write_text(
        json.dumps(
            {
                "positions": {
                    "A.SZ": {"ts_code": "A.SZ", "volume": 50000,
                             "can_use_volume": 50000, "avg_cost": 10.0}
                },
                "cash": {"available": 500000.0, "total_asset": 1000000.0,
                         "market_value": 500000.0},
            }
        )
    )
    pl.DataFrame(
        {"as_of_date": [date(2026, 1, 5).isoformat()], "nav_after": [1_000_000.0]}
    ).write_parquet(sess / "nav.parquet")

    args = build_parser().parse_args(["live", "status", "--session-dir", str(sess)])
    rc = _cmd_live_status(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "持仓数=1" in out  # 从 "positions" 键取，而非误报 0
    cash_field = out.split("现金=")[1].split(" ")[0]
    assert "{" not in cash_field  # 显示视图 cash 是 dict，应被解成数值而非原样打印
    float(cash_field)
