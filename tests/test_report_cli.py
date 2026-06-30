"""Tests for `fz report portfolio` CLI parser (Task 5)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def test_parser_has_report_portfolio():
    """report portfolio 子命令已注册，attrs: command=report / report_command=portfolio / callable func。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "report",
            "portfolio",
            "--sim-dir",
            "workspace/sim/run-001",
        ]
    )
    assert args.command == "report"
    assert args.report_command == "portfolio"
    assert callable(args.func)


def test_parser_report_portfolio_sim_dir():
    """--sim-dir 正确映射到 args.sim_dir。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "report",
            "portfolio",
            "--sim-dir",
            "workspace/sim/myrun",
        ]
    )
    assert args.sim_dir == "workspace/sim/myrun"


def test_parser_report_portfolio_portfolio_dir():
    """--portfolio-dir 正确映射到 args.portfolio_dir（可选）。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "report",
            "portfolio",
            "--sim-dir",
            "workspace/sim/run-001",
            "--portfolio-dir",
            "workspace/portfolios/run-001",
        ]
    )
    assert args.portfolio_dir == "workspace/portfolios/run-001"


def test_parser_report_portfolio_out_default_is_none():
    """--out 未指定时 args.out 为 None（handler 负责生成默认路径）。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "report",
            "portfolio",
            "--sim-dir",
            "workspace/sim/run-001",
        ]
    )
    assert args.out is None


def test_parser_report_portfolio_out_explicit():
    """--out 明确指定时 args.out 等于该值。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "report",
            "portfolio",
            "--sim-dir",
            "workspace/sim/run-001",
            "--out",
            "workspace/reports/my_report.html",
        ]
    )
    assert args.out == "workspace/reports/my_report.html"


def test_cmd_report_portfolio_renders_nav_chart_from_parquet(tmp_path: Path) -> None:
    """Fix 3: sim_dir 含 nav.parquet 时，_cmd_report_portfolio 应在 HTML 中渲染净值图。"""
    import polars as pl

    from factorzen.cli.main import _cmd_report_portfolio

    # 建 sim_dir / nav.parquet
    sim_dir = tmp_path / "sim_out"
    sim_dir.mkdir()
    nav_df = pl.DataFrame(
        {
            "trade_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "nav": [1.0, 1.01, 1.02],
            "gross_return": [0.0, 0.01, 0.01],
            "cost": [0.0, 0.0, 0.0],
            "borrow_cost": [0.0, 0.0, 0.0],
            "net_return": [0.0, 0.01, 0.01],
            "cash_weight": [1.0, 0.0, 0.0],
        }
    )
    nav_df.write_parquet(sim_dir / "nav.parquet")
    (sim_dir / "metrics.json").write_text(
        json.dumps({
            "ann_ret": 0.1,
            "ann_vol": 0.15,
            "sharpe": 0.8,
            "max_dd": -0.05,
            "ann_turnover": 2.0,
            "total_cost": 0.005,
        }),
        encoding="utf-8",
    )

    out_html = tmp_path / "portfolio_out.html"
    args = argparse.Namespace(
        sim_dir=str(sim_dir),
        portfolio_dir=None,
        out=str(out_html),
    )
    ret = _cmd_report_portfolio(args)
    assert ret == 0
    html = out_html.read_text(encoding="utf-8")
    assert "data:image/png;base64" in html, (
        "sim_dir 含 nav.parquet 时，report 应渲染净值图；未找到 base64 图表"
    )
