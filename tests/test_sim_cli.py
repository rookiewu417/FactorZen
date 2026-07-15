"""Tests for `fz sim run / sim show` CLI: parser shape + execution-level forwarding."""

from __future__ import annotations


def test_parser_has_sim_run():
    """sim run subcommand is registered with correct dest attrs and a callable handler."""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "sim",
            "run",
            "--portfolio-dir",
            "workspace/portfolios",
            "--start",
            "20230101",
            "--end",
            "20241231",
        ]
    )
    assert args.command == "sim"
    assert args.sim_command == "run"
    assert callable(args.func)


def test_parser_sim_run_optional_run_id():
    """sim run accepts optional --run-id; defaults to None when omitted."""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "sim",
            "run",
            "--portfolio-dir",
            "workspace/portfolios",
            "--start",
            "20230101",
            "--end",
            "20241231",
            "--run-id",
            "test-run-001",
        ]
    )
    assert args.run_id == "test-run-001"

    args_no_id = p.parse_args(
        [
            "sim",
            "run",
            "--portfolio-dir",
            "workspace/portfolios",
            "--start",
            "20230101",
            "--end",
            "20241231",
        ]
    )
    assert args_no_id.run_id is None


def test_parser_has_sim_show():
    """sim show subcommand is registered with --sim-dir and a callable handler."""
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "sim",
            "show",
            "--sim-dir",
            "workspace/sim/run-001",
        ]
    )
    assert args.command == "sim"
    assert args.sim_command == "show"
    assert callable(args.func)
    assert args.sim_dir == "workspace/sim/run-001"


def test_cmd_sim_run_forwards_filtered_run_dirs_without_explicit_cost_model(
    tmp_path, monkeypatch, capsys
):
    """_cmd_sim_run：只挑有 weights.parquet 的子目录(按路径排序) + out_dir/run_id 正确转发；

    CLI 层不显式传 cost_model，确认 run_portfolio_simulation 拿不到 cost_model kwarg
    （即走该函数自身默认值 CostModel()，非零成本回测，而不是被 CLI 悄悄改写）。
    """
    import polars as pl

    from factorzen.cli import main as cli

    portfolio_root = tmp_path / "portfolios"
    portfolio_root.mkdir()
    for name in ("run_b", "run_a"):
        d = portfolio_root / name
        d.mkdir()
        (d / "weights.parquet").touch()
        (d / "manifest.json").touch()  # 完整产物目录
    (portfolio_root / "no_weights").mkdir()  # 无 weights.parquet -> 应被过滤掉
    half = portfolio_root / "half_baked"  # 有 weights 无 manifest（半成品）-> 应被过滤掉
    half.mkdir()
    (half / "weights.parquet").touch()

    daily_df = pl.DataFrame({"trade_date": ["20230101"], "ts_code": ["000001.SZ"]})
    calls: dict = {}

    def fake_fetch_daily(start, end):
        calls["fetch_daily"] = (start, end)
        return daily_df

    def fake_run_portfolio_simulation(run_dirs, daily, **kwargs):
        calls["run_dirs"] = run_dirs
        calls["daily"] = daily
        calls["kwargs"] = kwargs
        return {
            "run_dir": "workspace/sim/myrun",
            "sharpe": 1.234,
            "max_dd": -0.05,
            "ann_ret": 0.15,
        }

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
    monkeypatch.setattr(
        "factorzen.sim.engine.run_portfolio_simulation", fake_run_portfolio_simulation
    )

    ret = cli.main(
        [
            "sim",
            "run",
            "--portfolio-dir",
            str(portfolio_root),
            "--start",
            "20230101",
            "--end",
            "20230201",
            "--run-id",
            "myrun",
        ]
    )

    assert ret == 0
    assert calls["fetch_daily"] == ("20230101", "20230201")
    assert calls["run_dirs"] == [
        str(portfolio_root / "run_a"),
        str(portfolio_root / "run_b"),
    ]
    assert calls["daily"] is daily_df
    from factorzen.config.settings import SIM_DIR

    assert calls["kwargs"] == {"out_dir": str(SIM_DIR), "run_id": "myrun"}
    assert "cost_model" not in calls["kwargs"]

    out = capsys.readouterr().out
    assert "run_dir=workspace/sim/myrun" in out
    assert "sharpe=1.2340" in out
    assert "max_dd=-0.0500" in out
    assert "ann_ret=0.1500" in out


def test_cmd_sim_run_missing_portfolio_dir_returns_error(tmp_path, monkeypatch, capsys):
    """portfolio-dir 不存在时返回码 2 + 报错打到 stderr，且不应尝试跑真实 pipeline。"""
    import polars as pl

    from factorzen.cli import main as cli

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: pl.DataFrame())

    missing_dir = tmp_path / "does_not_exist"
    ret = cli.main(
        [
            "sim",
            "run",
            "--portfolio-dir",
            str(missing_dir),
            "--start",
            "20230101",
            "--end",
            "20230201",
        ]
    )

    assert ret == 2
    assert "portfolio-dir not found" in capsys.readouterr().err


def test_cmd_sim_run_no_weights_found_returns_error(tmp_path, monkeypatch, capsys):
    """portfolio-dir 存在但没有任何子目录含 weights.parquet 时返回码 2。"""
    import polars as pl

    from factorzen.cli import main as cli

    portfolio_root = tmp_path / "portfolios"
    portfolio_root.mkdir()
    (portfolio_root / "empty_run").mkdir()  # 无 weights.parquet

    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: pl.DataFrame())

    ret = cli.main(
        [
            "sim",
            "run",
            "--portfolio-dir",
            str(portfolio_root),
            "--start",
            "20230101",
            "--end",
            "20230201",
        ]
    )

    assert ret == 2
    assert "no portfolio run dirs found" in capsys.readouterr().err


def test_cmd_sim_show_prints_known_metrics_and_json_extras(tmp_path, capsys):
    """_cmd_sim_show：已知 5 个 key 按 "key: value" 逐行打印，未知 key 落入 JSON extras 块。"""
    import json

    from factorzen.cli import main as cli

    sim_dir = tmp_path / "sim_out"
    sim_dir.mkdir()
    metrics = {
        "ann_ret": 0.12,
        "sharpe": 1.5,
        "max_dd": -0.08,
        "ann_turnover": 3.2,
        "total_cost": 0.01,
        "n_days": 250,  # 未知 key -> 应落入 JSON extras
    }
    (sim_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    ret = cli.main(["sim", "show", "--sim-dir", str(sim_dir)])

    assert ret == 0
    out = capsys.readouterr().out
    assert "ann_ret: 0.12" in out
    assert "sharpe: 1.5" in out
    assert "max_dd: -0.08" in out
    assert "ann_turnover: 3.2" in out
    assert "total_cost: 0.01" in out
    assert '"n_days": 250' in out


def test_cmd_sim_show_missing_metrics_returns_error(tmp_path, capsys):
    """sim-dir 存在但没有 metrics.json 时返回码 2 + 报错打到 stderr。"""
    from factorzen.cli import main as cli

    sim_dir = tmp_path / "sim_missing"
    sim_dir.mkdir()

    ret = cli.main(["sim", "show", "--sim-dir", str(sim_dir)])

    assert ret == 2
    assert "metrics.json not found" in capsys.readouterr().err
