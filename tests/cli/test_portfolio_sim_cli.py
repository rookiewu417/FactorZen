"""test_portfolio_cli.py：Tests for `fz portfolio build` CLI: parser shape + execution-level forwarding.
test_risk_cli.py：Tests for `fz risk build` CLI: parser shape + execution-level forwarding.
test_sim_cli.py：Tests for `fz sim run / sim show` CLI: parser shape + execution-level forwarding.
"""


from __future__ import annotations

# ==== 来自 test_portfolio_cli.py ====

def test_parser_has_portfolio_build():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "portfolio",
            "build",
            "--start",
            "20230101",
            "--end",
            "20241231",
            "--universe",
            "csi500",
            "--alpha-file",
            "a.parquet",
            "--lam",
            "2.0",
        ]
    )
    assert args.command == "portfolio"
    assert args.portfolio_command == "build"
    assert args.alpha_file == "a.parquet"
    assert args.lam == 2.0
    assert callable(args.func)


def test_parser_portfolio_build_defaults():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        [
            "portfolio",
            "build",
            "--start",
            "20230101",
            "--end",
            "20241231",
            "--alpha-file",
            "a.csv",
        ]
    )
    assert args.w_max == 0.05
    assert args.turnover is None
    assert args.industry_neutral is False


def test_cmd_portfolio_build_industry_neutral_uses_equal_weight_bench(
    tmp_path, monkeypatch, capsys
):
    """--industry-neutral 时：neutral_factors=ind_* 列、bench_weights=universe 等权（非绝对 0）、

    signal_date 从 YYYYMMDD 转成 YYYY-MM-DD，alpha 按 codes 顺序对齐、缺失填 0。
    """
    from types import SimpleNamespace

    import numpy as np
    import polars as pl

    from factorzen.cli import main as cli

    stocks_df = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "industry": ["银行", "地产", "银行"],
        }
    )
    daily_df = pl.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"], "trade_date": ["20230201"] * 3}
    )
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    risk_result = SimpleNamespace(
        factor_exposures=SimpleNamespace(codes=codes),
        factor_names=["beta", "ind_bank", "ind_property"],
    )
    calls: dict = {}

    def fake_get_universe(date_str, universe_name):
        calls["get_universe"] = (date_str, universe_name)
        return stocks_df

    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

    class FakeRiskModel:
        def __init__(self, *a, **kw):
            pass

        def build(self, daily, daily_basic, stocks, start, end):
            calls["risk_build_range"] = (start, end)
            return risk_result

    monkeypatch.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)

    def fake_run_portfolio(alpha, risk_result_arg, **kwargs):
        calls["alpha"] = alpha
        calls["risk_result_arg"] = risk_result_arg
        calls["kwargs"] = kwargs
        return {
            "status": "optimal",
            "n_holdings": 2,
            "run_dir": "workspace/portfolios/test_run",
        }

    monkeypatch.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)

    alpha_file = tmp_path / "alpha.csv"
    pl.DataFrame({"ts_code": ["000001.SZ", "000003.SZ"], "alpha": [0.5, -0.2]}).write_csv(
        alpha_file
    )

    ret = cli.main(
        [
            "portfolio",
            "build",
            "--start",
            "20230101",
            "--end",
            "20230201",
            "--universe",
            "csi500",
            "--alpha-file",
            str(alpha_file),
            "--lam",
            "2.5",
            "--w-max",
            "0.08",
            "--turnover",
            "0.3",
            "--industry-neutral",
        ]
    )

    assert ret == 0
    assert calls["get_universe"] == ("20230201", "csi500")
    assert calls["risk_build_range"] == ("20230101", "20230201")
    assert calls["risk_result_arg"] is risk_result

    # alpha 按 codes 顺序对齐，缺失（000002.SZ 未出现在 alpha 文件里）填 0
    np.testing.assert_allclose(calls["alpha"], [0.5, 0.0, -0.2])

    kwargs = calls["kwargs"]
    assert kwargs["codes"] == codes
    assert kwargs["sectors"] == ["银行", "地产", "银行"]
    assert kwargs["risk_aversion"] == 2.5
    assert kwargs["w_max"] == 0.08
    assert kwargs["turnover_budget"] == 0.3
    assert kwargs["neutral_factors"] == ["ind_bank", "ind_property"]
    # --industry-neutral 使用 universe 等权基准，而非绝对 0
    # （raw one-hot 行业列下 target=0 + long_only + Σw=1 必然 infeasible）
    np.testing.assert_allclose(kwargs["bench_weights"], [1 / 3, 1 / 3, 1 / 3])
    assert kwargs["signal_date"] == "2023-02-01"

    out = capsys.readouterr().out
    assert "status=optimal" in out
    assert "holdings=2" in out
    assert "workspace/portfolios/test_run" in out


def test_cmd_portfolio_build_signal_date_passthrough_when_end_not_yyyymmdd(
    tmp_path, monkeypatch
):
    """args.end 不是 8 位纯数字（如已经是 YYYY-MM-DD 格式）时，signal_date 原样
    透传而非重新拼接——此前只有 8 位数字格式（if 分支）有测试覆盖，else 分支
    未被覆盖。
    """
    from types import SimpleNamespace

    import polars as pl

    from factorzen.cli import main as cli

    stocks_df = pl.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]})
    daily_df = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20230201"]})
    codes = ["000001.SZ"]
    risk_result = SimpleNamespace(
        factor_exposures=SimpleNamespace(codes=codes),
        factor_names=["beta"],
    )
    calls: dict = {}

    monkeypatch.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

    class FakeRiskModel:
        def __init__(self, *a, **kw):
            pass

        def build(self, daily, daily_basic, stocks, start, end):
            return risk_result

    monkeypatch.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)

    def fake_run_portfolio(alpha, risk_result_arg, **kwargs):
        calls["kwargs"] = kwargs
        return {
            "status": "optimal",
            "n_holdings": 1,
            "run_dir": "workspace/portfolios/test_run",
        }

    monkeypatch.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)

    alpha_file = tmp_path / "alpha.csv"
    pl.DataFrame({"ts_code": ["000001.SZ"], "alpha": [0.5]}).write_csv(alpha_file)

    ret = cli.main(
        [
            "portfolio",
            "build",
            "--start",
            "20230101",
            "--end",
            "2023-02-01",  # 非 8 位数字格式，应命中 else 分支原样透传
            "--universe",
            "csi500",
            "--alpha-file",
            str(alpha_file),
        ]
    )

    assert ret == 0
    assert calls["kwargs"]["signal_date"] == "2023-02-01"


def test_cmd_portfolio_build_without_industry_neutral_has_no_bench(tmp_path, monkeypatch):
    """未传 --industry-neutral 时：neutral_factors/bench_weights/turnover_budget 均为 None，

    风险厌恶/w_max 用 argparse 默认值（--lam 未传→1.0，--w-max 未传→0.05）。
    """
    from types import SimpleNamespace

    import polars as pl

    from factorzen.cli import main as cli

    stocks_df = pl.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ"], "industry": ["银行", "地产"]}
    )
    daily_df = pl.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ"], "trade_date": ["20230201"] * 2}
    )
    codes = ["000001.SZ", "000002.SZ"]
    risk_result = SimpleNamespace(
        factor_exposures=SimpleNamespace(codes=codes),
        factor_names=["beta", "ind_bank"],
    )

    monkeypatch.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

    class FakeRiskModel:
        def __init__(self, *a, **kw):
            pass

        def build(self, daily, daily_basic, stocks, start, end):
            return risk_result

    monkeypatch.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)

    captured_kwargs: dict = {}

    def fake_run_portfolio(alpha, risk_result_arg, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "status": "optimal",
            "n_holdings": 1,
            "run_dir": "workspace/portfolios/default_run",
        }

    monkeypatch.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)

    alpha_file = tmp_path / "alpha.parquet"
    pl.DataFrame({"ts_code": ["000001.SZ"], "alpha": [0.1]}).write_parquet(alpha_file)

    ret = cli.main(
        [
            "portfolio",
            "build",
            "--start",
            "20230101",
            "--end",
            "20230201",
            "--alpha-file",
            str(alpha_file),
        ]
    )

    assert ret == 0
    assert captured_kwargs["neutral_factors"] is None
    assert captured_kwargs["bench_weights"] is None
    assert captured_kwargs["turnover_budget"] is None
    assert captured_kwargs["risk_aversion"] == 1.0  # --lam 默认值
    assert captured_kwargs["w_max"] == 0.05  # --w-max 默认值


def test_cmd_portfolio_build_forwards_run_id_and_out_dir(tmp_path, monkeypatch):
    """--run-id/--out-dir 应透传给 run_portfolio，使多期 CLI 构建落不同目录（不覆盖）。"""
    from types import SimpleNamespace

    import polars as pl

    from factorzen.cli import main as cli

    stocks_df = pl.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]})
    daily_df = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20230201"]})
    rr = SimpleNamespace(factor_exposures=SimpleNamespace(codes=["000001.SZ"]),
                         factor_names=["beta"])
    calls: dict = {}
    monkeypatch.setattr("factorzen.core.universe.get_universe", lambda d, n: stocks_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

    class FakeRM:
        def __init__(self, *a, **k):
            pass

        def build(self, *a, **k):
            return rr

    monkeypatch.setattr("factorzen.risk.model.RiskModel", FakeRM)
    monkeypatch.setattr(
        "factorzen.pipelines.portfolio_build.run_portfolio",
        lambda alpha, rr_arg, **kw: (calls.update(kwargs=kw),
                                     {"status": "optimal", "n_holdings": 1, "run_dir": "x"})[1],
    )
    alpha_file = tmp_path / "a.csv"
    pl.DataFrame({"ts_code": ["000001.SZ"], "alpha": [0.5]}).write_csv(alpha_file)

    rc = cli.main(["portfolio", "build", "--start", "20230101", "--end", "20230201",
                   "--universe", "csi500", "--alpha-file", str(alpha_file),
                   "--run-id", "reb_0201", "--out-dir", str(tmp_path / "po")])
    assert rc == 0
    assert calls["kwargs"]["run_id"] == "reb_0201"
    assert calls["kwargs"]["out_dir"] == str(tmp_path / "po")


# ==== 来自 test_risk_cli.py ====

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
    assert args.spec_half_life == 90
    assert args.spec_shrinkage == 0.3


def test_cmd_risk_build_forwards_params_and_filters_by_universe(monkeypatch, capsys):
    """_cmd_risk_build：universe 过滤 daily/daily_basic + 超参数正确转发给 run_risk_build。"""
    import polars as pl

    from factorzen.cli import main as cli
    from factorzen.pipelines.risk_build import risk_lookback_start

    stocks_df = pl.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ"], "industry": ["银行", "地产"]}
    )
    daily_df = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "999999.SZ"],
            "trade_date": ["20230101", "20230101", "20230101"],
        }
    )
    daily_basic_df = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "999999.SZ"],
            "trade_date": ["20230101", "20230101", "20230101"],
        }
    )
    calls: dict = {}

    def fake_get_universe(date_str, universe_name):
        calls["get_universe"] = (date_str, universe_name)
        return stocks_df

    def fake_fetch_daily(start, end):
        calls["fetch_daily"] = (start, end)
        return daily_df

    def fake_fetch_daily_basic(start, end):
        calls["fetch_daily_basic"] = (start, end)
        return daily_basic_df

    def fake_run_risk_build(daily, daily_basic, stocks, start, end, **kwargs):
        calls["run_risk_build_args"] = (daily, daily_basic, stocks, start, end)
        calls["run_risk_build_kwargs"] = kwargs
        return {
            "run_dir": "workspace/risk_models/risk_test",
            "r_squared": 0.42,
            "factor_names": ["beta", "size", "ind_bank"],
        }

    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", fake_fetch_daily_basic)
    monkeypatch.setattr("factorzen.pipelines.risk_build.run_risk_build", fake_run_risk_build)

    ret = cli.main(
        [
            "risk",
            "build",
            "--start",
            "20230101",
            "--end",
            "20230201",
            "--universe",
            "csi500",
            "--cov-half-life",
            "60",
            "--nw-lags",
            "3",
            "--spec-half-life",
            "45",
            "--spec-shrinkage",
            "0.5",
        ]
    )

    assert ret == 0
    assert calls["get_universe"] == ("20230201", "csi500")
    # daily/daily_basic 拉取须补 lookback 历史（预热滚动风格因子），故 start 早于 --start、
    # end 不变；截面回归区间仍由传给 run_risk_build 的原始 start/end 界定（见下 start_arg）。
    lb_start = risk_lookback_start("20230101")
    assert lb_start < "20230101"
    assert calls["fetch_daily"] == (lb_start, "20230201")
    assert calls["fetch_daily_basic"] == (lb_start, "20230201")

    daily_arg, daily_basic_arg, stocks_arg, start_arg, end_arg = calls["run_risk_build_args"]
    # daily/daily_basic 在 handler 内被过滤到 universe codes，999999.SZ（不在 stocks 里）应被剔除
    assert sorted(daily_arg["ts_code"].to_list()) == ["000001.SZ", "000002.SZ"]
    assert sorted(daily_basic_arg["ts_code"].to_list()) == ["000001.SZ", "000002.SZ"]
    assert stocks_arg is stocks_df
    assert start_arg == "20230101"
    assert end_arg == "20230201"
    assert calls["run_risk_build_kwargs"] == {
        "cov_half_life": 60,
        "nw_lags": 3,
        "spec_half_life": 45,
        "spec_shrinkage": 0.5,
    }

    out = capsys.readouterr().out
    assert "factors=3" in out
    assert "R2=0.4200" in out
    assert "workspace/risk_models/risk_test" in out


def test_cmd_risk_build_forwards_default_hyperparams_when_flags_omitted(monkeypatch):
    """未显式传 --cov-half-life 等 flag 时，argparse 默认值应原样转发给 run_risk_build。"""
    import polars as pl

    from factorzen.cli import main as cli

    stocks_df = pl.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]})
    daily_df = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20230101"]})

    monkeypatch.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
    monkeypatch.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

    captured_kwargs: dict = {}

    def fake_run_risk_build(daily, daily_basic, stocks, start, end, **kwargs):
        captured_kwargs.update(kwargs)
        return {"run_dir": "r", "r_squared": 0.0, "factor_names": []}

    monkeypatch.setattr("factorzen.pipelines.risk_build.run_risk_build", fake_run_risk_build)

    ret = cli.main(
        ["risk", "build", "--start", "20230101", "--end", "20230201", "--universe", "all_a"]
    )

    assert ret == 0
    assert captured_kwargs == {
        "cov_half_life": 90,
        "nw_lags": 2,
        "spec_half_life": 90,
        "spec_shrinkage": 0.3,
    }


# ==== 来自 test_sim_cli.py ====

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
