"""Tests for `fz risk build` CLI: parser shape + execution-level forwarding."""


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
    assert calls["fetch_daily"] == ("20230101", "20230201")
    assert calls["fetch_daily_basic"] == ("20230101", "20230201")

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
