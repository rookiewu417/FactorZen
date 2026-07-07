"""Tests for `fz portfolio build` CLI: parser shape + execution-level forwarding."""

from __future__ import annotations


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
