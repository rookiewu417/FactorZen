"""test_portfolio_cli.py：Tests for `fz portfolio build` CLI: parser shape + execution-level forwarding.
test_risk_cli.py：Tests for `fz risk build` CLI: parser shape + execution-level forwarding.
test_sim_cli.py：Tests for `fz sim run / sim show` CLI: parser shape + execution-level forwarding.
"""


from __future__ import annotations

# ==== 来自 test_portfolio_cli.py ====
import pytest


def test_parser_portfolio_build_suite():
    """test_parser_has_portfolio_build；test_parser_portfolio_build_defaults"""
    # -- 原 test_parser_has_portfolio_build --
    def _section_0_test_parser_has_portfolio_build():
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

    _section_0_test_parser_has_portfolio_build()

    # -- 原 test_parser_portfolio_build_defaults --
    def _section_1_test_parser_portfolio_build_defaults():
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

    _section_1_test_parser_portfolio_build_defaults()


def test_cmd_portfolio_build_suite(tmp_path, capsys):
    """--industry-neutral 时：neutral_factors=ind_* 列、bench_weights=universe 等权（非绝对 0）、；args.end 不是 8 位纯数字（如已经是 YYYY-MM-DD 格式）时，signal_date 原样；未传 --industry-neutral 时：neutral_factors/bench_weights/turnover_budget 均为 None，；--run-id/--out-dir 应透传给 run_portfolio，使多期 CLI 构建落不同目录（不覆盖）。"""
    # -- 原 test_cmd_portfolio_build_industry_neutral_uses_equal_weight_bench --
    def _section_0_test_cmd_portfolio_build_industry_neutral_uses_equal_weight_bench(tmp_path, mp, capsys):
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

        mp.setattr("factorzen.core.universe.get_universe", fake_get_universe)
        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

        class FakeRiskModel:
            def __init__(self, *a, **kw):
                pass

            def build(self, daily, daily_basic, stocks, start, end):
                calls["risk_build_range"] = (start, end)
                return risk_result

        mp.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)

        def fake_run_portfolio(alpha, risk_result_arg, **kwargs):
            calls["alpha"] = alpha
            calls["risk_result_arg"] = risk_result_arg
            calls["kwargs"] = kwargs
            return {
                "status": "optimal",
                "n_holdings": 2,
                "run_dir": "workspace/portfolios/test_run",
            }

        mp.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)

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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cmd_portfolio_build_industry_neutral_uses_equal_weight_bench(_tp0, mp, capsys)

    # -- 原 test_cmd_portfolio_build_signal_date_passthrough_when_end_not_yyyymmdd --
    def _section_1_test_cmd_portfolio_build_signal_date_passthrough_when_end_not_yyyymmdd(tmp_path, mp):
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

        mp.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

        class FakeRiskModel:
            def __init__(self, *a, **kw):
                pass

            def build(self, daily, daily_basic, stocks, start, end):
                return risk_result

        mp.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)

        def fake_run_portfolio(alpha, risk_result_arg, **kwargs):
            calls["kwargs"] = kwargs
            return {
                "status": "optimal",
                "n_holdings": 1,
                "run_dir": "workspace/portfolios/test_run",
            }

        mp.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)

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

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_portfolio_build_signal_date_passthrough_when_end_not_yyyymmdd(_tp1, mp)

    # -- 原 test_cmd_portfolio_build_without_industry_neutral_has_no_bench --
    def _section_2_test_cmd_portfolio_build_without_industry_neutral_has_no_bench(tmp_path, mp):
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

        mp.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

        class FakeRiskModel:
            def __init__(self, *a, **kw):
                pass

            def build(self, daily, daily_basic, stocks, start, end):
                return risk_result

        mp.setattr("factorzen.risk.model.RiskModel", FakeRiskModel)

        captured_kwargs: dict = {}

        def fake_run_portfolio(alpha, risk_result_arg, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "status": "optimal",
                "n_holdings": 1,
                "run_dir": "workspace/portfolios/default_run",
            }

        mp.setattr("factorzen.pipelines.portfolio_build.run_portfolio", fake_run_portfolio)

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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_portfolio_build_without_industry_neutral_has_no_bench(_tp2, mp)

    # -- 原 test_cmd_portfolio_build_forwards_run_id_and_out_dir --
    def _section_3_test_cmd_portfolio_build_forwards_run_id_and_out_dir(tmp_path, mp):
        from types import SimpleNamespace

        import polars as pl

        from factorzen.cli import main as cli

        stocks_df = pl.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]})
        daily_df = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20230201"]})
        rr = SimpleNamespace(factor_exposures=SimpleNamespace(codes=["000001.SZ"]),
                             factor_names=["beta"])
        calls: dict = {}
        mp.setattr("factorzen.core.universe.get_universe", lambda d, n: stocks_df)
        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

        class FakeRM:
            def __init__(self, *a, **k):
                pass

            def build(self, *a, **k):
                return rr

        mp.setattr("factorzen.risk.model.RiskModel", FakeRM)
        mp.setattr(
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

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cmd_portfolio_build_forwards_run_id_and_out_dir(_tp3, mp)


def _write_mini_risk_dir_for_cli(run_dir):
    """手写 3 股 × 2 因子微型 risk 产物（与 tests/risk 中 round-trip fixture 同形）。"""
    import json
    from pathlib import Path

    import numpy as np
    import polars as pl

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    factor_names = ["size", "value"]
    matrix = np.array([[1.0, 0.2], [0.0, -0.5], [-1.0, 0.3]], dtype=float)
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    pl.DataFrame({"ts_code": codes}).hstack(
        pl.DataFrame(matrix, schema=factor_names, orient="row")
    ).write_parquet(run_dir / "exposures.parquet")
    pl.DataFrame(cov, schema=factor_names, orient="row").write_parquet(
        run_dir / "factor_covariance.parquet"
    )
    # 故意打乱行序
    pl.DataFrame(
        {
            "ts_code": ["000003.SZ", "000001.SZ", "000002.SZ"],
            "specific_risk": [0.25, 0.20, 0.30],
        }
    ).write_parquet(run_dir / "specific_risk.parquet")
    pl.DataFrame(
        {"trade_date": ["20240102"], "size": [0.01], "value": [0.005]}
    ).write_parquet(run_dir / "factor_returns.parquet")
    (run_dir / "manifest.json").write_text(
        json.dumps({"r_squared": 0.42, "n_valid_dates": 10}), encoding="utf-8"
    )


def test_cmd_portfolio_build_risk_dir_suite(tmp_path, capsys):
    """无 --risk-dir：RiskModel.build 被调 1 次；有 --risk-dir：旁路 build，weights 产出且 target_weight 求和≈1。"""
    import numpy as np
    import polars as pl

    from factorzen.cli import main as cli
    from factorzen.risk.exposures import ExposureMatrix
    from factorzen.risk.model import RiskModelResult

    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    stocks_df = pl.DataFrame(
        {"ts_code": codes, "industry": ["银行", "地产", "银行"]}
    )
    mini_risk = RiskModelResult(
        factor_exposures=ExposureMatrix(
            codes=codes,
            factor_names=["size", "value"],
            matrix=np.array([[1.0, 0.2], [0.0, -0.5], [-1.0, 0.3]], dtype=float),
        ),
        factor_covariance=np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float),
        specific_risk=np.array([0.20, 0.30, 0.25], dtype=float),
        factor_returns=pl.DataFrame(),
        r_squared=0.42,
        factor_names=["size", "value"],
    )
    daily_df = pl.DataFrame({"ts_code": codes, "trade_date": ["20230201"] * 3})
    basic_df = daily_df.clone()

    # -- 正控：无 --risk-dir → build 被调 1 次 --
    def _section_0_default_calls_build(tmp_path, mp):
        build_calls = {"n": 0}

        # 先 import 下游模块，避免整类替换 RiskModel 时污染其模块级绑定
        import factorzen.pipelines.portfolio_build  # noqa: F401

        mp.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
        mp.setattr(
            "factorzen.pipelines.risk_build.load_risk_inputs",
            lambda loader, start, end, uni: (daily_df, basic_df),
        )

        def fake_build(self, *a, **k):
            build_calls["n"] += 1
            return mini_risk

        mp.setattr("factorzen.risk.model.RiskModel.build", fake_build)
        mp.setattr(
            "factorzen.pipelines.portfolio_build.run_portfolio",
            lambda alpha, rr, **kw: {
                "status": "optimal",
                "n_holdings": 2,
                "run_dir": str(tmp_path / "po_default"),
            },
        )
        alpha_file = tmp_path / "alpha.csv"
        pl.DataFrame({"ts_code": codes, "alpha": [0.5, 0.1, -0.2]}).write_csv(alpha_file)

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
                "--out-dir",
                str(tmp_path / "po_default"),
            ]
        )
        assert ret == 0
        assert build_calls["n"] == 1

    _tp0 = tmp_path / "_risk_dir_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_default_calls_build(_tp0, mp)

    # -- 旁路：--risk-dir 跳过 build；weights.parquet 产出且 target_weight 求和≈1 --
    def _section_1_risk_dir_bypasses_build(tmp_path, mp):
        import factorzen.attribution.risk_attribution as ra_mod
        import factorzen.pipelines.portfolio_build as pb_mod
        import factorzen.risk.model as risk_model_mod

        risk_dir = tmp_path / "risk_run"
        _write_mini_risk_dir_for_cli(risk_dir)

        def boom_build(self, *a, **k):
            raise AssertionError("有 --risk-dir 时不应调用 RiskModel.build")

        # 先前 suite 可能用整类 FakeRiskModel 污染过 risk_attribution/portfolio_build
        # 的模块级绑定；此处强制恢复真实类，只拦截 build。
        real_rm = risk_model_mod.RiskModel
        mp.setattr(ra_mod, "RiskModel", real_rm)
        mp.setattr(pb_mod, "RiskModel", real_rm)
        mp.setattr(real_rm, "build", boom_build)

        mp.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
        mp.setattr(
            "factorzen.pipelines.risk_build.load_risk_inputs",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("有 --risk-dir 时不应 load_risk_inputs")
            ),
        )

        alpha_file = tmp_path / "alpha.csv"
        pl.DataFrame({"ts_code": codes, "alpha": [0.8, 0.3, 0.1]}).write_csv(alpha_file)
        out_dir = tmp_path / "po_risk_dir"

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
                "--risk-dir",
                str(risk_dir),
                "--out-dir",
                str(out_dir),
                "--run-id",
                "from_risk",
                # 默认 w_max=0.05 在 3 只股票上 Σw 上限 0.15 < budget=1 → infeasible
                "--w-max",
                "1.0",
            ]
        )
        assert ret == 0
        weights_path = out_dir / "from_risk" / "weights.parquet"
        assert weights_path.exists(), f"应产出 weights.parquet: {weights_path}"
        w = pl.read_parquet(weights_path)
        assert "target_weight" in w.columns
        assert abs(float(w["target_weight"].sum()) - 1.0) < 1e-5

    _tp1 = tmp_path / "_risk_dir_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_risk_dir_bypasses_build(_tp1, mp)

    # -- crypto + --risk-dir → return 2 --
    def _section_2_crypto_rejects_risk_dir(tmp_path, mp, capsys):
        alpha_file = tmp_path / "a.csv"
        pl.DataFrame({"ts_code": ["BTC/USDT"], "alpha": [0.1]}).write_csv(alpha_file)
        ret = cli.main(
            [
                "portfolio",
                "build",
                "--market",
                "crypto",
                "--start",
                "20230101",
                "--end",
                "20230201",
                "--alpha-file",
                str(alpha_file),
                "--risk-dir",
                str(tmp_path / "any"),
            ]
        )
        assert ret == 2
        err = capsys.readouterr().err
        assert "--risk-dir" in err

    _tp2 = tmp_path / "_risk_dir_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_crypto_rejects_risk_dir(_tp2, mp, capsys)


# ==== 来自 test_risk_cli.py ====

def test_risk_build_suite(capsys):
    """test_parser_has_risk_build；_cmd_risk_build：universe 过滤 daily/daily_basic + 超参数正确转发给 run_risk_build。；未显式传 --cov-half-life 等 flag 时，argparse 默认值应原样转发给 run_risk_build。"""
    # -- 原 test_parser_has_risk_build --
    def _section_0_test_parser_has_risk_build():
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

    _section_0_test_parser_has_risk_build()

    # -- 原 test_cmd_risk_build_forwards_params_and_filters_by_universe --
    def _section_1_test_cmd_risk_build_forwards_params_and_filters_by_universe(mp, capsys):
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

        mp.setattr("factorzen.core.universe.get_universe", fake_get_universe)
        mp.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", fake_fetch_daily_basic)
        mp.setattr("factorzen.pipelines.risk_build.run_risk_build", fake_run_risk_build)

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

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_risk_build_forwards_params_and_filters_by_universe(mp, capsys)

    # -- 原 test_cmd_risk_build_forwards_default_hyperparams_when_flags_omitted --
    def _section_2_test_cmd_risk_build_forwards_default_hyperparams_when_flags_omitted(mp):
        import polars as pl

        from factorzen.cli import main as cli

        stocks_df = pl.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]})
        daily_df = pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20230101"]})

        mp.setattr("factorzen.core.universe.get_universe", lambda d, u: stocks_df)
        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: daily_df)
        mp.setattr("factorzen.core.loader.fetch_daily_basic", lambda s, e: daily_df)

        captured_kwargs: dict = {}

        def fake_run_risk_build(daily, daily_basic, stocks, start, end, **kwargs):
            captured_kwargs.update(kwargs)
            return {"run_dir": "r", "r_squared": 0.0, "factor_names": []}

        mp.setattr("factorzen.pipelines.risk_build.run_risk_build", fake_run_risk_build)

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

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_risk_build_forwards_default_hyperparams_when_flags_omitted(mp)


# ==== 来自 test_sim_cli.py ====


def test_sim_run_suite(tmp_path, capsys):
    """sim run accepts optional --run-id; defaults to None when omitted.；_cmd_sim_run：只挑有 weights.parquet 的子目录(按路径排序) + out_dir/run_id 正确转发；；portfolio-dir 不存在时返回码 2 + 报错打到 stderr，且不应尝试跑真实 pipeline。；portfolio-dir 存在但没有任何子目录含 weights.parquet 时返回码 2。"""
    # -- 原 test_parser_sim_run_optional_run_id --
    def _section_0_test_parser_sim_run_optional_run_id():
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

    _section_0_test_parser_sim_run_optional_run_id()

    # -- 原 test_cmd_sim_run_forwards_filtered_run_dirs_without_explicit_cost_model --
    def _section_1_test_cmd_sim_run_forwards_filtered_run_dirs_without_explicit_cost_model(tmp_path, mp, capsys):
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

        mp.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
        mp.setattr(
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

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_sim_run_forwards_filtered_run_dirs_without_explicit_cost_model(_tp1, mp, capsys)

    # -- 原 test_cmd_sim_run_missing_portfolio_dir_returns_error --
    def _section_2_test_cmd_sim_run_missing_portfolio_dir_returns_error(tmp_path, mp, capsys):
        import polars as pl

        from factorzen.cli import main as cli

        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: pl.DataFrame())

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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_sim_run_missing_portfolio_dir_returns_error(_tp2, mp, capsys)

    # -- 原 test_cmd_sim_run_no_weights_found_returns_error --
    def _section_3_test_cmd_sim_run_no_weights_found_returns_error(tmp_path, mp, capsys):
        import polars as pl

        from factorzen.cli import main as cli

        portfolio_root = tmp_path / "portfolios"
        portfolio_root.mkdir()
        (portfolio_root / "empty_run").mkdir()  # 无 weights.parquet

        mp.setattr("factorzen.core.loader.fetch_daily", lambda s, e: pl.DataFrame())

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

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cmd_sim_run_no_weights_found_returns_error(_tp3, mp, capsys)


def test_sim_show_suite(tmp_path, capsys):
    """_cmd_sim_show：已知 5 个 key 按 "key: value" 逐行打印，未知 key 落入 JSON extras 块。；sim-dir 存在但没有 metrics.json 时返回码 2 + 报错打到 stderr。"""
    # -- 原 test_cmd_sim_show_prints_known_metrics_and_json_extras --
    def _section_0_test_cmd_sim_show_prints_known_metrics_and_json_extras(tmp_path, capsys):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_cmd_sim_show_prints_known_metrics_and_json_extras(_tp0, capsys)

    # -- 原 test_cmd_sim_show_missing_metrics_returns_error --
    def _section_1_test_cmd_sim_show_missing_metrics_returns_error(tmp_path, capsys):
        from factorzen.cli import main as cli

        sim_dir = tmp_path / "sim_missing"
        sim_dir.mkdir()

        ret = cli.main(["sim", "show", "--sim-dir", str(sim_dir)])

        assert ret == 2
        assert "metrics.json not found" in capsys.readouterr().err

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_cmd_sim_show_missing_metrics_returns_error(_tp1, capsys)


