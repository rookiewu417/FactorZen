# tests/test_validation_cli.py
"""Tests for `fz validate overfit` CLI command."""


def test_parser_has_validate_overfit():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args(
        ["validate", "overfit", "momentum_12_1", "--start", "20230101", "--end", "20240101"]
    )
    assert args.command == "validate"
    assert args.validate_command == "overfit"
    assert args.factor == "momentum_12_1"
    assert callable(args.func)


def _install_fake_overfit_pipeline(monkeypatch):
    """monkeypatch `_cmd_validate_overfit` 依赖的每一步，返回按依赖名分组的调用记录。

    `_cmd_validate_overfit` 自己串起 get_factor → FactorDataContext → factor.compute →
    cross_sectional_zscore → DataBundle.build → compute_rank_ic → block_bootstrap_ic_ci →
    deflated_sharpe 这条链（没有单一 pipeline 入口可 monkeypatch），所以逐个依赖打桩，
    只让 rename/select 这类真实 polars 操作跑真的，其余全部替身、离线可跑。
    """
    import polars as pl

    calls: dict[str, list] = {
        "get_factor": [],
        "context": [],
        "get_universe": [],
        "zscore": [],
        "bundle_build": [],
        "compute_rank_ic": [],
        "bootstrap": [],
        "deflated_sharpe": [],
    }

    class FakeFactor:
        lookback_days = 45

        def compute(self, ctx):
            return "FAKE_FACTOR_DF"

    def fake_get_factor(name):
        calls["get_factor"].append(name)
        return FakeFactor

    def fake_get_universe(date_str, universe_name):
        calls["get_universe"].append((date_str, universe_name))
        return pl.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})

    class FakeDaily:
        def collect(self):
            return "FAKE_COLLECTED_DAILY"

    class FakeContext:
        def __init__(self, *, start, end, required_data, lookback_days, universe):
            calls["context"].append(
                {
                    "start": start,
                    "end": end,
                    "required_data": required_data,
                    "lookback_days": lookback_days,
                    "universe": universe,
                }
            )
            self.daily = FakeDaily()

    def fake_cross_sectional_zscore(fdf, col):
        calls["zscore"].append((fdf, col))
        return pl.DataFrame(
            {
                "trade_date": ["20230101", "20230102"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "factor_value_z": [0.1, -0.1],
            }
        )

    class FakeBundle:
        fwd_returns = "FAKE_FWD_RETURNS"

    class FakeDataBundle:
        @staticmethod
        def build(daily, train_ratio):
            calls["bundle_build"].append((daily, train_ratio))
            return FakeBundle()

    class FakeIcResult:
        ic_mean = 0.056
        ir = 1.234
        ic_series = pl.DataFrame({"ic": [0.05, 0.06, 0.07]})

    def fake_compute_rank_ic(factor_df, fwd_returns, *, factor_col, frequency):
        calls["compute_rank_ic"].append(
            {
                "factor_df": factor_df,
                "fwd_returns": fwd_returns,
                "factor_col": factor_col,
                "frequency": frequency,
            }
        )
        return FakeIcResult()

    def fake_block_bootstrap_ic_ci(ic_vals):
        calls["bootstrap"].append(list(ic_vals))
        return (-0.01, 0.09)

    def fake_deflated_sharpe(ir, *, n_trials, n_obs):
        calls["deflated_sharpe"].append({"ir": ir, "n_trials": n_trials, "n_obs": n_obs})
        return (2.5, 0.012)

    monkeypatch.setattr("factorzen.daily.factors.registry.get_factor", fake_get_factor)
    monkeypatch.setattr("factorzen.daily.data.context.FactorDataContext", FakeContext)
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    monkeypatch.setattr(
        "factorzen.daily.preprocessing.normalizer.cross_sectional_zscore",
        fake_cross_sectional_zscore,
    )
    monkeypatch.setattr("factorzen.discovery.scoring.DataBundle", FakeDataBundle)
    monkeypatch.setattr(
        "factorzen.daily.evaluation.ic_analysis.compute_rank_ic", fake_compute_rank_ic
    )
    monkeypatch.setattr(
        "factorzen.validation.bootstrap.block_bootstrap_ic_ci", fake_block_bootstrap_ic_ci
    )
    monkeypatch.setattr(
        "factorzen.validation.deflated_sharpe.deflated_sharpe", fake_deflated_sharpe
    )
    return calls


def test_cmd_validate_overfit_forwards_args_and_prints_metrics(monkeypatch, capsys):
    """`fz validate overfit`（无 --universe）应把 factor/start/end 转发到底层每一步。"""
    from factorzen.cli import main as cli

    calls = _install_fake_overfit_pipeline(monkeypatch)

    rc = cli.main(
        ["validate", "overfit", "momentum_12_1", "--start", "20230101", "--end", "20230601"]
    )

    assert rc == 0
    assert calls["get_factor"] == ["momentum_12_1"]
    assert calls["get_universe"] == []  # 未传 --universe，不应查询股票池
    assert calls["context"] == [
        {
            "start": "20230101",
            "end": "20230601",
            "required_data": ["daily", "daily_basic"],
            "lookback_days": 45,  # 取自 FakeFactor.lookback_days，而非硬编码的 60
            "universe": None,
        }
    ]
    assert calls["bundle_build"] == [("FAKE_COLLECTED_DAILY", 1.0)]
    assert calls["zscore"] == [("FAKE_FACTOR_DF", "factor_value")]
    assert len(calls["compute_rank_ic"]) == 1
    ic_call = calls["compute_rank_ic"][0]
    assert ic_call["fwd_returns"] == "FAKE_FWD_RETURNS"
    assert ic_call["factor_col"] == "factor_clean"
    assert ic_call["frequency"] == "daily"
    assert ic_call["factor_df"].columns == ["trade_date", "ts_code", "factor_clean"]
    assert calls["bootstrap"] == [[0.05, 0.06, 0.07]]
    assert calls["deflated_sharpe"] == [{"ir": 1.234, "n_trials": 1, "n_obs": 3}]

    out = capsys.readouterr().out.splitlines()
    assert out[0] == (
        "[validate] momentum_12_1: IC=0.0560 IR=1.2340 "
        "DSR_p=0.0120 IC_95%CI=[-0.0100,0.0900]"
    )
    assert out[1] == "[validate] 注：单因子 N=1（无多重检验扣减）；PBO 仅适用候选池，此处略。"


def test_cmd_validate_overfit_resolves_universe_into_context(monkeypatch):
    """`fz validate overfit --universe` 应查询 get_universe 并把股票列表转发进 context。"""
    from factorzen.cli import main as cli

    calls = _install_fake_overfit_pipeline(monkeypatch)

    rc = cli.main(
        [
            "validate",
            "overfit",
            "momentum_12_1",
            "--start",
            "20230101",
            "--end",
            "20230601",
            "--universe",
            "csi500",
        ]
    )

    assert rc == 0
    assert calls["get_universe"] == [("20230601", "csi500")]
    assert calls["context"][0]["universe"] == ["000001.SZ", "000002.SZ"]
