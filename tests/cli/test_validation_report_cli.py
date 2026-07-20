"""test_validation_cli.py：Tests for `fz validate overfit` CLI command.
test_report_cli.py：Tests for `fz report portfolio` CLI parser (Task 5).
test_ops_cli_smoke.py：fz ops daily/status CLI 冒烟(dispatch/日期解析/返回码/状态打印)。
"""


from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from factorzen.cli.main import main
from factorzen.ops.state import OpsState

# ==== 来自 test_validation_cli.py ====

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

    def fake_deflated_sharpe(ir, n_trials, n_obs, skew=0.0, kurt=3.0, sharpe_variance=None):
        calls["deflated_sharpe"].append({"ir": ir, "n_trials": n_trials, "n_obs": n_obs})
        return (2.5, 0.012)

    monkeypatch.setattr("factorzen.daily.factors.registry.get_factor", fake_get_factor)
    monkeypatch.setattr("factorzen.daily.data.context.FactorDataContext", FakeContext)
    monkeypatch.setattr("factorzen.core.universe.get_universe", fake_get_universe)
    # 合并后 validate 走 scoring.ic_overfit_report(crypto 共用重构),它在 scoring 顶层
    # import 了这两个函数,故须 patch scoring 命名空间(patch 源模块对已绑定名字无效)。
    monkeypatch.setattr(
        "factorzen.discovery.scoring.cross_sectional_zscore",
        fake_cross_sectional_zscore,
    )
    monkeypatch.setattr("factorzen.discovery.scoring.DataBundle", FakeDataBundle)
    monkeypatch.setattr(
        "factorzen.discovery.scoring.compute_rank_ic", fake_compute_rank_ic
    )
    monkeypatch.setattr(
        "factorzen.validation.bootstrap.block_bootstrap_ic_ci", fake_block_bootstrap_ic_ci
    )
    # ic_overfit_report 经 guardrails.deflated_pvalue 调用 deflated_sharpe（两条挖掘路径的
    # DSR 唯一入口，见 test_deflation_recipe_parity 的架构守卫）。guardrails 顶层已绑定该名字，
    # 故须 patch guardrails 命名空间——patch 源模块对已绑定名字无效（此前碰巧生效只因
    # guardrails 尚未被导入，依赖导入时序，脆弱）。
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.deflated_sharpe", fake_deflated_sharpe
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


# ==== 来自 test_report_cli.py ====

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

    # 修复4：_cmd_report_portfolio 重建的 sim_result 此前只设置 .nav、未设置
    # .returns，导致 _make_monthly_return_heatmap（只读 .returns）在生产路径下
    # 恒返回 None，月度收益热力图永不渲染（死代码）。同一份 nav_df 已含
    # net_return 列，足够同时驱动两张图。
    assert "月度收益热力图" in html, (
        "月度收益热力图应被渲染进 HTML；sim_result.returns 未设置会导致该图永远是"
        "死代码（'暂无数据'而非真正接通）"
    )
    assert html.count("data:image/png;base64") >= 2, (
        f"应同时渲染净值曲线图与月度收益热力图两张 base64 图表，实际出现"
        f" {html.count('data:image/png;base64')} 次"
    )


# ==== 来自 test_ops_cli_smoke.py ====

def _write_cfg(tmp_path, state_dir=None):
    p = tmp_path / "ops.yaml"
    sd = state_dir or (tmp_path / "state")
    p.write_text(
        f"session_dir: s\nportfolio_run_dirs_glob: g\nstate_dir: {sd}\n",
        encoding="utf-8",
    )
    return p


def test_fz_ops_daily_dispatches_with_date(monkeypatch, tmp_path):
    p = _write_cfg(tmp_path)
    seen: dict = {}

    def fake_run(cfg, as_of, notifier=None):
        seen["as_of"] = as_of
        seen["session_dir"] = cfg.session_dir
        return 0

    monkeypatch.setattr("factorzen.ops.runner.run_ops_daily", fake_run)
    rc = main(["ops", "daily", "--config", str(p), "--date", "20260720"])
    assert rc == 0
    assert seen["as_of"] == date(2026, 7, 20)
    assert seen["session_dir"] == "s"


def test_fz_ops_daily_defaults_to_today(monkeypatch, tmp_path):
    p = _write_cfg(tmp_path)
    seen: dict = {}

    def fake_run(cfg, as_of, notifier=None):
        seen["as_of"] = as_of
        return 0

    monkeypatch.setattr("factorzen.ops.runner.run_ops_daily", fake_run)
    rc = main(["ops", "daily", "--config", str(p)])
    assert rc == 0
    assert seen["as_of"] == date.today()


def test_fz_ops_daily_propagates_return_code(monkeypatch, tmp_path):
    p = _write_cfg(tmp_path)
    monkeypatch.setattr(
        "factorzen.ops.runner.run_ops_daily", lambda cfg, as_of, notifier=None: 1
    )
    rc = main(["ops", "daily", "--config", str(p), "--date", "20260720"])
    assert rc == 1


def test_fz_ops_status_prints_summary(tmp_path, capsys):
    sd = tmp_path / "state"
    p = _write_cfg(tmp_path, state_dir=sd)
    OpsState(sd, date(2026, 7, 20)).mark_done("guard", detail="ok")
    rc = main(["ops", "status", "--config", str(p), "--date", "20260720"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "guard" in out and "done" in out
