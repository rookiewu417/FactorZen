# tests/test_portfolio_pipeline.py
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult


def _risk_result(n=6, k=3):
    rng = np.random.default_rng(2)
    names = ["size", "ind_A", "ind_B"]
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [1, 1, 1, 0, 0, 0]
    mat[:, 2] = [0, 0, 0, 1, 1, 1]
    F = rng.standard_normal((k, k))
    F = F @ F.T * 0.01
    return RiskModelResult(
        factor_exposures=ExposureMatrix([f"{i:06d}.SZ" for i in range(n)], names, mat),
        factor_covariance=F, specific_risk=np.full(n, 0.1), factor_names=names)


def test_run_portfolio_writes_products(tmp_path: Path):
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4, out_dir=str(tmp_path), run_id="t1")
    run_dir = Path(res["run_dir"])
    for f in ["weights.parquet", "attribution.csv", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f
    assert res["status"] == "optimal"
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "optimal" and "objective" in m


def test_run_portfolio_attribution_placeholder_flag(tmp_path: Path):
    """建仓时 stock_returns=zeros + factor_returns_latest={} → manifest 含占位标注。"""
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.zeros(6),       # 建仓时点全 0
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={},        # 无因子收益
                        risk_aversion=1.0, w_max=0.4,
                        out_dir=str(tmp_path), run_id="placeholder")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["return_attribution_available"] is False, (
        "stock_returns=zeros + factor_returns={}  时应标注归因不可用"
    )
    assert "return_attribution_note" in m and m["return_attribution_note"] is not None


def test_run_portfolio_attribution_available_when_returns_provided(tmp_path: Path):
    """传入真实收益时 return_attribution_available=True，note=None。"""
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4,
                        out_dir=str(tmp_path), run_id="real_returns")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["return_attribution_available"] is True
    assert m["return_attribution_note"] is None


def test_run_portfolio_records_signal_date(tmp_path: Path):
    """run_portfolio 传 signal_date 后，manifest.json 应记录该字段供 M7 sim 对齐。"""
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.zeros(6),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={},
                        risk_aversion=1.0, w_max=0.4,
                        out_dir=str(tmp_path), run_id="sig_date",
                        signal_date="2024-12-31")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m.get("signal_date") == "2024-12-31", (
        f"manifest.json 应含 signal_date='2024-12-31', 实际: {m.get('signal_date')!r}"
    )


def test_run_portfolio_infeasible_does_not_crash(tmp_path: Path):
    """w_max=0.1 → n*w_max=0.6 < 1，违反 Σw=1，优化器返回 infeasible，pipeline 不崩溃。"""
    rr = _risk_result()  # n=6 stocks
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    # w_max=0.1 with 6 stocks: max feasible sum = 0.6 < 1 → infeasible
    res = run_portfolio(
        alpha, rr,
        codes=rr.factor_exposures.codes,
        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
        sectors=["A", "A", "A", "B", "B", "B"],
        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
        risk_aversion=1.0, w_max=0.1, out_dir=str(tmp_path), run_id="infeasible",
    )
    run_dir = Path(res["run_dir"])

    # 4 产物文件必须存在
    for f in ["weights.parquet", "attribution.csv", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f"missing: {f}"

    # status 非 optimal
    assert res["status"] != "optimal"

    # weights.parquet 的 target_weight 全 0
    df_w = pl.read_parquet(run_dir / "weights.parquet")
    assert (df_w["target_weight"] == 0.0).all(), "infeasible 时 target_weight 应全 0"

    # attribution.csv 为空（0 行）
    df_a = pl.read_csv(run_dir / "attribution.csv")
    assert len(df_a) == 0, "infeasible 时 attribution 应为空"

    # manifest.json status 非 optimal
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] != "optimal"


def test_run_portfolio_manifest_has_reproducibility_fields(tmp_path: Path):
    """manifest.json 应含 command/git_dirty/pixi_lock_sha256/schema_version（复用 core.experiment 的
    build_manifest_base，而非各自手写精简版 manifest）。"""
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4, out_dir=str(tmp_path), run_id="repro1")
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())

    assert m["schema_version"] == "1"
    assert isinstance(m["git_dirty"], bool)
    assert isinstance(m["pixi_lock_sha256"], str) and m["pixi_lock_sha256"]
    assert isinstance(m["command"], list) and m["command"]
    assert m.get("git_sha")
    # 原有字段不应回归丢失
    assert m["run_id"] == "repro1"
    assert "duration_seconds" in m


def test_run_portfolio_manifest_command_override(tmp_path: Path):
    """显式传 command 时应原样记录，供复现当时具体怎么跑的。"""
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    res = run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
                        sectors=["A", "A", "A", "B", "B", "B"],
                        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
                        risk_aversion=1.0, w_max=0.4, out_dir=str(tmp_path), run_id="repro2",
                        command=["fz", "portfolio", "build", "--w-max", "0.4"])
    run_dir = Path(res["run_dir"])
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["command"] == ["fz", "portfolio", "build", "--w-max", "0.4"]


def test_run_portfolio_warns_when_turnover_set_without_prev_weights(tmp_path: Path, capsys):
    """L2：给了 turnover_budget 但无 prev_weights → 换手约束静默丢弃，须告警。"""
    rr = _risk_result()
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    run_portfolio(alpha, rr, codes=rr.factor_exposures.codes,
                  stock_returns=np.zeros(6), sectors=["A"] * 6,
                  factor_returns_latest={}, risk_aversion=1.0, w_max=0.4,
                  turnover_budget=0.3, out_dir=str(tmp_path), run_id="tw")
    err = capsys.readouterr().err
    assert "turnover" in err and "不生效" in err
