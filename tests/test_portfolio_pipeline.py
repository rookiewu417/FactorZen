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
