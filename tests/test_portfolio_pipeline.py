# tests/test_portfolio_pipeline.py
import json
from pathlib import Path

import numpy as np

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
