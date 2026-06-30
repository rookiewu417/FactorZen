"""风险模型构建 pipeline：build → 落产物 + 轻量风险报告。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.risk import RiskModel


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_risk_build(daily, daily_basic, stocks, start, end, *, out_dir="workspace/risk_models",
                   cov_half_life=90, nw_lags=2, spec_half_life=90, spec_shrinkage=0.3,
                   run_id=None) -> dict:
    t0 = time.perf_counter()
    model = RiskModel(cov_half_life=cov_half_life, nw_lags=nw_lags,
                      spec_half_life=spec_half_life, spec_shrinkage=spec_shrinkage)
    result = model.build(daily, daily_basic, stocks, start, end)

    rid = run_id or f"risk_{start}_{end}"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    names = result.factor_names
    exp = result.factor_exposures
    # exposures.parquet
    if exp.n_stocks > 0:
        exp_df = pl.DataFrame({"ts_code": exp.codes}).hstack(
            pl.DataFrame(exp.matrix, schema=names))
    else:
        exp_df = pl.DataFrame({"ts_code": []})
    exp_df.write_parquet(run_dir / "exposures.parquet")
    # factor_covariance.parquet
    cov = result.factor_covariance
    cov_df = pl.DataFrame(cov, schema=names) if cov.size else pl.DataFrame()
    cov_df.write_parquet(run_dir / "factor_covariance.parquet")
    # specific_risk.parquet
    sr = result.specific_risk
    sr_df = pl.DataFrame({"ts_code": exp.codes, "specific_risk": sr.tolist()}) \
        if exp.n_stocks and sr.size else pl.DataFrame({"ts_code": [], "specific_risk": []})
    sr_df.write_parquet(run_dir / "specific_risk.parquet")
    # factor_returns.parquet
    result.factor_returns.write_parquet(run_dir / "factor_returns.parquet")

    # ── 轻量报告 risk_summary.csv ──
    factor_vol = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov.size else np.array([])
    summary_rows = [{"factor": n, "factor_vol": float(factor_vol[i])} for i, n in enumerate(names)] \
        if factor_vol.size else []
    pl.DataFrame(summary_rows if summary_rows else {"factor": [], "factor_vol": []}) \
        .write_csv(run_dir / "risk_summary.csv")

    # 等权组合风险分解示例
    decomp = {}
    if exp.n_stocks > 0:
        w = np.full(exp.n_stocks, 1.0 / exp.n_stocks)
        decomp = model.decompose_risk(w, result)

    manifest = {"run_id": rid, "start": start, "end": end, "universe_size": exp.n_stocks,
                "cov_half_life": cov_half_life, "nw_lags": nw_lags,
                "spec_half_life": spec_half_life, "spec_shrinkage": spec_shrinkage,
                "r_squared": result.r_squared, "factor_names": names,
                "specific_risk_mean": float(sr.mean()) if sr.size else 0.0,
                "equal_weight_decomp": {k: round(v, 6) for k, v in decomp.items()},
                "git_sha": _git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    return {"run_dir": str(run_dir), "r_squared": result.r_squared, "factor_names": names}
