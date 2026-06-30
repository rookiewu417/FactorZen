"""组合构建 pipeline：α + M3 风险模型 → 优化 → 归因 → 落盘。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.attribution.brinson import brinson_attribution
from factorzen.attribution.risk_attribution import risk_factor_attribution
from factorzen.portfolio.constraints import ConstraintConfig
from factorzen.portfolio.optimizer import optimize_portfolio
from factorzen.risk.model import RiskModel


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_portfolio(alpha, risk_result, *, codes, stock_returns, sectors,
                  factor_returns_latest, bench_weights=None, prev_weights=None,
                  risk_aversion=1.0, neutral_factors=None, turnover_budget=None,
                  w_max=0.05, out_dir="workspace/portfolios", run_id=None,
                  signal_date: str | None = None) -> dict:
    t0 = time.perf_counter()
    cfg = ConstraintConfig(w_max=w_max, neutral_factors=neutral_factors,
                           benchmark_weights=bench_weights,
                           turnover_budget=turnover_budget, prev_weights=prev_weights)
    opt = optimize_portfolio(alpha, risk_result, risk_aversion=risk_aversion,
                             constraint_config=cfg)
    rid = run_id or "portfolio"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    w = opt.weights if opt.weights is not None else np.zeros(len(codes))

    pl.DataFrame({"ts_code": codes, "target_weight": w.tolist(),
                  "prev_weight": (prev_weights.tolist() if prev_weights is not None
                                  else [0.0] * len(codes))}).write_parquet(run_dir / "weights.parquet")

    # 归因（仅 optimal 时有意义）
    attrib_rows = []
    if opt.weights is not None:
        ra = risk_factor_attribution(w, risk_result, factor_returns_latest,
                                     stock_returns=np.asarray(stock_returns))
        for k, v in ra.factor_return_contrib.items():
            attrib_rows.append({"type": "factor_return", "key": k, "value": v})
        attrib_rows.append({"type": "specific_return", "key": "specific", "value": ra.specific_return})
        bench = bench_weights if bench_weights is not None else np.full(len(codes), 1.0 / len(codes))
        br = brinson_attribution(w, bench, np.asarray(stock_returns), sectors)
        for s, v in br.allocation.items():
            attrib_rows.append({"type": "brinson_allocation", "key": s, "value": v})
        for s, v in br.selection.items():
            attrib_rows.append({"type": "brinson_selection", "key": s, "value": v})
    pl.DataFrame(attrib_rows if attrib_rows else {"type": [], "key": [], "value": []}) \
        .write_csv(run_dir / "attribution.csv")

    # 风险摘要（复用 M3 decompose）
    risk_rows = []
    if opt.weights is not None:
        decomp = RiskModel().decompose_risk(w, risk_result)
        risk_rows = [{"metric": k, "value": float(v)} for k, v in decomp.items()]
    pl.DataFrame(risk_rows if risk_rows else {"metric": [], "value": []}) \
        .write_csv(run_dir / "risk_summary.csv")

    # 收益归因可用性标注：建仓时点无持仓期收益时，Brinson/factor_return 为占位 0
    _sr = np.asarray(stock_returns)
    _attrib_placeholder = bool(np.all(_sr == 0) or len(factor_returns_latest) == 0)
    manifest = {"run_id": rid, "signal_date": signal_date, "status": opt.status,
                "objective": opt.objective_value,
                "n_holdings": int((w > 1e-6).sum()), "risk_aversion": risk_aversion,
                "w_max": w_max, "neutral_factors": neutral_factors,
                "turnover_budget": turnover_budget,
                "turnover": (float(np.abs(w - prev_weights).sum()) if prev_weights is not None else None),
                "return_attribution_available": not _attrib_placeholder,
                "return_attribution_note": (
                    "建仓时点无持仓期收益，收益归因(Brinson/factor_return)为占位 0；风险归因(risk_summary)有效"
                    if _attrib_placeholder else None),
                "git_sha": _git_sha(), "duration_seconds": round(time.perf_counter() - t0, 3)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"run_dir": str(run_dir), "status": opt.status,
            "n_holdings": manifest["n_holdings"], "objective": opt.objective_value}


def compute_sector_returns(daily: pl.DataFrame, stocks: pl.DataFrame) -> pl.DataFrame:
    """行业等权收益：daily(pct_chg) + stocks(industry) → [trade_date, sector, ret]。"""
    j = daily.join(stocks.select(["ts_code", "industry"]), on="ts_code")
    return (j.group_by(["trade_date", "industry"])
            .agg((pl.col("pct_chg") / 100.0).mean().alias("ret"))
            .rename({"industry": "sector"}))
