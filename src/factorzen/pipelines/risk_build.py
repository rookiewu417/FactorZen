"""风险模型构建 pipeline：build → 落产物 + 轻量风险报告。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorzen.core.experiment import build_manifest_base
from factorzen.risk import RiskModel


def risk_lookback_start(start: str, calendar_days: int = 420) -> str:
    """风险模型的滚动风格因子（momentum rolling_sum(252)、growth shift(252) 等最长
    252 交易日窗）需要 start 之前的历史预热，否则窗口早期这些因子全空、模型静默
    退化为少数非滚动因子。返回 start 往前推 calendar_days 日历日的 "YYYYMMDD"。

    252 交易日在 A 股约需 380 日历日，默认 420 日历日留有春节等长假余量。
    """
    from datetime import datetime, timedelta

    d = datetime.strptime(start.replace("-", ""), "%Y%m%d").date() - timedelta(days=calendar_days)
    return d.strftime("%Y%m%d")


def load_risk_inputs(
    loader: Any, start: str, end: str, universe_codes: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """为风险模型构建拉取 daily / daily_basic，并补足 lookback 历史。

    生产 CLI 若只 fetch [start,end]，窗口首日的滚动风格因子（252/60 日窗）全空，
    RiskModel.build 会把因子集钉死在退化的截面并跳过大量交易日。这里统一用
    risk_lookback_start 往前多拉历史用于因子预热；build 内部仍只对 [start,end]
    做截面回归（按 trade_date 过滤），故不改变回归区间、只补数据。

    Args:
        loader: 提供 fetch_daily / fetch_daily_basic 的数据加载器（core.loader 模块）。
        start, end: 回归区间 "YYYYMMDD"。
        universe_codes: 股票池代码，用于收窄行情。

    Returns:
        (daily, daily_basic)，含 [lookback_start, end] 且已按 universe 过滤。
    """
    lb_start = risk_lookback_start(start)
    daily = loader.fetch_daily(lb_start, end).filter(pl.col("ts_code").is_in(universe_codes))
    daily_basic = loader.fetch_daily_basic(lb_start, end).filter(
        pl.col("ts_code").is_in(universe_codes)
    )
    return daily, daily_basic


def run_risk_build(daily, daily_basic, stocks, start, end, *, out_dir="workspace/risk_models",
                   cov_half_life=90, nw_lags=2, spec_half_life=90, spec_shrinkage=0.3,
                   run_id=None, command: list[str] | None = None) -> dict:
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
            pl.DataFrame(exp.matrix, schema=names, orient="row"))
    else:
        exp_df = pl.DataFrame({"ts_code": []})
    exp_df.write_parquet(run_dir / "exposures.parquet")
    # factor_covariance.parquet
    cov = result.factor_covariance
    cov_df = pl.DataFrame(cov, schema=names, orient="row") if cov.size else pl.DataFrame()
    cov_df.write_parquet(run_dir / "factor_covariance.parquet")
    # specific_risk.parquet
    sr = result.specific_risk
    sr_df = pl.DataFrame({"ts_code": exp.codes, "specific_risk": sr.tolist()}) \
        if exp.n_stocks and sr.size else pl.DataFrame({"ts_code": [], "specific_risk": []})
    sr_df.write_parquet(run_dir / "specific_risk.parquet")
    # factor_returns.parquet
    result.factor_returns.write_parquet(run_dir / "factor_returns.parquet")

    # 等权组合风险分解示例（先算，后写 CSV）
    decomp = {}
    if exp.n_stocks > 0:
        w = np.full(exp.n_stocks, 1.0 / exp.n_stocks)
        decomp = model.decompose_risk(w, result)

    # ── 轻量报告 risk_summary.csv（长表：section / metric / value）──
    # 人读 30 秒看懂风险来自哪：因子波动 / 特质风险分布 / R² / 风格暴露 / 组合分解
    rows: list[dict] = []

    # §1 因子波动
    factor_vol = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov.size else np.array([])
    for i, n in enumerate(names):
        rows.append({"section": "factor_vol", "metric": n, "value": float(factor_vol[i])})

    # §2 特质风险分布
    if sr.size:
        rows.append({"section": "specific_risk", "metric": "mean",   "value": float(sr.mean())})
        rows.append({"section": "specific_risk", "metric": "median", "value": float(np.median(sr))})
        rows.append({"section": "specific_risk", "metric": "p25",    "value": float(np.percentile(sr, 25))})
        rows.append({"section": "specific_risk", "metric": "p75",    "value": float(np.percentile(sr, 75))})
        rows.append({"section": "specific_risk", "metric": "max",    "value": float(sr.max())})

    # §3 平均回归 R²
    rows.append({"section": "r_squared", "metric": "r_squared", "value": float(result.r_squared)})

    # §4 风格暴露统计（非 ind_ 行业列）
    style_mask = np.array([not n.startswith("ind_") for n in names], dtype=bool)
    style_names = [n for n in names if not n.startswith("ind_")]
    if exp.n_stocks > 0 and style_names:
        style_matrix = exp.matrix[:, style_mask]          # (n_stocks, n_style)
        style_mean = style_matrix.mean(axis=0)
        style_std  = style_matrix.std(axis=0)
        for j, sn in enumerate(style_names):
            rows.append({"section": "style_exposure", "metric": f"{sn}_mean", "value": float(style_mean[j])})
            rows.append({"section": "style_exposure", "metric": f"{sn}_std",  "value": float(style_std[j])})

    # §5 等权组合风险分解示例
    if decomp:
        tr    = decomp.get("total_risk", 0.0)
        fr    = decomp.get("factor_risk", 0.0)
        srisk = decomp.get("specific_risk", 0.0)
        rows.append({"section": "decomp", "metric": "total_risk",    "value": round(tr, 6)})
        rows.append({"section": "decomp", "metric": "factor_risk",   "value": round(fr, 6)})
        rows.append({"section": "decomp", "metric": "specific_risk", "value": round(srisk, 6)})
        # 分解是方差加和：total_var = factor_var + specific_var；占比用方差比（std² / std²）
        rows.append({"section": "decomp", "metric": "factor_pct",    "value": round(fr**2 / tr**2, 4) if tr > 0 else 0.0})
        rows.append({"section": "decomp", "metric": "specific_pct",  "value": round(srisk**2 / tr**2, 4) if tr > 0 else 0.0})

    pl.DataFrame(rows if rows else {"section": [], "metric": [], "value": []}) \
        .write_csv(run_dir / "risk_summary.csv")

    # 可复现性基础字段（schema_version/git_sha/git_dirty/pixi_lock_sha256/command/config/start_ts）
    # 复用 core.experiment.build_manifest_base，与 daily_single/generate_report 的 manifest 同源，
    # 不再各自手写精简版 _git_sha()。command 缺省时取当前进程 argv（记录“当时具体怎么跑的”）。
    build_config = {"start": start, "end": end, "out_dir": str(out_dir),
                    "cov_half_life": cov_half_life, "nw_lags": nw_lags,
                    "spec_half_life": spec_half_life, "spec_shrinkage": spec_shrinkage}
    manifest = build_manifest_base(command if command is not None else list(sys.argv), build_config)
    manifest.update({
        "run_id": rid, "start": start, "end": end, "universe_size": exp.n_stocks,
        "cov_half_life": cov_half_life, "nw_lags": nw_lags,
        "spec_half_life": spec_half_life, "spec_shrinkage": spec_shrinkage,
        "r_squared": result.r_squared, "factor_names": names,
        "specific_risk_mean": float(sr.mean()) if sr.size else 0.0,
        "equal_weight_decomp": {k: round(v, 6) for k, v in decomp.items()},
        "duration_seconds": round(time.perf_counter() - t0, 3),
    })
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    return {"run_dir": str(run_dir), "r_squared": result.r_squared, "factor_names": names}
