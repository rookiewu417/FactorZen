# src/factorzen/discovery/guardrails.py
"""防过拟合护栏的单点判定 + 池级 PBO——消除 M1 与 M5/M6 双路径漂移。"""
from __future__ import annotations

import numpy as np
import polars as pl

from factorzen.validation.pbo import compute_pbo


def guardrail_passed(
    *,
    ic_train: float | None,
    holdout_ic: float | None,
    dsr_pvalue: float | None,
    ci_low: float | None,
    ci_high: float | None = None,
    dsr_alpha: float = 0.05,
) -> bool:
    """DSR 显著(pval<dsr_alpha) + holdout 同号 + holdout CI 方向门槛。任一 None/NaN → False。"""
    required = [ic_train, holdout_ic, dsr_pvalue, ci_low]
    if any(v is None for v in required):
        return False
    if any(v != v for v in required):
        return False
    same_sign = (holdout_ic > 0) == (ic_train > 0)  # type: ignore[operator]
    dsr_sig = dsr_pvalue < dsr_alpha  # type: ignore[operator]
    if ic_train > 0:  # type: ignore[operator]
        ci_ok = ci_low > 0  # type: ignore[operator]
    elif ci_high is not None:
        ci_ok = ci_high < 0
    else:
        ci_ok = ci_low > 0  # type: ignore[operator]
    return bool(dsr_sig and same_sign and ci_ok)


def pool_pbo(
    factor_dfs: list[pl.DataFrame],
    fwd_returns: pl.DataFrame,
    *,
    n_splits: int = 10,
    max_cand: int = 30,
) -> float:
    """对候选池因子帧算池级 PBO（CSCV）。候选<2 或周期不足 → nan。与 mining_session._pool_pbo 共享 compute_pbo。"""
    from factorzen.daily.evaluation.ic_analysis import compute_rank_ic
    from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore

    series: list[np.ndarray] = []
    dates_ref = None
    for fdf in factor_dfs[:max_cand]:
        try:
            clean = cross_sectional_zscore(fdf, col="factor_value").rename(
                {"factor_value_z": "factor_clean"}
            )
            ic_res = compute_rank_ic(
                clean.select(["trade_date", "ts_code", "factor_clean"]),
                fwd_returns, factor_col="factor_clean", frequency="daily",
            )
            ser = ic_res.ic_series.sort("trade_date")
            if dates_ref is None:
                dates_ref = ser["trade_date"]
            ser = ser.join(
                pl.DataFrame({"trade_date": dates_ref}), on="trade_date", how="right"
            ).sort("trade_date")
            series.append(ser["ic"].fill_null(0.0).to_numpy())
        except Exception:
            continue
    if len(series) < 2:
        return float("nan")
    return compute_pbo(np.vstack(series), n_splits=n_splits)
