# src/factorzen/discovery/guardrails.py
"""防过拟合护栏的单点判定 + DSR deflation 配方 + 池级 PBO——消除 M1 与 M5/M6 双路径漂移。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from factorzen.validation.deflated_sharpe import deflated_sharpe
from factorzen.validation.pbo import compute_pbo


@dataclass(frozen=True)
class DeflationBasis:
    """DSR deflation 的基准：trial 池 IR 的经验方差 + 与之**同源**的 N。

    R8：``n_trials`` 与 ``sharpe_variance`` 必须来自同一批 trial，否则 ``expected_max_sharpe``
    的 deflation 基准不自洽。因 ``expected_max_sharpe ∝ sqrt(sharpe_variance)``，而多样化
    trial 池的经验方差恒大于 ``deflated_sharpe`` 的 H0 默认值 ``1/n_obs``，漏传 sharpe_variance
    会让门槛系统性偏小、放行过拟合因子（漂移倍数 ``sqrt(var_emp × n_obs)``，实测 1.60x）。

    M1(`mining_session`) 与 Agent(`agents/nodes`) 必须**共同调用**本类构造基准、经
    `deflated_pvalue` 求 p 值。有架构守卫测试禁止任一路径直接调 ``deflated_sharpe``。
    """

    n_trials: int
    sharpe_variance: float

    @classmethod
    def from_ir_pool(cls, ir_pool: Sequence[float | None]) -> DeflationBasis:
        """从「评估过且拿到有效 IR」的 trial 池构造。

        None（死表达式）与 nan/inf 一律剔除：它们会同时污染方差与计数——把 0.0 之类的
        sentinel 灌进池子会拉低经验方差，使 deflation 基准算在垃圾上。
        池大小 < 2 时经验方差无意义，退化为 1.0（与 M1 既有行为一致）。
        """
        arr = np.asarray([x for x in ir_pool if x is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        return cls(n_trials=n, sharpe_variance=float(arr.var()) if n > 1 else 1.0)


def deflated_pvalue(sharpe: float, basis: DeflationBasis, n_obs: int) -> tuple[float, float]:
    """(dsr, pvalue)。两条挖掘路径的 DSR 唯一入口。

    ``n_obs`` 须是**该因子自己的有效 IC 天数**，不是 train 段日历交易日数——后者更大，
    会系统性放大显著性（``z ∝ sqrt(n_obs − 1)``）。
    """
    return deflated_sharpe(sharpe, basis.n_trials, n_obs,
                           sharpe_variance=basis.sharpe_variance)


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
