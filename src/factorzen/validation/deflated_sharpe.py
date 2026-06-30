"""Deflated Sharpe Ratio（Bailey & López de Prado 2014）。

挖掘 = 多重检验：从 N 个候选里选最优，观测 Sharpe 被夸大。DSR 用「期望最大
Sharpe」作为 deflation 基准，评估观测 Sharpe 扣除多重检验后是否仍显著。
本项目以因子 IC 的 IR 作为「Sharpe」、IC 序列长度作为样本数。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """N 次独立试验下、零假设期望最大 Sharpe（deflation 基准）。"""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    e = np.e
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(sharpe_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_variance: float | None = None,
) -> tuple[float, float]:
    """返回 (dsr, pvalue)。dsr=PSR(期望最大 Sharpe)，pvalue=1-dsr，<0.05 视为显著。"""
    if n_obs < 2:
        return (0.0, 1.0)
    if sharpe_variance is None:
        sharpe_variance = 1.0 / n_obs  # H0 下 per-period Sharpe 的方差近似 1/T
    sr0 = expected_max_sharpe(sharpe_variance, n_trials)
    denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return (0.0, 1.0)
    z = (sharpe - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    dsr = float(norm.cdf(z))
    return (dsr, 1.0 - dsr)
