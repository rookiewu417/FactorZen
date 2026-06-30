"""IC 序列的 moving block bootstrap 置信区间（保留时序自相关）。"""
from __future__ import annotations

import numpy as np


def block_bootstrap_ic_ci(
    ic_series,
    block_size: int = 10,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = ic.size
    if n < block_size or n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([ic[s : s + block_size] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
