"""PBO（Probability of Backtest Overfitting）via CSCV（López de Prado 2016）。

把时间分 S 块，枚举所有 S/2 块为 IS、其余为 OOS 的对称划分；每种里在 IS
选最优候选，看它在 OOS 的相对秩。PBO = IS 最优在 OOS 落后半区的频率。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def compute_pbo(perf_matrix: np.ndarray, n_splits: int = 10) -> float:
    perf = np.asarray(perf_matrix, dtype=float)
    n_cand, n_periods = perf.shape
    if n_cand < 2 or n_periods < n_splits or n_splits % 2 != 0:
        return float("nan")
    block = n_periods // n_splits
    # 每块每候选的平均表现 → (n_splits, n_cand)
    block_means = np.array([perf[:, i * block : (i + 1) * block].mean(axis=1) for i in range(n_splits)])
    half = n_splits // 2
    logit_list: list[float] = []
    for is_idx in combinations(range(n_splits), half):
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_perf = block_means[list(is_idx)].mean(axis=0)
        oos_perf = block_means[oos_idx].mean(axis=0)
        best = int(np.argmax(is_perf))
        # best 在 OOS 的相对秩 ∈ (0,1)
        rank = float((oos_perf <= oos_perf[best]).sum())  # 含自身
        rel = rank / (n_cand + 1)
        rel = min(max(rel, 1e-6), 1 - 1e-6)
        logit_list.append(np.log(rel / (1 - rel)))
    logits: np.ndarray = np.asarray(logit_list)
    return float((logits <= 0).mean())
