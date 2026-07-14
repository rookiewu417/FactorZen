"""全仓 numpy 侧 Spearman / RankIC 的**单一实现**。

口径与 polars ``rank(method="average")`` 后做 Pearson 一致（ties 取组内平均秩，
1-based）。禁止各处再内联双 ``argsort``（ordinal rank 在 ties 下依赖行序，
与主 IC 口径漂移——LightGBM 预测等高 ties 场景尤甚）。
"""
from __future__ import annotations

import numpy as np


def avg_rank(x: np.ndarray) -> np.ndarray:
    """1-based 平均秩；``mergesort`` 稳定，ties 组内秩相同。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_avg_rank(a: np.ndarray, b: np.ndarray) -> float | None:
    """单日 Spearman = Pearson(avg_rank(a), avg_rank(b))；退化截面 → None。

    守卫：size < 2、任一侧 std < 1e-12、相关非有限 → None。
    """
    if a.size < 2:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    ra = avg_rank(a)
    rb = avg_rank(b)
    c = float(np.corrcoef(ra, rb)[0, 1])
    return c if np.isfinite(c) else None
