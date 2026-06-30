"""Brinson 归因（M4 版，股票级输入）：单期 BHB 配置效应 + 选股效应(交互归入选股)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrinsonResult:
    allocation: dict[str, float]   # 各行业配置效应
    selection: dict[str, float]    # 各行业选股效应(含交互)
    total_excess: float


def brinson_attribution(port_weights, bench_weights, stock_returns, sectors) -> BrinsonResult:
    """单期 Brinson。各行业:

    配置 = (w_p − w_b)·(r_b_sector − r_b_total)
    选股 = w_p·(r_p_sector − r_b_sector)   ← 交互项归入选股

    守恒: Σ(配置+选股) = port_ret − bench_ret。
    """
    w_p = np.asarray(port_weights, dtype=float)
    w_b = np.asarray(bench_weights, dtype=float)
    r = np.asarray(stock_returns, dtype=float)
    secs = ["" if s is None else s for s in sectors]
    uniq = sorted(set(secs))
    r_b_total = float(w_b @ r)
    allocation: dict[str, float] = {}
    selection: dict[str, float] = {}
    for s in uniq:
        m = np.array([x == s for x in secs])
        wp_s = float(w_p[m].sum())
        wb_s = float(w_b[m].sum())
        r_p_s = float(w_p[m] @ r[m] / wp_s) if wp_s > 1e-12 else 0.0
        r_b_s = float(w_b[m] @ r[m] / wb_s) if wb_s > 1e-12 else 0.0
        allocation[s] = (wp_s - wb_s) * (r_b_s - r_b_total)
        selection[s] = wp_s * (r_p_s - r_b_s)
    total_excess = float(w_p @ r) - r_b_total
    return BrinsonResult(allocation=allocation, selection=selection, total_excess=total_excess)
