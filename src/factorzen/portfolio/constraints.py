"""组合优化约束构造（cvxpy）：box / budget / 行业风格中性 / 换手。"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class ConstraintConfig:
    w_max: float = 0.05
    long_only: bool = True
    neutral_factors: list[str] | None = None      # 要中性的 exposure 列名（风格/行业）
    benchmark_weights: np.ndarray | None = None    # 中性目标基准权重（None → 中性到 0）
    turnover_budget: float | None = None
    prev_weights: np.ndarray | None = None


def build_constraints(w, *, exposures, config: ConstraintConfig) -> list:
    """返回 cvxpy 约束列表。w 为 cp.Variable(n_stocks)。"""
    cons = [cp.sum(w) == 1.0]                       # budget 全额
    if config.long_only:
        cons.append(w >= 0.0)
    cons.append(w <= config.w_max)                  # box 个股上限
    # 行业/风格中性：选定列暴露 == benchmark 暴露（或 0）
    if config.neutral_factors:
        idx = [exposures.factor_names.index(n) for n in config.neutral_factors
               if n in exposures.factor_names]
        if idx:
            X_s = exposures.matrix[:, idx]          # (n, len(idx))
            target = (X_s.T @ config.benchmark_weights
                      if config.benchmark_weights is not None
                      else np.zeros(len(idx)))
            cons.append(X_s.T @ w == target)
    # 换手：L1
    if config.turnover_budget is not None and config.prev_weights is not None:
        cons.append(cp.norm1(w - config.prev_weights) <= config.turnover_budget)
    return cons
