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
    budget: float | None = 1.0                     # Σw 目标（A股=1 全额；crypto 市场中性=0；None 不约束）
    gross_limit: float | None = None               # Σ|w| 上限（做空组合的杠杆约束）


def build_constraints(w, *, exposures, config: ConstraintConfig) -> list:
    """返回 cvxpy 约束列表。w 为 cp.Variable(n_stocks)。

    .. warning::
        **one-hot 行业哑变量陷阱**：``benchmark_weights=None`` 时中性目标为 0
        （绝对零暴露），仅在 exposures 列已去均值/标准化时可行。若传入原始
        one-hot 行业哑变量列（取值 0/1），``long_only=True`` + ``Σw=1`` 下要求
        所有行业暴露同时为 0，即组合在每个行业的总权重均为 0，与 ``Σw=1``
        矛盾，必然 infeasible。此时须传入 ``benchmark_weights``（基准行业暴露
        向量），将约束改为 ``X_s.T @ w == X_s.T @ w_benchmark``，使中性目标
        对齐基准而非绝对零点。
    """
    cons = []
    if config.budget is not None:
        cons.append(cp.sum(w) == config.budget)     # budget（A股=1 全额；crypto 市场中性=0）
    cons.append(w <= config.w_max)                  # box 个股上限
    if config.long_only:
        cons.append(w >= 0.0)
    else:
        cons.append(w >= -config.w_max)             # 做空下界（对称 box）
    if config.gross_limit is not None:
        cons.append(cp.norm1(w) <= config.gross_limit)  # 杠杆/毛敞口上限
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
