"""组合优化器抽象基类与约束定义。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizerConstraints:
    """组合优化约束。"""

    max_weight: float = 0.1  # 单资产最大权重（多头）
    min_weight: float = 0.0  # 单资产最小权重（0=不允许做空）
    gross_exposure: float = 1.0  # 总杠杆上限（|w|之和）
    net_exposure: float = 1.0  # 净暴露上限（w之和）
    turnover_limit: float | None = None  # 单期换手限制（None=不限）
    prev_weights: np.ndarray = field(default_factory=lambda: np.array([]))  # 上期权重（用于换手约束）


class PortfolioOptimizer(ABC):
    """组合优化器抽象基类。"""

    @abstractmethod
    def solve(
        self,
        expected_returns: np.ndarray,  # shape (n,)
        cov_matrix: np.ndarray,  # shape (n, n)
        constraints: OptimizerConstraints,
    ) -> np.ndarray:
        """求解最优权重向量。

        Returns:
            weights: shape (n,)，求解失败时返回 constraints.prev_weights 或等权。
        """
