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


def unsupported_constraint_warnings(constraints: OptimizerConstraints) -> list[str]:
    """返回被显式设为非默认值、但 MaxSharpe/RiskParity 不会施加的约束描述。

    这两个优化器都硬编码归一到 sum(w)=1、不消费 turnover_limit/net_exposure/
    gross_exposure。用本函数把"约束被静默忽略"变成显式告警——避免用户以为
    turnover/net/gross 生效、实际未生效而使换手/暴露口径失真、net-of-cost
    研究结论不成立。默认值(turnover=None、net/gross=1.0)不列入（本就等价）。
    """
    msgs: list[str] = []
    if constraints.turnover_limit is not None:
        msgs.append(f"turnover_limit={constraints.turnover_limit}")
    if abs(constraints.net_exposure - 1.0) > 1e-9:
        msgs.append(f"net_exposure={constraints.net_exposure}")
    if abs(constraints.gross_exposure - 1.0) > 1e-9:
        msgs.append(f"gross_exposure={constraints.gross_exposure}")
    return msgs


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
