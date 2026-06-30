from __future__ import annotations

import numpy as np

from factorzen.discovery.expression import Constant, Feature, Node, OpNode
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS

_LEAVES = list(LEAF_FEATURES.keys())
_OPS = list(OPERATORS.keys())
_WINDOWS = [3, 5, 10, 20, 60]


def random_expression(rng: np.random.Generator, max_depth: int = 3) -> Node:
    """按算子类型签名递归生成合法 AST。叶子为特征或（少量）常数。"""
    if max_depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.1:
            return Constant(float(rng.choice([0.5, 1.0, 2.0])))
        return Feature(str(rng.choice(_LEAVES)))
    op = str(rng.choice(_OPS))
    spec = OPERATORS[op]
    children = [random_expression(rng, max_depth - 1) for _ in range(spec.arity)]
    window = int(rng.choice(_WINDOWS)) if spec.has_window else None
    return OpNode(op, children, window)


class RandomSearcher:
    def __init__(self, rng: np.random.Generator, max_depth: int = 3) -> None:
        self.rng = rng
        self.max_depth = max_depth

    def propose(self) -> Node:
        return random_expression(self.rng, self.max_depth)
