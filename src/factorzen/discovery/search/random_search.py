from __future__ import annotations

import numpy as np

from factorzen.discovery.expression import Constant, Feature, Node, OpNode
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS

_LEAVES = list(LEAF_FEATURES.keys())
_OPS = list(OPERATORS.keys())
_WINDOWS = [3, 5, 10, 20, 60]
_DEFAULT_MAX_DEPTH = 3  # 随机/遗传搜索的默认树深；search_space_max_lookback 据此推预热


def search_space_max_lookback() -> int:
    """搜索空间内表达式的最大 required_lookback（交易日）：最深路径全取最大窗口。

    = max(_WINDOWS) × _DEFAULT_MAX_DEPTH。`prepare_mining_daily` 的预热前缀据此设，
    保证搜索空间内任意表达式都不会因预热门（warmup_bars < required_lookback）被误拒。
    """
    return max(_WINDOWS) * _DEFAULT_MAX_DEPTH


def random_expression(
    rng: np.random.Generator, max_depth: int = _DEFAULT_MAX_DEPTH, leaves: list[str] | None = None
) -> Node:
    """按算子类型签名递归生成合法 AST。叶子为特征或（少量）常数。

    ``leaves`` 为可用叶子名列表(默认 A 股 _LEAVES)，传入其他市场叶子集即可
    在该市场叶子空间搜索。
    """
    leaf_names = _LEAVES if leaves is None else leaves
    if max_depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.1:
            return Constant(float(rng.choice([0.5, 1.0, 2.0])))
        return Feature(str(rng.choice(leaf_names)))
    op = str(rng.choice(_OPS))
    spec = OPERATORS[op]
    children = [random_expression(rng, max_depth - 1, leaf_names) for _ in range(spec.arity)]
    window = int(rng.choice(_WINDOWS)) if spec.has_window else None
    return OpNode(op, children, window)


class RandomSearcher:
    def __init__(
        self, rng: np.random.Generator, max_depth: int = _DEFAULT_MAX_DEPTH, leaves: list[str] | None = None
    ) -> None:
        self.rng = rng
        self.max_depth = max_depth
        self.leaves = leaves

    def propose(self) -> Node:
        return random_expression(self.rng, self.max_depth, self.leaves)
