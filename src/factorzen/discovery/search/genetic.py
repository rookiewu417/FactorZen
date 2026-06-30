from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np

from factorzen.discovery.expression import Node, OpNode, complexity
from factorzen.discovery.search.random_search import random_expression


def _all_nodes(node: Node) -> list[Node]:
    out = [node]
    if isinstance(node, OpNode):
        for c in node.children:
            out.extend(_all_nodes(c))
    return out


def _replace_random_subtree(root: Node, new_sub: Node, rng: np.random.Generator) -> Node:
    root = copy.deepcopy(root)
    nodes = [n for n in _all_nodes(root) if isinstance(n, OpNode)]
    if not nodes:
        return new_sub
    target = nodes[int(rng.integers(len(nodes)))]
    if target.children:
        target.children[int(rng.integers(len(target.children)))] = new_sub
    return root


def crossover(a: Node, b: Node, rng: np.random.Generator) -> Node:
    donor_subtrees = _all_nodes(b)
    donor = copy.deepcopy(donor_subtrees[int(rng.integers(len(donor_subtrees)))])
    return _replace_random_subtree(a, donor, rng)


def mutate(node: Node, rng: np.random.Generator, max_depth: int = 3) -> Node:
    return _replace_random_subtree(node, random_expression(rng, max_depth=max_depth - 1), rng)


class GeneticSearcher:
    def __init__(self, rng: np.random.Generator, max_depth: int = 3) -> None:
        self.rng = rng
        self.max_depth = max_depth

    def evolve(
        self,
        score_fn: Callable[[Node], float],
        pop_size: int = 30,
        generations: int = 8,
        elite: int = 2,
    ) -> list[Node]:
        pop = [random_expression(self.rng, self.max_depth) for _ in range(pop_size)]
        for _ in range(generations):
            scored = sorted(pop, key=lambda n: score_fn(n), reverse=True)
            survivors = scored[: max(elite, pop_size // 2)]
            children = list(scored[:elite])
            while len(children) < pop_size:
                a = survivors[int(self.rng.integers(len(survivors)))]
                b = survivors[int(self.rng.integers(len(survivors)))]
                child = crossover(a, b, self.rng)
                if self.rng.random() < 0.3:
                    child = mutate(child, self.rng, self.max_depth)
                if complexity(child) <= 12:  # 防膨胀
                    children.append(child)
            pop = children
        return sorted(pop, key=lambda n: score_fn(n), reverse=True)
