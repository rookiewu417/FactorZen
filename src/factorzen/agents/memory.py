# src/factorzen/agents/memory.py
"""session 记忆：Negative RAG 平面召回 + family-aware 并查集分组（无向量库/无 sklearn）。"""
from __future__ import annotations


def negative_recall(seen: list[tuple[str, float]], *, k: int = 3,
                    ic_threshold: float = 0.0) -> list[str]:
    """从 (表达式, IC) 历史里召回低 IC 负例，供 Negative RAG 注入 prompt。
    只取 |IC| < threshold 的，按 |IC| 升序（最没用优先），最多 k 个。"""
    low = [(e, ic) for e, ic in seen if abs(ic) < ic_threshold]
    low.sort(key=lambda t: abs(t[1]))
    return [e for e, _ in low[:k]]


def family_groups(corr_pairs: dict[tuple[str, str], float], names: list[str],
                  *, threshold: float = 0.7) -> list[set[str]]:
    """按两两相关 > threshold 并查集分组（family-aware 多样性）。"""
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for (a, b), c in corr_pairs.items():
        if a in parent and b in parent and abs(c) > threshold:
            union(a, b)
    groups: dict[str, set[str]] = {}
    for n in names:
        groups.setdefault(find(n), set()).add(n)
    return list(groups.values())
