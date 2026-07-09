# src/factorzen/agents/roles/librarian.py
"""Librarian 角色：跨 session 长期记忆的读（recall）与写（record）。"""
from __future__ import annotations

from dataclasses import dataclass

from factorzen.discovery.expression import parse_expr, to_expr_string


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


@dataclass
class Recall:
    seen: set[str]
    known_invalid: list[str]
    known_valid: list[str]


def recall(index, *, k: int = 5) -> Recall:
    return Recall(
        seen=index.seen_expressions(),
        known_invalid=index.known_invalid(k=k),
        known_valid=index.known_valid(k=k),
    )


def record(
    index,
    attempts,
    run_id: str,
    *,
    candidates: list[dict] | None = None,
) -> None:
    """把本 run 所有 AttemptRecord 写入 experiment_index。

    落盘的是**事实**：`passed`（过了定量护栏）、`verdict`（Critic 裁决）、
    `decorrelated`（因与已有候选高度相关而未入候选池）。
    「可否借鉴」这个**决策**不在此计算，由 `ExperimentIndex.known_valid()` 综合三者推出
    ——一处判定，避免同一语义散落在写入侧的多个分支里互相矛盾。

    candidates: 可选。含 holdout_ic 的候选列表，用于归一化匹配后回填 holdout_ic 到记录，
    供 known_valid 按 |holdout_ic| 排序。
    """
    # 构建 holdout_ic 查找字典（归一化匹配，Important 2）
    hic_map: dict[str, float] = {}
    if candidates:
        for c in candidates:
            if "expression" in c and c.get("holdout_ic") is not None:
                hic_map[_normalize(c["expression"])] = c["holdout_ic"]

    records = []
    for a in attempts:
        if not a.compile_ok or a.ic_train is None:
            continue
        rec: dict = {
            "expression": a.expression,
            "hypothesis": a.hypothesis,
            "ic_train": a.ic_train,
            "passed": a.passed_guardrails,          # 事实：过了定量护栏
            "verdict": a.critic_verdict,            # 决策：Critic 裁决（known_valid 会读它）
            "decorrelated": a.decorrelated,         # 决策：与已有候选高度相关，未入候选池
            "run_id": run_id,
        }
        # 回填 holdout_ic（归一化匹配）
        hic = hic_map.get(_normalize(a.expression))
        if hic is not None:
            rec["holdout_ic"] = hic
        records.append(rec)
    index.append(records)
