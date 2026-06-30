# src/factorzen/agents/roles/librarian.py
"""Librarian 角色：跨 session 长期记忆的读（recall）与写（record）。"""
from __future__ import annotations

from dataclasses import dataclass


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


def record(index, attempts, run_id: str) -> None:
    """把本 run 所有 AttemptRecord 写入 experiment_index（passed = passed_guardrails）。"""
    records = []
    for a in attempts:
        if not a.compile_ok or a.ic_train is None:
            continue
        records.append(
            {
                "expression": a.expression,
                "hypothesis": a.hypothesis,
                "ic_train": a.ic_train,
                "passed": a.passed_guardrails,
                "verdict": a.critic_verdict,
                "run_id": run_id,
            }
        )
    index.append(records)
