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
    """把本 run 所有 AttemptRecord 写入 experiment_index（passed = passed_guardrails）。

    candidates: 可选。含 holdout_ic 的候选列表，用于归一化匹配后回填 holdout_ic 到记录。
    若有匹配，known_valid 排序即可按 holdout_ic 降序正常工作。
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
            "passed": a.passed_guardrails,
            "verdict": a.critic_verdict,
            "run_id": run_id,
        }
        # 回填 holdout_ic（归一化匹配）
        hic = hic_map.get(_normalize(a.expression))
        if hic is not None:
            rec["holdout_ic"] = hic
        records.append(rec)
    index.append(records)
