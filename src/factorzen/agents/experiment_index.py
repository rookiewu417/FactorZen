# src/factorzen/agents/experiment_index.py
"""跨 session 长期记忆：experiment_index.jsonl 读写 + 归一化查重 + 已知有效/无效。"""
from __future__ import annotations

import json
from pathlib import Path

from factorzen.discovery.expression import parse_expr, to_expr_string


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


class ExperimentIndex:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def append(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def seen_expressions(self) -> set[str]:
        return {_normalize(r["expression"]) for r in self.load() if "expression" in r}

    def known_invalid(self, k: int = 5) -> list[str]:
        recs = [r for r in self.load() if not r.get("passed", False)]
        recs.sort(key=lambda r: abs(r.get("ic_train") or 0.0))  # 最没用的优先
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]

    def known_valid(self, k: int = 5) -> list[str]:
        recs = [r for r in self.load() if r.get("passed", False)]
        recs.sort(key=lambda r: r.get("holdout_ic") or 0.0, reverse=True)
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]
