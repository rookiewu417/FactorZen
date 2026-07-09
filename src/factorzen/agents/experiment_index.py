# src/factorzen/agents/experiment_index.py
"""跨 session 长期记忆：experiment_index.jsonl 读写 + 归一化查重 + 已知有效/无效。"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import fcntl  # POSIX 文件锁（Linux 优先；Windows 无此模块时降级为无锁追加）
except ImportError:  # pragma: no cover - 仅非 POSIX 平台
    fcntl = None  # type: ignore[assignment]

from factorzen.discovery.expression import parse_expr, to_expr_string

# Critic 否决了「方向」的裁决 → 该因子不再作为「可借鉴的已验证有效方向」喂给后续假设生成。
# revise_expr 不在此列：方向对、只是表达式需改，思路仍值得借鉴。
_VETOED_VERDICTS = frozenset({"drop", "revise_hypothesis"})


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
        # team workers / 并行 session 会并发写同一 jsonl；无锁多次 write 会交错、
        # 产出损坏行。整批组装成单个 payload + POSIX 独占锁一次写入，保证行原子、不交错。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        if not payload:
            return
        with self.path.open("a") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(payload)
                f.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def seen_expressions(self) -> set[str]:
        return {_normalize(r["expression"]) for r in self.load() if "expression" in r}

    def known_invalid(self, k: int = 5) -> list[str]:
        """「已验证无效」= 没过定量护栏。按 |IC| 升序（最没用的优先）喂给 LLM 作负例。

        注意判据是 `not passed` 这个**事实**——被去相关剔除、或被 Critic 否决的因子
        `passed` 仍为 True，它们不是「无效因子」，不该混进负例污染 LLM 的认知。
        """
        recs = [r for r in self.load() if not r.get("passed", False)]
        recs.sort(key=lambda r: abs(r.get("ic_train") or 0.0))  # 最没用的优先
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]

    def known_valid(self, k: int = 5) -> list[str]:
        """「可供借鉴」是一个**决策**，由事实（passed）与两类否决共同推出，此处集中判定。

        - `passed`：过了定量护栏（不可变事实，见 `AttemptRecord.passed_guardrails`）
        - `verdict not in {drop, revise_hypothesis}`：Critic 未否决这个**方向**
          （`revise_expr` = 方向对、表达式需改 → 思路仍值得借鉴，保留）
        - `not decorrelated`：未因与已有候选高度相关而被剔除（重复的思路无需再借鉴）

        排序按 **|holdout_ic|** 降序：护栏明确接纳负 IC 反转因子
        （`guardrail_passed` 的 `same_sign` + `ci_high<0` 分支），带符号排序会把最强的
        反转因子挤到末尾、被 top-k 截断，系统性把 LLM 的借鉴方向偏离反转因子族。
        """
        recs = [
            r for r in self.load()
            if r.get("passed", False)
            and r.get("verdict") not in _VETOED_VERDICTS
            and not r.get("decorrelated", False)
        ]
        recs.sort(key=lambda r: abs(r.get("holdout_ic") or 0.0), reverse=True)
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]
