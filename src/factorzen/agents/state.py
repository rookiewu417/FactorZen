"""Agent 闭环的显式状态（JSON 可序列化 dataclass）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AttemptRecord:
    iteration: int
    hypothesis: str
    expression: str
    compile_ok: bool
    ic_train: float | None
    passed_guardrails: bool
    critic_verdict: str | None
    error: str | None
    ir_train: float | None = None
    turnover: float | None = None


@dataclass
class AgentState:
    seed: int
    iteration: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    seen_expressions: set[str] = field(default_factory=set)
    negative_examples: list[str] = field(default_factory=list)
    pbo: float | None = None

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "iteration": self.iteration,
            "attempts": [asdict(a) for a in self.attempts],
            "candidates": self.candidates,
            "seen_expressions": sorted(self.seen_expressions),
            "negative_examples": self.negative_examples,
            "pbo": self.pbo,
        }
