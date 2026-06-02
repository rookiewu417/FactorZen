"""Schema and parser for compact LLM factor explanations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

Rating = Literal["strong", "moderate", "weak", "invalid"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class LLMExplanation:
    rating: Rating
    confidence: Confidence
    factor_intuition: str
    evidence_assessment: str
    risk_flags: list[str]
    usage_suggestion: str
    next_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_short_list(value: Any, limit: int) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [str(item).strip() for item in value if str(item).strip()]
    return items[:limit]


def _required_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _from_dict(data: dict[str, Any]) -> LLMExplanation | None:
    rating = data.get("rating")
    confidence = data.get("confidence")
    if rating not in {"strong", "moderate", "weak", "invalid"}:
        return None
    if confidence not in {"high", "medium", "low"}:
        return None

    factor_intuition = _required_str(data.get("factor_intuition"))
    evidence_assessment = _required_str(data.get("evidence_assessment"))
    usage_suggestion = _required_str(data.get("usage_suggestion"))
    risk_flags = _as_short_list(data.get("risk_flags"), 5)
    next_steps = _as_short_list(data.get("next_steps"), 4)
    if factor_intuition is None or evidence_assessment is None or usage_suggestion is None:
        return None
    if risk_flags is None or next_steps is None:
        return None

    return LLMExplanation(
        rating=rating,
        confidence=confidence,
        factor_intuition=factor_intuition,
        evidence_assessment=evidence_assessment,
        risk_flags=risk_flags,
        usage_suggestion=usage_suggestion,
        next_steps=next_steps,
    )


def parse_llm_explanation(raw: str) -> LLMExplanation | None:
    """Parse strict JSON returned by an LLM.

    Invalid or incomplete output returns ``None`` so report generation can
    degrade without interrupting the core research pipeline.
    """

    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return _from_dict(data)


def explanation_from_dict(data: dict[str, Any]) -> LLMExplanation | None:
    return _from_dict(data)
