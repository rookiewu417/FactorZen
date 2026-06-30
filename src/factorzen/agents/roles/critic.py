"""Critic（Risk Auditor）角色：读候选指标判过拟合，给否决回路 verdict。"""
from __future__ import annotations

from dataclasses import dataclass

from factorzen.llm.generation import LLMFn, _extract_json

_VALID_VERDICTS = {"keep", "revise_expr", "revise_hypothesis", "drop"}


@dataclass
class CriticVerdict:
    verdict: str
    reason: str


def critique(candidate: dict, llm_fn: LLMFn) -> CriticVerdict:
    """读候选 + 指标，判 keep/revise_expr/revise_hypothesis/drop。解析失败/非法 → keep（不误杀）。"""
    msgs = [
        {"role": "system", "content": (
            "你是量化风控审计员。读因子候选的指标（train IC / holdout IC / DSR），"
            "判断它是否过拟合、经济直觉是否成立。只输出 JSON: "
            '{"verdict": "keep"|"revise_expr"|"revise_hypothesis"|"drop", "reason": "..."}。'
            "keep=可入库；revise_expr=方向对但表达式需改；revise_hypothesis=方向需换；drop=丢弃。")},
        {"role": "user", "content": (
            f"表达式: {candidate.get('expression')}\n假设: {candidate.get('hypothesis')}\n"
            f"train_IC: {candidate.get('ic_train')}\nholdout_IC: {candidate.get('holdout_ic')}\n"
            f"DSR: {candidate.get('dsr')} (p={candidate.get('dsr_pvalue')})")},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or obj.get("verdict") not in _VALID_VERDICTS:
        return CriticVerdict("keep", str(obj.get("reason", "")) if obj else "")
    return CriticVerdict(str(obj["verdict"]), str(obj.get("reason", "")))
