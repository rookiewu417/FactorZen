"""Hypothesis 角色：提经济直觉方向，注入长期记忆（避开已知无效，借鉴已知有效）。"""
from __future__ import annotations

from factorzen.llm.generation import LLMFn, _extract_json


def propose_hypotheses(
    llm_fn: LLMFn,
    *,
    known_invalid: list[str],
    known_valid: list[str],
    feedback: str = "",
    n: int = 1,
) -> list[str]:
    """提 n 个经济直觉方向（自然语言）。解析失败 → 空列表。"""
    sys = (
        "你是量化研究员，提出有经济直觉的选股方向（自然语言，不写公式）。"
        '只输出 JSON: {"hypotheses": ["方向1", "方向2"]}。'
    )
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS
    sys = sys + "\n" + ASHARE_CAVEATS
    user = f"提出 {n} 个新方向。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if known_invalid:
        user += "\n以下表达式已验证无效，避开这些思路:\n" + "\n".join(
            f"- {e}" for e in known_invalid
        )
    if known_valid:
        user += "\n以下表达式已验证有效，可借鉴其思路方向（但不要照抄）:\n" + "\n".join(
            f"- {e}" for e in known_valid
        )
    obj = _extract_json(
        llm_fn([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    )
    if not obj:
        return []
    hyps = obj.get("hypotheses")
    return [str(h) for h in hyps] if isinstance(hyps, list) else []
