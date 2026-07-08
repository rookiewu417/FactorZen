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


def propose_structured(
    llm_fn: LLMFn,
    *,
    known_invalid: list[str],
    known_valid: list[str],
    feedback: str = "",
    n: int = 1,
) -> list[dict]:
    """结构化假设（RD-Agent 步1）：每个含 direction/mechanism/expected_sign/falsification。"""
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS
    sys = (
        "你是量化研究员，提出结构化选股假设。每个假设含四要素："
        "direction(方向,自然语言)、mechanism(经济机制)、expected_sign(预期IC符号,+1或-1)、"
        "falsification(可证伪判据)。只输出 JSON: "
        '{"hypotheses":[{"direction":"...","mechanism":"...","expected_sign":1,"falsification":"..."}]}。'
    )
    sys = sys + "\n" + ASHARE_CAVEATS
    user = f"提出 {n} 个结构化假设。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if known_invalid:
        user += "\n避开已验证无效:\n" + "\n".join(f"- {e}" for e in known_invalid)
    if known_valid:
        user += "\n可借鉴已验证有效:\n" + "\n".join(f"- {e}" for e in known_valid)
    obj = _extract_json(
        llm_fn([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    )
    if not obj:
        return []
    hyps = obj.get("hypotheses")
    if not isinstance(hyps, list):
        return []
    out: list[dict] = []
    for h in hyps:
        if isinstance(h, dict) and h.get("direction"):
            out.append({
                "direction": str(h.get("direction", "")),
                "mechanism": str(h.get("mechanism", "")),
                "expected_sign": h.get("expected_sign"),
                "falsification": str(h.get("falsification", "")),
            })
    return out


def format_structured(h: dict) -> str:
    """把结构化假设渲染成供 Coder 翻译的自然语言方向文本。"""
    parts = [h.get("direction", "")]
    if h.get("mechanism"):
        parts.append(f"机制: {h['mechanism']}")
    if h.get("expected_sign") is not None:
        parts.append(f"预期IC符号: {h['expected_sign']}")
    if h.get("falsification"):
        parts.append(f"证伪判据: {h['falsification']}")
    return "；".join(p for p in parts if p)
