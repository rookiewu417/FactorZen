"""LLM 因子生成层：假设 + 表达式提议 + 语义对齐自检 + prompt 模板。"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

LLMFn = Callable[[list[dict[str, str]]], str]


@dataclass
class FactorProposal:
    hypothesis: str
    expressions: list[str]
    rationale: str


def _extract_json(raw: str) -> dict | None:
    """容错解析：直接 json.loads；失败找首个 {...} 子串；再失败返回 None。"""
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def generate_factor_proposal(
    messages: list[dict[str, str]],
    llm_fn: LLMFn,
    *,
    n_hypotheses: int = 1,
) -> list[FactorProposal]:
    """调用 LLM 生成 1+ 个 (假设, 表达式集)。解析失败的丢弃（降级不抛）。"""
    proposals: list[FactorProposal] = []
    for _ in range(max(1, n_hypotheses)):
        obj = _extract_json(llm_fn(messages))
        if not obj:
            continue
        exprs = obj.get("expressions")
        if not isinstance(exprs, list) or not exprs:
            continue
        proposals.append(
            FactorProposal(
                hypothesis=str(obj.get("hypothesis", "")),
                expressions=[str(e) for e in exprs],
                rationale=str(obj.get("rationale", "")),
            )
        )
    return proposals


def semantic_check(
    hypothesis: str, expression: str, llm_fn: LLMFn
) -> tuple[bool, str]:
    """LLM 自查表达式是否实现假设。返回 (一致?, 理由)。解析失败 → (True, '') 放行（避免误杀）。"""
    msgs = [
        {
            "role": "system",
            "content": (
                "你判断量化因子表达式是否实现了给定假设，只输出 JSON: "
                '{"consistent": true/false, "reason": "..."}'
            ),
        },
        {"role": "user", "content": f"假设: {hypothesis}\n表达式: {expression}"},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or "consistent" not in obj:
        return True, ""  # 解析失败放行，不误杀
    return bool(obj["consistent"]), str(obj.get("reason", ""))


def build_agent_messages(
    op_names: list[str],
    leaf_names: list[str],
    feedback: str = "",
    negatives: list[str] | None = None,
) -> list[dict[str, str]]:
    """构造生成 prompt：算子/特征清单 + 上轮反馈 + Negative RAG 负例。"""
    neg = negatives or []
    system = (
        "你是量化研究员，提出有经济直觉的假设并翻译成因子表达式。\n"
        f"可用算子: {', '.join(op_names)}\n"
        f"可用特征(叶子): {', '.join(leaf_names)}\n"
        "时序算子最后一个参数是整型窗口，如 ts_mean(close, 20)。\n"
        '只输出 JSON: {"hypothesis": "...", "expressions": ["...", "..."], "rationale": "..."}'
    )
    user = "提出一个新假设并给出 2-4 个候选表达式。"
    if feedback:
        user += f"\n上一轮反馈: {feedback}"
    if neg:
        user += "\n避免以下已探索过/低效的模式:\n" + "\n".join(f"- {n}" for n in neg)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
