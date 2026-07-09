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
    """读候选多维指标，判 keep/revise_expr/revise_hypothesis/drop。解析失败/非法 → keep（不误杀）。"""
    icir = candidate.get("holdout_ir")
    if icir is None:
        icir = candidate.get("ir_train")
    msgs = [
        {"role": "system", "content": (
            "你是量化风控审计员。读因子候选的多维指标（train IC / holdout IC / DSR / "
            "ICIR / 换手率），判断它是否过拟合、经济直觉是否成立、是否可实现超额收益。"
            "注意：换手率高意味着交易成本侵蚀，IC 高但换手率高的因子未必可实现超额（成本双杀）；"
            "ICIR（信息比率）越高越稳定。只输出 JSON: "
            '{"verdict": "keep"|"revise_expr"|"revise_hypothesis"|"drop", "reason": "..."}。'
            "keep=可入库；revise_expr=方向对但表达式需改；revise_hypothesis=方向需换；drop=丢弃。")},
        {"role": "user", "content": (
            f"表达式: {candidate.get('expression')}\n假设: {candidate.get('hypothesis')}\n"
            f"train_IC: {candidate.get('ic_train')}\nholdout_IC: {candidate.get('holdout_ic')}\n"
            f"ICIR: {icir}\n换手率(单边,成本代理): {candidate.get('turnover')}\n"
            f"DSR: {candidate.get('dsr')} (p={candidate.get('dsr_pvalue')})")},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or obj.get("verdict") not in _VALID_VERDICTS:
        return CriticVerdict("keep", str(obj.get("reason", "")) if obj else "")
    return CriticVerdict(str(obj["verdict"]), str(obj.get("reason", "")))
