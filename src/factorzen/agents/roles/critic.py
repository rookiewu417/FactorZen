"""Critic（Risk Auditor）角色：读候选指标判过拟合，给否决回路 verdict。"""
from __future__ import annotations

from dataclasses import dataclass

from factorzen.llm.generation import LLMFn, _extract_json

_VALID_VERDICTS = {"keep", "revise_expr", "revise_hypothesis", "drop"}


@dataclass
class CriticVerdict:
    verdict: str
    reason: str


def critique(
    candidate: dict,
    llm_fn: LLMFn,
    *,
    lift_rejected: list[dict] | None = None,
) -> CriticVerdict:
    """读候选多维指标，判 keep/revise_expr/revise_hypothesis/drop。解析失败/非法 → keep（不误杀）。

    ``lift_rejected``：组合层已证无增量的方向列表；None → 不注入（零回归）。
    正交字段（``residual_ic_train`` / ``residual_holdout_ic`` / ``max_corr_library``）
    仅当 candidate 中存在时注入 user_content（None-gating）。
    """
    icir = candidate.get("holdout_ir")
    if icir is None:
        icir = candidate.get("ir_train")
    user_content = (
        f"表达式: {candidate.get('expression')}\n假设: {candidate.get('hypothesis')}\n"
        f"train_IC: {candidate.get('ic_train')}\nholdout_IC: {candidate.get('holdout_ic')}\n"
        f"n_holdout_days(holdout 有效天数): {candidate.get('n_holdout_days')}\n"
        f"ICIR: {icir}\n换手率(单边,成本代理): {candidate.get('turnover')}\n"
        f"DSR: {candidate.get('dsr')} (p={candidate.get('dsr_pvalue')})\n"
        "提示: n_holdout_days 过低表示 holdout 缺数据（非方向错误）；"
        "勿把覆盖不足误判为经济直觉失败。"
    )
    # W6 正交审计：残差 IC / 库相关（有则注入）
    ric_tr = candidate.get("residual_ic_train")
    ric_ho = candidate.get("residual_holdout_ic")
    if ric_tr is not None or ric_ho is not None:
        user_content += (
            f"\n对库残差IC(train/holdout): {ric_tr}/{ric_ho}"
        )
    mc_lib = candidate.get("max_corr_library")
    if mc_lib is not None:
        user_content += f"\n与库最大相关: {mc_lib}"
    if lift_rejected:
        from factorzen.llm.prompt_fragments import format_lift_rejected
        frag = format_lift_rejected(lift_rejected)
        if frag:
            user_content = user_content + "\n" + frag
    msgs = [
        {"role": "system", "content": (
            "你是量化风控审计员。读因子候选的多维指标（train IC / holdout IC / DSR / "
            "ICIR / 换手率），判断它是否过拟合、经济直觉是否成立、是否可实现超额收益。"
            "注意：换手率高意味着交易成本侵蚀，IC 高但换手率高的因子未必可实现超额（成本双杀）；"
            "ICIR（信息比率）越高越稳定。"
            "残差 IC 才是相对因子库的增量信号；|残差| 低或与库相关高 → 倾向 revise_hypothesis"
            "（换正交方向）而非 keep。"
            "只输出 JSON: "
            '{"verdict": "keep"|"revise_expr"|"revise_hypothesis"|"drop", "reason": "..."}。'
            "keep=可入库；revise_expr=方向对但表达式需改；revise_hypothesis=方向需换；drop=丢弃。")},
        {"role": "user", "content": user_content},
    ]
    obj = _extract_json(llm_fn(msgs))
    if not obj or obj.get("verdict") not in _VALID_VERDICTS:
        return CriticVerdict("keep", str(obj.get("reason", "")) if obj else "")
    return CriticVerdict(str(obj["verdict"]), str(obj.get("reason", "")))
