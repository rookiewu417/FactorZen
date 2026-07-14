"""收尾复核与首轮护栏同口径：residual 候选按 residual 指标+floor 复核，防 objective 漂移误杀。

背景：`node_guardrails` 在 residual 模式下用 residual IC / DEFAULT_RESIDUAL_IC_FLOOR 判定，
但 `node_finalize_guardrails` 曾始终用 raw ic_train + 默认 floor 0.015 复核——
raw IC=0.005、residual IC=0.020 的候选首轮通过、收尾被「train_IC 太弱」误杀。
"""

from __future__ import annotations

from factorzen.agents.nodes import node_finalize_guardrails
from factorzen.agents.state import AgentState, AttemptRecord

_N_OBS = 303
_N_HOLDOUT = 80  # ≥ DEFAULT_HOLDOUT_MIN_DAYS(60)


def _attempt(expr: str, *, ic_train: float, ir_train: float = 0.3) -> AttemptRecord:
    return AttemptRecord(
        iteration=0, hypothesis="h", expression=expr, compile_ok=True,
        ic_train=ic_train, passed_guardrails=True, critic_verdict=None, error=None,
        ir_train=ir_train, turnover=0.3, n_train=_N_OBS,
        n_holdout_days=_N_HOLDOUT,
    )


def _cand_base(expr: str, *, ic_train: float, holdout_ic: float | None = None,
               ir_train: float = 0.3) -> dict:
    """library 门下 DSR 不参与入池；IR/CI 仅占位供 finalize 写回 dsr_pvalue。"""
    h = holdout_ic if holdout_ic is not None else (0.05 if ic_train >= 0 else -0.05)
    return {
        "expression": expr,
        "hypothesis": "h",
        "ic_train": ic_train,
        "ir_train": ir_train,
        "turnover": 0.3,
        "holdout_ic": h,
        "holdout_ir": 0.5 if h >= 0 else -0.5,
        "ic_ci_low": 0.01 if h >= 0 else -0.09,
        "ic_ci_high": 0.09 if h >= 0 else -0.01,
        "n_train": _N_OBS,
        "n_holdout_days": _N_HOLDOUT,
        "dsr": 0.99,
        "dsr_pvalue": 0.001,
    }


def _state(*, objective: str, candidates: list[dict],
           attempts: list[AttemptRecord] | None = None) -> AgentState:
    state = AgentState(seed=1, objective=objective)
    if attempts is not None:
        state.attempts.extend(attempts)
    else:
        for c in candidates:
            state.attempts.append(_attempt(c["expression"], ic_train=c["ic_train"],
                                           ir_train=c.get("ir_train", 0.3)))
    state.candidates.extend(candidates)
    return state


# ── 1. residual 候选：raw 弱但 residual 强 → 收尾保留 ──────────────────────


def test_residual_candidate_survives_when_raw_ic_below_floor():
    """raw IC 低于 0.015、residual IC 高于 residual floor → finalize 后仍保留。

    修复前会被「train_IC 太弱(|0.0050|<0.015)」误杀（TDD 反例）。
    """
    expr = "rank(neg(pb))"
    cand = _cand_base(expr, ic_train=0.005, holdout_ic=0.004)
    cand["residual_ic_train"] = 0.020
    cand["residual_holdout_ic"] = 0.018
    cand["n_residual_holdout_days"] = _N_HOLDOUT

    state = _state(objective="residual", candidates=[cand])
    node_finalize_guardrails(state)  # gate 默认 library

    assert len(state.candidates) == 1, (
        f"residual 强候选应保留，实得 survivors={state.candidates!r}"
    )
    assert state.candidates[0]["expression"] == expr
    a = next(x for x in state.attempts if x.expression == expr)
    assert a.passed_guardrails is True


# ── 2. residual 弱候选：死因文案是 residual 风格 ──────────────────────────


def test_residual_reject_reason_uses_residual_style():
    """residual_ic_train 低于 residual floor → 被删且文案含「残差」、不含 raw 弱 IC 文案。"""
    expr = "rank(ts_mean(volume, 5))"
    cand = _cand_base(expr, ic_train=0.020, holdout_ic=0.015)  # raw 本身够强
    cand["residual_ic_train"] = 0.001  # < DEFAULT_RESIDUAL_IC_FLOOR 0.010
    cand["residual_holdout_ic"] = 0.001
    cand["n_residual_holdout_days"] = _N_HOLDOUT

    state = _state(objective="residual", candidates=[cand])
    node_finalize_guardrails(state)

    assert state.candidates == [], "residual 弱候选应收尾剔除"
    a = next(x for x in state.attempts if x.expression == expr)
    assert a.passed_guardrails is False
    reason = a.reject_reason or ""
    assert "残差" in reason, f"应收尾 residual 文案，实得: {reason!r}"
    assert "train_IC 太弱" not in reason, f"不应出现 raw 弱 IC 文案: {reason!r}"


# ── 3. 库空退化：objective=residual 但候选无 residual 字段 → 回退 raw ────


def test_missing_residual_fields_falls_back_to_raw_gate():
    """objective 仍是 residual，但候选无 residual_*（库空退化入池）→ 按 raw 口径删。"""
    expr = "rank(neg(pe))"
    cand = _cand_base(expr, ic_train=0.005, holdout_ic=0.004)  # 无 residual 键

    state = _state(objective="residual", candidates=[cand])
    node_finalize_guardrails(state)

    assert state.candidates == [], "无 residual 字段时应回退 raw floor 并剔除"
    a = next(x for x in state.attempts if x.expression == expr)
    assert a.passed_guardrails is False
    reason = a.reject_reason or ""
    assert "train_IC 太弱" in reason, f"回退 raw 应出 train_IC 文案，实得: {reason!r}"


# ── 4. raw 模式零回归 ────────────────────────────────────────────────────


def test_raw_mode_strong_survives_weak_dropped():
    """objective=raw：强候选保留、弱候选删除，行为与修复前一致。"""
    strong = _cand_base("rank(neg(pb))", ic_train=0.030, holdout_ic=0.025)
    weak = _cand_base("rank(ts_std(close, 10))", ic_train=0.005, holdout_ic=0.004)

    state = _state(objective="raw", candidates=[strong, weak])
    node_finalize_guardrails(state)

    exprs = {c["expression"] for c in state.candidates}
    assert exprs == {"rank(neg(pb))"}, f"仅强候选应存活，实得 {exprs}"
    a_strong = next(x for x in state.attempts if x.expression == "rank(neg(pb))")
    a_weak = next(x for x in state.attempts if x.expression == "rank(ts_std(close, 10))")
    assert a_strong.passed_guardrails is True
    assert a_weak.passed_guardrails is False
    assert "train_IC 太弱" in (a_weak.reject_reason or "")
