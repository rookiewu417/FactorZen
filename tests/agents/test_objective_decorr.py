"""合并自 agents 相关碎片测试（test_objective_decorr.py）。

test_finalize_objective_parity.py：收尾复核与首轮护栏同口径：residual 候选按 residual 指标+floor 复核
test_decorr_boundary.py：M1/Agent 去相关边界语义（阈值 < vs >=）与 library orthogonal
test_agent_multiobjective.py：Workstream A 多目标评估：换手率 + evaluate_expressions 多维契约
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_finalize_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.evaluation import _factor_turnover, evaluate_expressions
from factorzen.discovery.scoring import DEFAULT_DECORR_THRESHOLD, DataBundle

# ==== 来自 test_finalize_objective_parity.py ====
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

# ==== 来自 test_decorr_boundary.py ====
_SRC = Path(__file__).resolve().parents[2] / "src" / "factorzen"


def _mk_daily(n_days=80, n_stocks=30, seed=3):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for c in codes:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7,
            })
    return pl.DataFrame(rows)


# ── 三处语义契约（参数化）──────────────────────────────────────────────────




@pytest.mark.parametrize(
    "mc,expect_ok",
    [
        (DEFAULT_DECORR_THRESHOLD, False),
        (0.699, True),
        (0.701, False),
    ],
)
def test_library_orthogonal_check_boundary(mc, expect_ok, monkeypatch):
    """library_orthogonal_check：``mc >= threshold`` → ok=False。"""
    from factorzen.discovery import scoring as scoring_mod

    monkeypatch.setattr(
        scoring_mod, "max_correlation_detail",
        lambda _f, _p, panel=None: (mc, "pool_expr"),
    )
    ok, got, nearest = scoring_mod.library_orthogonal_check(
        pl.DataFrame({"trade_date": ["d"], "ts_code": ["s"], "factor_value": [1.0]}),
        {"pool_expr": pl.DataFrame(
            {"trade_date": ["d"], "ts_code": ["s"], "factor_value": [1.0]},
        )},
        threshold=DEFAULT_DECORR_THRESHOLD,
    )
    assert ok is expect_ok
    assert got == mc
    assert nearest == "pool_expr"


# ── Agent 路径 runtime：node_guardrails 会话池去相关 ────────────────────────


def _seed_attempt(state, expr: str, *, ic: float = 0.05, ir: float = 1.2, n: int = 100):
    from factorzen.agents.state import AttemptRecord

    state.attempts.append(AttemptRecord(
        iteration=state.iteration, hypothesis="h", expression=expr,
        compile_ok=True, ic_train=ic, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=ir, turnover=0.3, n_train=n,
    ))


@pytest.mark.parametrize(
    "corr_value,expect_decorrelated",
    [
        (DEFAULT_DECORR_THRESHOLD, True),   # 恰 0.7 → 拒
        (0.699, False),                     # 略低 → 放行
    ],
)
def test_node_guardrails_session_decorr_boundary(
    corr_value, expect_decorrelated, monkeypatch,
):
    """Agent node_guardrails：会话池 max_correlation 恰阈值时与 M1 同拒。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily()
    bundle = DataBundle.build(daily)

    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda fdf, hdf: HoldoutICResult(0.05, 0.5, (0.01, 0.09), n_days=100),
    )
    # 护栏定量门恒过
    import factorzen.discovery.guardrails as gmod
    monkeypatch.setattr(gmod, "acceptance_reasons", lambda **_kw: [])

    # 第一个候选入池时 pool 空 → max_corr=0；第二个起返回受控 corr
    def _fake_max_corr(fdf, pool, panel=None):
        if not pool:
            return 0.0
        return float(corr_value)

    monkeypatch.setattr(
        "factorzen.discovery.scoring.max_correlation", _fake_max_corr,
    )

    state = AgentState(seed=1)
    _seed_attempt(state, "rank(close)", ic=0.06)
    _seed_attempt(state, "rank(vol)", ic=0.05)

    node_guardrails(
        state, daily=daily, holdout_df=daily, bundle=bundle,
        ledger=TrialLedger(), top_k=5, lib_pool=None,
    )

    first = next(a for a in state.attempts if a.expression == "rank(close)")
    second = next(a for a in state.attempts if a.expression == "rank(vol)")
    assert first.expression in {c["expression"] for c in state.candidates}
    if expect_decorrelated:
        assert second.decorrelated is True
        assert second.expression not in {c["expression"] for c in state.candidates}
        assert second.reject_reason and "高度相关" in second.reject_reason
        # 文案用 ≥ 而非 >
        assert "≥" in second.reject_reason or ">=" in second.reject_reason
    else:
        assert second.decorrelated is False
        assert second.expression in {c["expression"] for c in state.candidates}


# ── M1 源码 + runtime 边界（贪心入选）──────────────────────────────────────


def test_m1_source_uses_strict_lt_threshold():
    """M1 mining_session 必须用 ``mc < decorr_threshold``（恰等拒）。"""
    text = (_SRC / "discovery" / "mining_session.py").read_text(encoding="utf-8")
    assert "mc < decorr_threshold" in text


def test_agent_source_uses_ge_default_decorr_threshold():
    """Agent 必须用 ``corr >= DEFAULT_DECORR_THRESHOLD``，禁止硬编码 ``corr > 0.7``。"""
    text = (_SRC / "agents" / "nodes.py").read_text(encoding="utf-8")
    assert "corr >= DEFAULT_DECORR_THRESHOLD" in text
    # 会话池去相关处不再出现开区间硬编码
    assert "corr > 0.7" not in text


def test_m1_greedy_boundary_via_max_correlation_mock(tmp_path, monkeypatch):
    """M1 top-K 路径：mock max_correlation 恰阈值时第二因子不入选。"""
    from factorzen.discovery.mining_session import run_session

    daily = _mk_daily(n_days=60, n_stocks=35)
    exprs = ["rank(close)", "rank(vol)"]
    idx = {"i": 0}

    class _FakeSearcher:
        def __init__(self, *a, **k):
            pass

        def propose(self):
            from factorzen.discovery.expression import parse_expr
            e = exprs[idx["i"] % len(exprs)]
            idx["i"] += 1
            return parse_expr(e)

    monkeypatch.setattr(
        "factorzen.discovery.mining_session.RandomSearcher", _FakeSearcher,
    )

    def _fake_max_corr(fdf, pool, panel=None):
        if not pool:
            return 0.0
        return float(DEFAULT_DECORR_THRESHOLD)  # 恰阈值 → 应拒

    # mining_session 顶层 from-import，须 patch 模块内绑定名
    monkeypatch.setattr(
        "factorzen.discovery.mining_session.max_correlation", _fake_max_corr,
    )
    res = run_session(
        daily, n_trials=4, top_k=3, seed=1, method="random",
        out_dir=str(tmp_path / "sessions"),
        update_library=False,
        library_orthogonal=False,
        library_root=str(tmp_path / "empty_lib"),
    )
    cands = res["candidates"]
    # 恰阈值第二因子不得以 max_corr=0.7 入选
    for c in cands:
        if c.get("expression") == "rank(vol)":
            mc = c.get("max_corr")
            if mc is not None:
                assert float(mc) < DEFAULT_DECORR_THRESHOLD

# ==== 来自 test_agent_multiobjective.py ====
def _mock_daily(n_stocks=40, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px,
                         "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _factor_df(values: dict) -> pl.DataFrame:
    """{(date,code): value} → [trade_date, ts_code, factor_value]。"""
    rows = [{"trade_date": d, "ts_code": c, "factor_value": v} for (d, c), v in values.items()]
    return pl.DataFrame(rows)


def test_turnover_constant_ranking_is_zero():
    """每天排序完全一致（因子值=股票固定特征）→ top-k 持仓不变 → 换手率 ≈ 0。"""
    days = [dt.date(2022, 1, 3) + dt.timedelta(days=i) for i in range(10)]
    codes = [f"{i:06d}.SZ" for i in range(40)]
    values = {(d, c): float(idx) for d in days for idx, c in enumerate(codes)}
    to = _factor_turnover(_factor_df(values), quantile=0.2)
    assert to is not None
    assert to < 1e-9, f"常数排序换手率应为 0，实际 {to}"


def test_turnover_random_reshuffle_is_high():
    """每天完全随机重排 → top-k 频繁换血 → 换手率显著 > 0。"""
    rng = np.random.default_rng(7)
    days = [dt.date(2022, 1, 3) + dt.timedelta(days=i) for i in range(30)]
    codes = [f"{i:06d}.SZ" for i in range(40)]
    values = {(d, c): float(rng.standard_normal()) for d in days for c in codes}
    to = _factor_turnover(_factor_df(values), quantile=0.2)
    assert to is not None
    assert to > 0.5, f"随机重排换手率应显著>0，实际 {to}"


def test_turnover_single_day_is_none():
    """单个交易日无法算相邻变化 → None。"""
    day = dt.date(2022, 1, 3)
    codes = [f"{i:06d}.SZ" for i in range(40)]
    values = {(day, c): float(i) for i, c in enumerate(codes)}
    assert _factor_turnover(_factor_df(values), quantile=0.2) is None


def test_turnover_empty_is_none():
    empty = pl.DataFrame({"trade_date": [], "ts_code": [], "factor_value": []})
    assert _factor_turnover(empty, quantile=0.2) is None


def test_evaluate_expressions_has_turnover_field():
    """多目标契约：合法/非法结果都含 turnover 键（不破坏现有 4 字段契约）。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["ts_mean(close,5)", "not_a_func("], daily, bundle)
    assert len(out) == 2
    for r in out:
        assert "turnover" in r, "结果必须含 turnover 字段"
    ok = next(r for r in out if r["compile_ok"])
    # 旧断言 `is None or isinstance(float)` 恒真（None 与任意 float 全覆盖）。
    # turnover 是单边换手率，语义上必落在 [0, 1]：0=从不换仓，1=每日全部换掉。
    assert ok["turnover"] is not None, "可评估的表达式应算得出换手率"
    assert 0.0 <= ok["turnover"] <= 1.0, f"单边换手率必须 ∈ [0,1]，实得 {ok['turnover']}"
    bad = next(r for r in out if not r["compile_ok"])
    assert bad["turnover"] is None


def test_evaluate_expressions_icir_is_ir():
    """ICIR 即 ir_train（IC_mean/IC_std），多目标评估保留并暴露。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["ts_mean(close,5)"], daily, bundle)
    assert out[0]["ir_train"] is not None
    assert isinstance(out[0]["ir_train"], float)



def test_node_evaluate_records_turnover():
    """M5 node_evaluate 把 evaluate 的 turnover 写进 AttemptRecord。"""
    from factorzen.agents.nodes import _PendingExpr, node_evaluate
    from factorzen.agents.state import AgentState
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    state = AgentState(seed=0)
    state._pending = [_PendingExpr("动量", "ts_mean(close, 5)")]  # type: ignore[attr-defined]
    node_evaluate(state, daily=daily, bundle=bundle)
    assert len(state.attempts) == 1
    a = state.attempts[0]
    # 同上：恒真断言换成语义断言。ts_mean(close,5) 是平滑价格，换手率应显著低于「每日重排」。
    assert a.turnover is not None
    assert 0.0 <= a.turnover <= 1.0, f"单边换手率必须 ∈ [0,1]，实得 {a.turnover}"



def test_critique_prompt_includes_cost_metrics():
    """Critic prompt 必须注入 ICIR + 换手率(成本代理)，引导「IC 高≠可实现超额」判断。"""
    from factorzen.agents.roles.critic import critique
    captured: dict = {}

    def fake_llm(messages):
        captured["msgs"] = messages
        return json.dumps({"verdict": "keep", "reason": "ok"})
    cand = {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
            "holdout_ic": 0.03, "dsr": 0.7, "dsr_pvalue": 0.01,
            "ir_train": 0.55, "turnover": 0.83}
    critique(cand, fake_llm)
    alltext = " ".join(m["content"] for m in captured["msgs"])
    assert "换手" in alltext, "Critic prompt 应含换手率"
    assert "0.83" in alltext, "Critic prompt 应展示 turnover 数值"
    assert "ICIR" in alltext or "0.55" in alltext, "Critic prompt 应含 ICIR"


def test_node_critic_prompt_includes_cost_metrics():
    """M5 node_critic prompt 同样注入多维指标（与 M6 Critic 口径一致）。"""
    from factorzen.agents.nodes import node_critic
    from factorzen.agents.state import AgentState, AttemptRecord
    captured: dict = {}

    def fake_llm(messages):
        captured["msgs"] = messages
        return '{"verdict":"keep","reason":"ok"}'
    state = AgentState(seed=0)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="动量", expression="ts_mean(close,5)", compile_ok=True,
        ic_train=0.05, passed_guardrails=True, critic_verdict=None, error=None,
        ir_train=0.55, turnover=0.83))
    node_critic(state, fake_llm)
    alltext = " ".join(m["content"] for m in captured["msgs"])
    assert "换手" in alltext and "0.83" in alltext
