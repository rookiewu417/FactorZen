# tests/test_agent_multiobjective.py
"""Workstream A 多目标评估：换手率 + evaluate_expressions 多维契约 + 全链路写入/prompt 注入。

换手率语义（独立 ground-truth 可验，非恒真）：纯多头 top-quantile 组合的单边换手率
= 0.5·Σ|w_t − w_{t-1}| ∈ [0,1]。常数排序因子 → 0（持仓不变）；每日随机重排 → 高。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from factorzen.agents.evaluation import _factor_turnover, evaluate_expressions
from factorzen.discovery.scoring import DataBundle


def _mock_daily(n_stocks=20, n_days=120, seed=1):
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
    codes = [f"{i:06d}.SZ" for i in range(20)]
    values = {(d, c): float(idx) for d in days for idx, c in enumerate(codes)}
    to = _factor_turnover(_factor_df(values), quantile=0.2)
    assert to is not None
    assert to < 1e-9, f"常数排序换手率应为 0，实际 {to}"


def test_turnover_random_reshuffle_is_high():
    """每天完全随机重排 → top-k 频繁换血 → 换手率显著 > 0。"""
    rng = np.random.default_rng(7)
    days = [dt.date(2022, 1, 3) + dt.timedelta(days=i) for i in range(30)]
    codes = [f"{i:06d}.SZ" for i in range(20)]
    values = {(d, c): float(rng.standard_normal()) for d in days for c in codes}
    to = _factor_turnover(_factor_df(values), quantile=0.2)
    assert to is not None
    assert to > 0.5, f"随机重排换手率应显著>0，实际 {to}"


def test_turnover_single_day_is_none():
    """单个交易日无法算相邻变化 → None。"""
    day = dt.date(2022, 1, 3)
    codes = [f"{i:06d}.SZ" for i in range(20)]
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
    assert ok["turnover"] is None or isinstance(ok["turnover"], float)
    bad = next(r for r in out if not r["compile_ok"])
    assert bad["turnover"] is None


def test_evaluate_expressions_icir_is_ir():
    """ICIR 即 ir_train（IC_mean/IC_std），多目标评估保留并暴露。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["ts_mean(close,5)"], daily, bundle)
    assert out[0]["ir_train"] is not None
    assert isinstance(out[0]["ir_train"], float)


def test_attempt_record_turnover_field_defaults_none():
    """AttemptRecord 增 turnover 字段，向后兼容（默认 None，不破坏旧构造）。"""
    from factorzen.agents.state import AttemptRecord
    r = AttemptRecord(iteration=0, hypothesis="h", expression="e", compile_ok=True,
                      ic_train=0.05, passed_guardrails=False, critic_verdict=None, error=None)
    assert hasattr(r, "turnover")
    assert r.turnover is None
    assert "turnover" in r.__dict__


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
    assert hasattr(a, "turnover")
    assert a.turnover is None or isinstance(a.turnover, float)


def test_team_records_turnover(tmp_path):
    """M6 team _evaluate_and_record 同样写 turnover（双路径一致）。"""
    from factorzen.agents.team_orchestrator import run_team_agent
    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})] * 10
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    daily = _mock_daily(n_days=180)
    res = run_team_agent(daily, fn, n_rounds=1, seed=1, index_path=str(tmp_path / "e.jsonl"))
    assert res.state.attempts
    for a in res.state.attempts:
        assert hasattr(a, "turnover")


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
