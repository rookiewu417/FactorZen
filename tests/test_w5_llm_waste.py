# tests/test_w5_llm_waste.py
"""W5：未知算子不进 heal + 窗口钳制 + 空轮跳 critic。"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from factorzen.agents.self_heal import heal_expressions
from factorzen.discovery.expression import clamp_window_literals, parse_expr

# ── W5a ──────────────────────────────────────────────────────────────────────

def test_heal_drops_unknown_op_no_llm():
    """ts_delta 等未知算子不触发 revise_from_error，计数=1。"""
    calls = {"n": 0}

    def fake(_msgs):
        calls["n"] += 1
        raise AssertionError("未知算子不应触发 LLM")

    stats: dict = {}
    healed = heal_expressions(
        ["ts_delta(close, 5)"], "动量", fake, max_rounds=2, stats=stats,
    )
    assert healed == []
    assert calls["n"] == 0
    assert stats.get("n_unknown_op_dropped") == 1


def test_heal_syntax_error_still_enters_heal():
    """普通语法错（非未知算子）仍进 heal 调 LLM。"""
    calls = {"n": 0}

    def fake(_msgs):
        calls["n"] += 1
        return json.dumps({"expressions": ["ts_mean(close, 5)"]})

    healed = heal_expressions(["ts_mean()"], "动量", fake, max_rounds=2)
    assert calls["n"] >= 1
    assert any("ts_mean" in h for h in healed)
    for h in healed:
        parse_expr(h)


def test_heal_drop_unknown_ops_false_restores_old():
    """drop_unknown_ops=False 时未知算子仍进 heal（兼容开关）。"""
    calls = {"n": 0}

    def fake(_msgs):
        calls["n"] += 1
        return json.dumps({"expressions": ["ts_mean(close, 5)"]})

    healed = heal_expressions(
        ["ts_delta(close, 5)"], "h", fake, max_rounds=2, drop_unknown_ops=False,
    )
    assert calls["n"] >= 1
    assert healed


# ── W5b ──────────────────────────────────────────────────────────────────────

def test_clamp_window_over_budget():
    """504 窗 + 预算 400 → 钳到 400。"""
    out, did = clamp_window_literals(
        "ts_mean(amount, 504)", {"amount": 400}, None,
    )
    assert did is True
    assert "400" in out
    assert "504" not in out
    node = parse_expr(out)
    assert getattr(node, "window", None) == 400


def test_clamp_window_budget_sufficient_unchanged():
    """预算充足 → 不动。"""
    expr = "ts_mean(amount, 20)"
    out, did = clamp_window_literals(expr, {"amount": 400}, None)
    assert did is False
    assert out == expr


def test_clamp_window_no_window_literal():
    """无窗口字面量（纯截面）→ 原样。"""
    expr = "rank(amount)"
    out, did = clamp_window_literals(expr, {"amount": 400}, None)
    assert did is False
    assert out == expr


def test_clamp_window_parse_fail_passthrough():
    expr = "not_parseable!!!"
    out, did = clamp_window_literals(expr, {"amount": 400}, None)
    assert out == expr and did is False


def test_clamp_nested_windows():
    """嵌套：仅超 cap 的窗口被钳。"""
    out, did = clamp_window_literals(
        "ts_mean(delta(amount, 20), 504)", {"amount": 400}, None,
    )
    assert did is True
    node = parse_expr(out)
    assert node.window == 400
    assert node.children[0].window == 20  # type: ignore[attr-defined]


# ── W5c ──────────────────────────────────────────────────────────────────────

def _mock_daily(n_stocks=30, n_days=120, seed=3):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def test_empty_round_skips_critic_llm(tmp_path: Path):
    """new_cands=[] 时 critic 零调用，verdict=revise_hypothesis，critic_skipped=True。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    critic_calls = {"n": 0}

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            critic_calls["n"] += 1
            return json.dumps({"verdict": "keep", "reason": "should_not_run"})
        if "翻译成" in text:
            # 故意产重复/已评估表达式 → 评估后可能无新候选进护栏
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    # 让护栏拒绝所有候选 → new_cands 空
    def _guard_reject(state, **kw):
        return state  # 不注入 candidates

    daily = _mock_daily()
    with patch("factorzen.agents.team_orchestrator.node_guardrails", side_effect=_guard_reject):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"),
        )
    assert critic_calls["n"] == 0, f"空轮不应调 critic，实得 {critic_calls['n']}"
    assert res.rounds_log
    r0 = res.rounds_log[0]
    assert r0.get("critic_skipped") is True
    assert r0["verdict"] == "revise_hypothesis"
    assert "无新候选" in r0["reason"]


def test_nonempty_round_still_calls_critic(tmp_path: Path):
    """有新候选时 critic 仍被调用。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    critic_calls = {"n": 0}

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            critic_calls["n"] += 1
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in text:
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    def _inject(state, *, ledger, **_kw):
        for a in state.attempts:
            if a.iteration != state.iteration:
                continue
            a.passed_guardrails = True
            state.candidates.append({
                "expression": a.expression, "hypothesis": a.hypothesis,
                "ic_train": 0.05, "ir_train": 0.4, "turnover": 0.1,
                "holdout_ic": 0.04, "holdout_ir": 0.3,
                "dsr": 0.7, "dsr_pvalue": 0.05, "n_train": 100,
                "n_holdout_days": 80, "ic_ci_low": 0.01, "ic_ci_high": 0.08,
            })
            ledger.record(1)
        return state

    daily = _mock_daily()
    with patch("factorzen.agents.team_orchestrator.node_guardrails", side_effect=_inject):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"),
        )
    assert critic_calls["n"] >= 1
    assert res.rounds_log[0].get("critic_skipped") is False
