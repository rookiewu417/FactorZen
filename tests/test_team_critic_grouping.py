# tests/test_team_critic_grouping.py
"""Critic 按 hypothesis 分组裁决：多假设轮不得连坐/交叉污染。"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent


def _mock_daily(n_stocks=40, n_days=180, seed=1):
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
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _inject_guardrails_from_attempts(state, *, ledger, **_kwargs):
    """把本轮 attempts 全量注入 candidates（含 hypothesis），模拟护栏通过。

    字段对齐 nodes.py cand_row；ic/holdout 强制同号正值，保证 library 收尾复核不误杀。
    """
    n = 0
    for a in state.attempts:
        if a.iteration != state.iteration:
            continue
        a.passed_guardrails = True
        state.candidates.append({
            "expression": a.expression,
            "hypothesis": a.hypothesis,
            "ic_train": 0.05,
            "ir_train": 0.4,
            "turnover": 0.1,
            "holdout_ic": 0.04,
            "holdout_ir": 0.3,
            "dsr": 0.7,
            "dsr_pvalue": 0.05,
            "n_train": a.n_train if a.n_train is not None else 100,
            "n_holdout_days": 80,  # ≥ DEFAULT_HOLDOUT_MIN_DAYS，收尾 library 覆盖门
            "ic_ci_low": 0.01,
            "ic_ci_high": 0.08,
        })
        n += 1
    if n:
        ledger.record(n)
    return state


def test_critic_groups_by_hypothesis_no_cross_kill(tmp_path: Path):
    """两假设各 1 候选：H1 drop / H2 keep → 只杀 H1，verdict 不交叉污染。"""
    h1, h2 = "HYPG1", "HYPG2"
    expr1, expr2 = "ts_mean(close,5)", "ts_std(close,10)"
    # 评估规范化后带空格
    norm1, norm2 = "ts_mean(close, 5)", "ts_std(close, 10)"

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            # critique user 内容含「假设: ...」；按代表候选的 hypothesis 分流
            if f"假设: {h1}" in text:
                return json.dumps({"verdict": "drop", "reason": "H1 过拟合"})
            if f"假设: {h2}" in text:
                return json.dumps({"verdict": "keep", "reason": "H2 稳健"})
            return json.dumps({"verdict": "keep", "reason": "fallback"})
        if "翻译成" in text:
            if h1 in text:
                return json.dumps({"expressions": [expr1]})
            if h2 in text:
                return json.dumps({"expressions": [expr2]})
            return json.dumps({"expressions": ["rank(vol)"]})
        return json.dumps({"hypotheses": [h1, h2]})

    daily = _mock_daily()
    with patch(
        "factorzen.agents.team_orchestrator.node_guardrails",
        _inject_guardrails_from_attempts,
    ):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"), hypotheses_per_round=2,
        )

    cand_exprs = {c["expression"] for c in res.candidates}
    assert norm2 in cand_exprs, f"H2 keep 候选应保留: {cand_exprs}"
    assert norm1 not in cand_exprs, f"H1 drop 候选应移除: {cand_exprs}"

    by_expr = {a.expression: a for a in res.state.attempts}
    assert by_expr[norm1].critic_verdict == "drop"
    assert by_expr[norm2].critic_verdict == "keep"
    # 事实字段不许被 verdict 改写
    assert by_expr[norm1].passed_guardrails is True
    assert by_expr[norm2].passed_guardrails is True

    last = res.rounds_log[-1]
    assert "verdicts" in last and len(last["verdicts"]) == 2
    by_h = {v["hypothesis"]: v for v in last["verdicts"]}
    assert by_h[h1]["verdict"] == "drop"
    assert by_h[h2]["verdict"] == "keep"
    # 原键 = 最后一组（H2 keep）零回归语义
    assert last["verdict"] == "keep"
    assert last["reason"] == "H2 稳健"


def test_critic_single_hypothesis_drop_zero_regression(tmp_path: Path):
    """单假设 drop → 本轮候选全删、rounds_log['verdict']=='drop'（现状行为）。"""
    drop_expr = "ts_mean(close, 5)"

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            return json.dumps({"verdict": "drop", "reason": "过拟合"})
        if "翻译成" in text:
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    daily = _mock_daily()
    with patch(
        "factorzen.agents.team_orchestrator.node_guardrails",
        _inject_guardrails_from_attempts,
    ):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=42, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"),
        )

    assert all(c["expression"] != drop_expr for c in res.candidates), \
        f"单假设 drop 应清空本轮候选: {res.candidates}"
    assert res.rounds_log[-1]["verdict"] == "drop"
    dropped = [a for a in res.state.attempts if a.expression == drop_expr]
    assert dropped and all(a.critic_verdict == "drop" for a in dropped)
    assert all(a.passed_guardrails for a in dropped)


def test_critic_called_once_per_hypothesis(tmp_path: Path):
    """两假设轮恰好调用 2 次 critique（每假设一次）。"""
    h1, h2 = "HYPC1", "HYPC2"
    n_crit = {"k": 0}

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            n_crit["k"] += 1
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in text:
            if h1 in text:
                return json.dumps({"expressions": ["ts_mean(close,5)"]})
            if h2 in text:
                return json.dumps({"expressions": ["ts_std(close,10)"]})
            return json.dumps({"expressions": ["rank(vol)"]})
        return json.dumps({"hypotheses": [h1, h2]})

    daily = _mock_daily()
    with patch(
        "factorzen.agents.team_orchestrator.node_guardrails",
        _inject_guardrails_from_attempts,
    ):
        run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"), hypotheses_per_round=2,
        )

    assert n_crit["k"] == 2, f"两假设应 critique 恰好 2 次，实得 {n_crit['k']}"
