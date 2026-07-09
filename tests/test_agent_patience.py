# tests/test_agent_patience.py
"""Workstream G：自适应终止（连续 patience 轮无新 passed 候选则早停）。

此前 M5(单 Agent) 的 patience 只有 `assert "patience" in inspect.signature(run_llm_agent)`
——**参数存在 ≠ 参数被真正使用**，零判别力。且两条路径的**计数器重置分支**都没测：
唯一的行为测试用「候选永不增长」的退化场景，在那里「正确逻辑」与「计数器永不重置」的
变异体行为完全一致。「每轮持续产出候选却被误早停」这个真实 bug 没有测试能抓。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent


def _mock_daily(n_stocks=20, n_days=180, seed=1):
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
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _fn_invalid():
    """始终产非法表达式 → 永无候选过护栏 → 触发 patience 早停。"""
    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["not_a_func("]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def test_team_patience_early_stops(tmp_path):
    res = run_team_agent(_mock_daily(), _fn_invalid(), n_rounds=8, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), patience=2)
    assert res.state.iteration < 8, f"patience 未早停, iteration={res.state.iteration}"


def test_team_patience_none_runs_all_rounds(tmp_path):
    """patience=None（默认）→ 跑满 n_rounds（零回归）。"""
    res = run_team_agent(_mock_daily(), _fn_invalid(), n_rounds=3, seed=1,
                         index_path=str(tmp_path / "e.jsonl"))
    assert res.state.iteration == 3


# ── 计数器重置分支：此前两条路径都没测 ──────────────────────────────────────
#
# 唯一的行为测试用 `_fn_invalid`（候选**永不增长**）。在那个退化场景下，「正确逻辑」与
# 「计数器永不重置」的变异体行为完全一致——测试对重置分支零判别力。
# 「每轮持续产出候选却被误早停」这个真实 bug，此前没有任何测试能抓到。


def _fn_valid():
    """始终产出合法表达式；候选由 stub 的 node_guardrails 注入。"""
    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn


def _stub_guardrails(*, yields_candidate: bool):
    """替身护栏：`yields_candidate=True` 时每轮新增一个候选。"""
    def fake(state, *, daily, holdout_df, bundle, ledger, top_k=5, dsr_alpha=0.05,
             warmup_daily=None):
        ledger.record(1)
        if yields_candidate:
            state.candidates.append({"expression": f"e{state.iteration}", "hypothesis": "h",
                                     "ic_train": 0.05, "ir_train": 0.4, "turnover": 0.3,
                                     "holdout_ic": 0.04, "holdout_ir": 0.3,
                                     "dsr": 0.9, "dsr_pvalue": 0.01})
        return state
    return fake


def test_team_patience_resets_when_a_new_candidate_appears(tmp_path, monkeypatch):
    """每轮都出新候选 → 计数器必须重置 → 不该被 patience=2 早停。"""
    import factorzen.agents.team_orchestrator as team

    monkeypatch.setattr(team, "node_guardrails", _stub_guardrails(yields_candidate=True))
    res = run_team_agent(_mock_daily(), _fn_valid(), n_rounds=6, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), patience=2)

    assert res.state.iteration == 6, \
        f"每轮都有新候选，patience 不该触发；实得 iteration={res.state.iteration}"


def test_m5_patience_early_stops(monkeypatch):
    """单 Agent 路径的 patience **行为**此前从未被测（只断言了 signature 里有这个参数）。"""
    import factorzen.agents.orchestrator as orch
    from factorzen.agents.orchestrator import run_llm_agent

    monkeypatch.setattr(orch, "node_guardrails", _stub_guardrails(yields_candidate=False))
    res = run_llm_agent(_mock_daily(n_stocks=40), _fn_m5(), n_rounds=8, seed=1, patience=2)

    assert res.state.iteration == 2, f"连续 2 轮无新候选应早停，实得 {res.state.iteration}"


def test_m5_patience_resets_when_a_new_candidate_appears(monkeypatch):
    import factorzen.agents.orchestrator as orch
    from factorzen.agents.orchestrator import run_llm_agent

    monkeypatch.setattr(orch, "node_guardrails", _stub_guardrails(yields_candidate=True))
    res = run_llm_agent(_mock_daily(n_stocks=40), _fn_m5(), n_rounds=5, seed=1, patience=2)

    assert res.state.iteration == 5, "每轮都有新候选，patience 不该触发"


def _fn_m5():
    """单 Agent 的 LLM 脚本：proposal → semantic → critic，每轮表达式不同以避开去重。"""
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "consistent" in system:
            return json.dumps({"consistent": True, "reason": "ok"})
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        st["round"] += 1
        return json.dumps({"hypothesis": f"h{st['round']}",
                           "expressions": [f"ts_mean(close,{4 + st['round']})"],
                           "rationale": "r"})
    return fn


# ── patience=0 的边界：CLI 不该放行一个语义反直觉的值 ────────────────────────


def test_cli_rejects_non_positive_patience():
    """`no_improve >= patience` 在 patience=0 时于第 2 轮开头恒成立——**即使刚产出新候选**。

    于是 `--patience 0` 静默变成「只跑 1 轮」，无视 `--iterations`。而 help 文案说的是
    「连续 N 轮无新候选则早停」，用户传 0 期望「不早停/更激进」，得到的却相反。
    """
    import pytest

    from factorzen.cli.main import main

    for bad in ("0", "-1"):
        with pytest.raises(SystemExit) as ei:
            main(["mine", "agent", "--start", "20220101", "--end", "20231229",
                  "--patience", bad])
        assert ei.value.code == 2, "argparse 参数校验失败应退出码 2"

