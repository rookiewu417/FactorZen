"""合并自 agents 相关碎片测试（test_team_roles.py）。

test_team_coder.py：Coder 角色：写表达式、按 critic 修订、垃圾输出空列表
test_team_critic.py：Critic 角色：keep/drop/revise 与垃圾/未知 verdict 默认 keep
test_team_critic_grouping.py：Critic 按 hypothesis 分组评判，禁止跨假设误杀
test_team_hypothesis.py：Tests for Hypothesis 角色 propose_hypotheses
test_team_librarian.py：Librarian record/recall 与 holdout_ic/critic 回填
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.coder import revise_expressions, write_expressions
from factorzen.agents.roles.critic import CriticVerdict, critique
from factorzen.agents.roles.hypothesis import propose_hypotheses
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AttemptRecord
from factorzen.agents.team_orchestrator import run_team_agent


# ==== 来自 test_team_coder.py ====
class FakeLLM__coder:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_write_expressions_lists_ops():
    llm = FakeLLM__coder([json.dumps({"expressions": ["ts_mean(close,5)", "rank(vol)"]})])
    out = write_expressions("动量", llm)
    assert out == ["ts_mean(close,5)", "rank(vol)"]
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "ts_mean" in blob and "close" in blob  # 算子/特征清单进 prompt


def test_revise_uses_critic_reason():
    llm = FakeLLM__coder([json.dumps({"expressions": ["ts_mean(close,20)"]})])
    out = revise_expressions("动量", ["ts_mean(close,5)"], "窗口太短", llm)
    assert out == ["ts_mean(close,20)"]
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "窗口太短" in blob and "ts_mean(close,5)" in blob  # 反馈+原表达式进 prompt


def test_write_garbage_returns_empty():
    llm = FakeLLM__coder(["非 JSON"])
    assert write_expressions("动量", llm) == []

# ==== 来自 test_team_critic.py ====
class FakeLLM__critic:
    def __init__(self, responses):
        self._r = list(responses)

    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _cand(**kw):
    base = {"expression": "ts_mean(close,5)", "hypothesis": "动量", "ic_train": 0.05,
            "holdout_ic": 0.03, "dsr": 0.7, "dsr_pvalue": 0.01}
    base.update(kw)
    return base


def test_critique_keep():
    llm = FakeLLM__critic([json.dumps({"verdict": "keep", "reason": "稳健"})])
    v = critique(_cand(), llm)
    assert isinstance(v, CriticVerdict) and v.verdict == "keep"


def test_critique_drop_overfit():
    # DSR 不显著的候选 → drop
    llm = FakeLLM__critic([json.dumps({"verdict": "drop", "reason": "DSR 不显著疑过拟合"})])
    v = critique(_cand(dsr=0.2, dsr_pvalue=0.4), llm)
    assert v.verdict == "drop" and v.reason


def test_critique_revise_variants():
    llm = FakeLLM__critic([json.dumps({"verdict": "revise_expr", "reason": "窗口太短"}),
                   json.dumps({"verdict": "revise_hypothesis", "reason": "方向牵强"})])
    assert critique(_cand(), llm).verdict == "revise_expr"
    assert critique(_cand(), llm).verdict == "revise_hypothesis"


def test_critique_garbage_defaults_keep():
    # 解析失败 → 默认 keep（不误杀；与 M5 node_critic 容错一致）
    llm = FakeLLM__critic(["不是 JSON"])
    assert critique(_cand(), llm).verdict == "keep"


def test_critique_unknown_verdict_defaults_keep():
    llm = FakeLLM__critic([json.dumps({"verdict": "explode", "reason": "x"})])
    assert critique(_cand(), llm).verdict == "keep"   # 非法 verdict 归一到 keep

# ==== 来自 test_team_critic_grouping.py ====
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

# ==== 来自 test_team_hypothesis.py ====
class FakeLLM__hypothesis:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_propose_returns_directions():
    llm = FakeLLM__hypothesis([json.dumps({"hypotheses": ["小市值反转", "高换手动量"]})])
    out = propose_hypotheses(llm, known_invalid=[], known_valid=[], n=2)
    assert out == ["小市值反转", "高换手动量"]


def test_known_invalid_injected_into_prompt():
    llm = FakeLLM__hypothesis([json.dumps({"hypotheses": ["x"]})])
    propose_hypotheses(llm, known_invalid=["rank(vol)"], known_valid=["ts_mean(close, 5)"], n=1)
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "rank(vol)" in blob  # 已知无效注入(避开)
    assert "ts_mean(close, 5)" in blob  # 已知有效作方向参考


def test_propose_garbage_returns_empty():
    llm = FakeLLM__hypothesis(["非 JSON"])
    assert propose_hypotheses(llm, known_invalid=[], known_valid=[]) == []

# ==== 来自 test_team_librarian.py ====
def test_record_then_recall_roundtrip(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
    ]
    record(idx, attempts, run_id="r1")
    r = recall(idx, k=5)
    assert "ts_mean(close, 5)" in r.seen and "rank(vol)" in r.seen   # 归一化查重集
    assert "rank(vol)" in r.known_invalid                            # 未过护栏
    assert "ts_mean(close, 5)" in r.known_valid                      # 过护栏


def test_recall_empty_index(tmp_path: Path):
    r = recall(ExperimentIndex(str(tmp_path / "none.jsonl")), k=5)
    assert r.seen == set() and r.known_invalid == [] and r.known_valid == []


def test_record_backfills_holdout_ic(tmp_path: Path):
    """record(candidates=...) → holdout_ic 写入 index → known_valid 按 holdout_ic 降序。

    注意：idx.load() 返回原始（非归一化）expression 字符串，
    candidates 可用归一化或非归一化形式，_normalize 匹配均可；
    known_valid() 返回归一化形式。
    """
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "反转", "rank(vol)", True, 0.03, True, "keep", None, ir_train=0.2),
    ]
    # rank(vol) 的 holdout_ic 更高——若排序正确，known_valid[0] 应为 rank(vol)
    # candidates 用归一化形式（空格）验证 _normalize 匹配路径
    candidates = [
        {"expression": "ts_mean(close, 5)", "holdout_ic": 0.02, "ic_train": 0.05},
        {"expression": "rank(vol)", "holdout_ic": 0.06, "ic_train": 0.03},
    ]
    record(idx, attempts, run_id="r1", candidates=candidates)
    recs = idx.load()
    # idx.load() 返回原始 expression（AttemptRecord.expression，无空格）
    hic_map = {r["expression"]: r.get("holdout_ic") for r in recs}
    assert hic_map.get("ts_mean(close,5)") == 0.02, f"holdout_ic 未写入: {hic_map}"
    assert hic_map.get("rank(vol)") == 0.06
    # known_valid 按 holdout_ic 降序，返回归一化形式：rank(vol) 排第一
    r = recall(idx, k=5)
    assert r.known_valid[0] == "rank(vol)", f"期望 rank(vol) 排第一，实际 {r.known_valid}"


def test_record_backfills_critic_verdict(tmp_path: Path):
    """AttemptRecord.critic_verdict 非 None 时，record 正确写入 verdict 字段。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    attempts = [
        AttemptRecord(0, "动量", "ts_mean(close,5)", True, 0.05, True, "keep", None, ir_train=0.4),
        AttemptRecord(0, "换手", "rank(vol)", True, 0.001, False, "drop", None, ir_train=0.01),
    ]
    record(idx, attempts, run_id="r1")
    recs = idx.load()
    # idx.load() 返回原始 expression（AttemptRecord.expression，无空格）
    verdict_map = {r["expression"]: r.get("verdict") for r in recs}
    assert verdict_map.get("ts_mean(close,5)") == "keep"
    assert verdict_map.get("rank(vol)") == "drop"
