"""合并自 agents 相关碎片测试（test_eval_hygiene.py）。

test_exception_and_accounting_hygiene.py：异常日志、critic 围栏 JSON、compile 失败入账与 deflation 池隔离
test_w5_llm_waste.py：W5 减少 LLM 浪费：未知 op 直接 drop、window clamp 预算
test_agent_health_check.py：P1②：自愈循环纳入求值期诊断（CoSTEER 的另一半）
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from factorzen.agents.self_heal import heal_expressions
from factorzen.discovery.evaluation import make_health_check
from factorzen.discovery.expression import clamp_window_literals, parse_expr


# ==== 来自 test_exception_and_accounting_hygiene.py ====
def _daily(n_stocks: int = 40, n_days: int = 200, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99, "vol": 1e6, "amount": 1e7})
    return pl.DataFrame(rows)


# ── 1. 静默吞异常 ───────────────────────────────────────────────────────────


def test_node_guardrails_logs_when_a_candidate_blows_up(caplog):
    """一个候选的 holdout 求值失败必须留下日志，而不是静默 continue。

    静默 continue 会让「这个候选炸了」与「这个候选没过护栏」在产物上不可区分。
    """
    import factorzen.validation.holdout as hmod
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _daily()
    bundle = DataBundle.build(daily)
    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="rank(close)", compile_ok=True,
        ic_train=0.05, passed_guardrails=False, critic_verdict=None, error=None,
        ir_train=0.4, n_train=150))

    orig = hmod.holdout_ic_result

    def boom(*_a, **_kw):
        raise RuntimeError("holdout 求值失败")

    hmod.holdout_ic_result = boom
    try:
        with caplog.at_level(logging.WARNING, logger="factorzen.agents.nodes"):
            node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                            ledger=TrialLedger(), top_k=5, warmup_daily=daily)
    finally:
        hmod.holdout_ic_result = orig

    assert not state.candidates
    assert any("rank(close)" in r.getMessage() for r in caplog.records), \
        "候选护栏计算失败必须记日志（含表达式），不得静默吞掉"


# ── 2. node_critic 的双路径漂移 ─────────────────────────────────────────────


def test_node_critic_parses_fenced_json_like_team_critic_does():
    """真实 LLM 常返回 markdown 围栏。单 Agent 与 team 的 Critic 必须给出同一裁决。"""
    from factorzen.agents.nodes import node_critic
    from factorzen.agents.roles.critic import critique
    from factorzen.agents.state import AgentState, AttemptRecord

    fenced = '```json\n{"verdict": "drop", "reason": "过拟合"}\n```'

    def llm(_msgs):
        return fenced

    state = AgentState(seed=0)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="close", compile_ok=True, ic_train=0.5,
        passed_guardrails=True, critic_verdict=None, error=None, ir_train=0.3))
    node_critic(state, llm)

    assert state.attempts[0].critic_verdict == critique({"expression": "close"}, llm).verdict
    assert state.attempts[0].critic_verdict == "drop", "LLM 说 drop，不该被静默降级为 keep"


def test_node_critic_still_defaults_to_keep_on_garbage():
    """完全无法解析时仍 fail-open（不误杀），但那是**解析失败**，不是格式差异。"""
    from factorzen.agents.nodes import node_critic
    from factorzen.agents.state import AgentState, AttemptRecord

    state = AgentState(seed=0)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="close", compile_ok=True, ic_train=0.5,
        passed_guardrails=True, critic_verdict=None, error=None, ir_train=0.3))
    node_critic(state, lambda _m: "彻底不是 JSON")

    assert state.attempts[0].critic_verdict == "keep"



# ── 3. 编译失败的表达式必须进长期记忆 ───────────────────────────────────────


def test_record_persists_compile_failures_so_they_are_not_retried(tmp_path):
    """坏表达式不进 index → `seen_expressions()` 看不到 → 跨 session 反复生成同一语法坑。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import record
    from factorzen.agents.state import AttemptRecord

    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    bad = AttemptRecord(iteration=0, hypothesis="h", expression="ts_mean(close)",
                        compile_ok=False, ic_train=None, passed_guardrails=False,
                        critic_verdict=None, error="缺少窗口参数", ir_train=None)
    good = AttemptRecord(iteration=0, hypothesis="h", expression="rank(close)",
                         compile_ok=True, ic_train=0.02, passed_guardrails=False,
                         critic_verdict="keep", error=None, ir_train=0.1, n_train=200)
    record(idx, [bad, good], run_id="r1")

    assert idx.seen_expressions() == {"ts_mean(close)", "rank(close)"}, \
        "编译失败的表达式必须进 seen，否则跨 session 会重复生成"
    stored = {r["expression"]: r for r in idx.load()}
    assert stored["ts_mean(close)"]["compile_ok"] is False
    assert stored["ts_mean(close)"]["error"] == "缺少窗口参数"


def test_known_invalid_excludes_compile_failures(tmp_path):
    """`known_invalid` 的语义是「能编译但无效」。语法坑 ic_train=None → 排序键 0.0 会排最前，
    把有信息的低 IC 负例全部挤出 top-k。它们的价值在 seen 去重，不在负例库。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import record
    from factorzen.agents.state import AttemptRecord

    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [
        AttemptRecord(0, "h", "bad_syntax", False, None, False, None, "boom", None),
        AttemptRecord(0, "h", "low_ic", True, 0.001, False, None, None, 0.01, n_train=200),
    ], run_id="r1")

    assert idx.known_invalid(k=5) == ["low_ic"], "语法坑不该占据「已验证无效」负例库"


def test_compile_failures_never_enter_the_deflation_pool(tmp_path):
    """编译失败记录的 ir_train=None，必须被 DeflationBasis 剔除——否则污染 N 与经验方差。"""
    from factorzen.discovery.guardrails import DeflationBasis

    basis = DeflationBasis.from_ir_pool([0.3, None, 0.1])
    assert basis.n_trials == 2


# ── 4. 同轮重复表达式的 N over-count ────────────────────────────────────────


def test_team_evaluate_deduplicates_within_the_batch():
    """`heal_rounds=0` 时 heal 的去重不生效；多个 task 翻译出同一表达式 → 不得评估两次。

    N 是多重检验的记账，多算一次就是记账不诚实（方向偏严，但仍是错的）。
    """
    from factorzen.agents.state import AgentState
    from factorzen.agents.team_orchestrator import _evaluate_and_record
    from factorzen.discovery.scoring import DataBundle

    daily = _daily(n_days=150, seed=7)
    bundle = DataBundle.build(daily)
    state = AgentState(seed=0)

    exprs = ["ts_mean(close, 5)", "ts_mean(close,5)", "rank(close)"]  # 前两个归一化后同一个
    results = _evaluate_and_record(state, exprs, "h", daily=daily, bundle=bundle, mem_seen=set())

    assert len(results) == 2, f"3 个输入含 1 个重复 → 只该评估 2 次，实得 {len(results)}"
    assert len(state.attempts) == 2
    uniq = {a.expression for a in state.attempts}
    assert len(uniq) == 2

    passed = [a for a in state.attempts if a.compile_ok and a.ic_train is not None]
    assert len(passed) == len(uniq), "node_guardrails 记的 N 必须等于唯一表达式数"



@pytest.mark.parametrize("garbage", ["", "{", "不是json", "[1,2]"])
def test_extract_json_never_raises_on_string_garbage(garbage):
    """回归守卫：`_extract_json` 对任何字符串都不抛（只返回 None 或 dict）。"""
    from factorzen.llm.generation import _extract_json

    out = _extract_json(garbage)
    assert out is None or isinstance(out, dict)

# ==== 来自 test_w5_llm_waste.py ====
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

def _mock_daily__llm_waste(n_stocks=30, n_days=120, seed=3):
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

    daily = _mock_daily__llm_waste()
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

    daily = _mock_daily__llm_waste()
    with patch("factorzen.agents.team_orchestrator.node_guardrails", side_effect=_inject):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"),
        )
    assert critic_calls["n"] >= 1
    assert res.rounds_log[0].get("critic_skipped") is False

# ==== 来自 test_agent_health_check.py ====
_ALL_NULL = "div(close, sub(close, close))"   # 分母恒 0 → _safe_div 全列 null
_HEALTHY = "ts_mean(close, 5)"


def _mock_daily__health_check(n_days: int = 60, n_codes: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(n_codes)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98, "vol": 1e6, "amount": 1e7})
    return pl.DataFrame(rows)


# ─────────────────────────── make_health_check ───────────────────────────

def test_healthy_expression_reports_no_diagnosis():
    check = make_health_check(_mock_daily__health_check())
    assert check(_HEALTHY) is None


def test_all_null_factor_is_diagnosed():
    """全 null = 静默失明，必须被抓出来（PR #61 那类 bug 的兜底）。"""
    check = make_health_check(_mock_daily__health_check())
    diag = check(_ALL_NULL)
    assert diag is not None
    assert "null" in diag.lower() or "NaN" in diag


def test_null_ratio_threshold_is_configurable_and_respected():
    """健康表达式在极严阈值下也会被判不健康 —— 证明判据真的是比例而非硬编码。"""
    daily = _mock_daily__health_check()
    assert make_health_check(daily, max_null_ratio=0.5)(_HEALTHY) is None
    assert make_health_check(daily, max_null_ratio=0.001)(_HEALTHY) is not None


def test_parse_error_is_diagnosed():
    check = make_health_check(_mock_daily__health_check())
    diag = check("not_a_func(")
    assert diag is not None and "解析" in diag


def test_eval_error_is_diagnosed(monkeypatch):
    """求值抛异常 → 诊断带上异常类型与消息（供 LLM 修正）。"""
    from factorzen.discovery import evaluation as ev

    check = ev.make_health_check(_mock_daily__health_check())

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ev, "evaluate_materialized", boom)
    diag = check(_HEALTHY)
    assert diag is not None
    assert "求值失败" in diag and "RuntimeError" in diag and "boom" in diag


# ─────────────────── heal_expressions × health_check 集成 ───────────────────

def test_heal_revises_expression_that_parses_but_evaluates_all_null():
    """parse 通过但全 null → 诊断回灌 LLM → 换成健康表达式。"""
    prompts: list[str] = []

    def fake(msgs):
        prompts.append(msgs[1]["content"])
        return json.dumps({"expressions": [_HEALTHY]})

    healed = heal_expressions([_ALL_NULL], "动量", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily__health_check()))

    assert healed == [_HEALTHY]
    assert len(prompts) == 1, "健康表达式不应再触发第二次修正"
    assert _ALL_NULL in prompts[0]
    assert "null" in prompts[0].lower() or "NaN" in prompts[0]


def test_heal_gives_up_when_llm_keeps_producing_unhealthy_expressions():
    """LLM 持续产全 null → max_rounds 耗尽后丢弃，不死循环、不放行病态因子。"""
    def fake(_msgs):
        return json.dumps({"expressions": [_ALL_NULL]})

    healed = heal_expressions([_ALL_NULL], "h", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily__health_check()))
    assert healed == []


def test_healthy_expression_never_triggers_llm_even_with_health_check():
    """零额外成本不变量：健康表达式不调用 LLM。"""
    def fake(_msgs):
        raise AssertionError("健康表达式不应触发 LLM 修正")

    healed = heal_expressions([_HEALTHY], "h", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily__health_check()))
    assert len(healed) == 1


def test_no_health_check_is_zero_regression():
    """health_check=None（默认）→ 只查 parse，全 null 表达式照旧放行（既有行为）。"""
    def fake(_msgs):
        raise AssertionError("parse 通过的表达式不应触发 LLM")

    healed = heal_expressions([_ALL_NULL], "h", fake, max_rounds=2)
    assert len(healed) == 1
    assert "div" in healed[0]


# ─────────────────── 接线：health_check 必须真的抵达自愈循环 ───────────────────
# 能力层实现完 ≠ 用户用得上。这两条从 node_generate / run_team_agent 这一层出发，
# 断言全 null 表达式在真实闭环里被诊断并修正掉，而不是靠 inspect.signature 看形参。

def test_node_generate_heals_all_null_expression_end_to_end():
    """M5 单 Agent：LLM 产出全 null 表达式 → 求值诊断回灌 → pending 里只剩健康表达式。"""
    from factorzen.agents.nodes import node_generate
    from factorzen.agents.state import AgentState
    from factorzen.discovery.scoring import DataBundle

    daily = _mock_daily__health_check(n_days=120, n_codes=10)
    seq = [
        json.dumps({"hypothesis": "动量", "expressions": [_ALL_NULL], "rationale": "r"}),
        json.dumps({"expressions": [_HEALTHY]}),        # revise_from_error 的修正
        json.dumps({"consistent": True, "reason": "ok"}),  # semantic_check
    ]
    i = {"k": 0}

    def fn(_m):
        v = seq[min(i["k"], len(seq) - 1)]
        i["k"] += 1
        return v

    state = node_generate(AgentState(seed=1), fn, daily=daily,
                          bundle=DataBundle.build(daily), heal_rounds=2)
    pending = [p.expression for p in state._pending]
    assert pending == [_HEALTHY], f"全 null 表达式未被自愈: {pending}"


def test_team_agent_heals_all_null_expression_end_to_end(tmp_path):
    """M6 团队：Coder 写出全 null 表达式 → 求值诊断回灌 → 落库的是健康表达式。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    # n_codes≥30：叶子 holdout 覆盖门用 _MIN_CROSS_SAMPLES=30，截面太薄会被整批摘叶导致 0 评估。
    daily = _mock_daily__health_check(n_days=200, n_codes=40)
    seq = [
        json.dumps({"hypotheses": ["动量"]}),           # propose_hypotheses
        json.dumps({"expressions": [_ALL_NULL]}),       # write_expressions
        json.dumps({"expressions": [_HEALTHY]}),        # revise_from_error
        json.dumps({"verdict": "keep", "reason": "ok"}),  # critique
    ]
    i = {"k": 0}

    def fn(_m):
        v = seq[min(i["k"], len(seq) - 1)]
        i["k"] += 1
        return v

    res = run_team_agent(daily, fn, n_rounds=1, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), heal_rounds=2)
    exprs = [a.expression for a in res.state.attempts]
    assert exprs, "本轮应有被评估的表达式"
    assert all("div" not in e for e in exprs), f"全 null 表达式未被自愈就进了评估: {exprs}"
