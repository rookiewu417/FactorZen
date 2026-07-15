# tests/test_agent_health_check.py
"""P1②：自愈循环纳入**求值期**诊断（CoSTEER 的另一半）。

原 self_heal 只在 `parse_expr` 抛 ValueError 时回灌 LLM —— 只修语法错。而
`evaluate_expressions` 里真正的运行期失败被 `except Exception` 捕获、写进
`AttemptRecord.error` 后，全仓库无人读取，从不回灌。

对照 RD-Agent：CoSTEER 的评估器在沙箱里**真正执行**代码，把 Traceback **和 NaN 比例**
交给错误摘要模型压成建议再回灌。PR #61 的「嵌套 .over() 全 null」正是一个 parse 成功、
求值静默返回全 null 的 bug —— 那类因子在旧 heal 循环里连一次修正机会都没有。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from factorzen.agents.self_heal import heal_expressions
from factorzen.discovery.evaluation import make_health_check

_ALL_NULL = "div(close, sub(close, close))"   # 分母恒 0 → _safe_div 全列 null
_HEALTHY = "ts_mean(close, 5)"


def _mock_daily(n_days: int = 60, n_codes: int = 5) -> pl.DataFrame:
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
    check = make_health_check(_mock_daily())
    assert check(_HEALTHY) is None


def test_all_null_factor_is_diagnosed():
    """全 null = 静默失明，必须被抓出来（PR #61 那类 bug 的兜底）。"""
    check = make_health_check(_mock_daily())
    diag = check(_ALL_NULL)
    assert diag is not None
    assert "null" in diag.lower() or "NaN" in diag


def test_null_ratio_threshold_is_configurable_and_respected():
    """健康表达式在极严阈值下也会被判不健康 —— 证明判据真的是比例而非硬编码。"""
    daily = _mock_daily()
    assert make_health_check(daily, max_null_ratio=0.5)(_HEALTHY) is None
    assert make_health_check(daily, max_null_ratio=0.001)(_HEALTHY) is not None


def test_parse_error_is_diagnosed():
    check = make_health_check(_mock_daily())
    diag = check("not_a_func(")
    assert diag is not None and "解析" in diag


def test_eval_error_is_diagnosed(monkeypatch):
    """求值抛异常 → 诊断带上异常类型与消息（供 LLM 修正）。"""
    from factorzen.discovery import evaluation as ev

    check = ev.make_health_check(_mock_daily())

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
                              health_check=make_health_check(_mock_daily()))

    assert healed == [_HEALTHY]
    assert len(prompts) == 1, "健康表达式不应再触发第二次修正"
    assert _ALL_NULL in prompts[0]
    assert "null" in prompts[0].lower() or "NaN" in prompts[0]


def test_heal_gives_up_when_llm_keeps_producing_unhealthy_expressions():
    """LLM 持续产全 null → max_rounds 耗尽后丢弃，不死循环、不放行病态因子。"""
    def fake(_msgs):
        return json.dumps({"expressions": [_ALL_NULL]})

    healed = heal_expressions([_ALL_NULL], "h", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily()))
    assert healed == []


def test_healthy_expression_never_triggers_llm_even_with_health_check():
    """零额外成本不变量：健康表达式不调用 LLM。"""
    def fake(_msgs):
        raise AssertionError("健康表达式不应触发 LLM 修正")

    healed = heal_expressions([_HEALTHY], "h", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily()))
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

    daily = _mock_daily(n_days=120, n_codes=10)
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
    daily = _mock_daily(n_days=200, n_codes=40)
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
