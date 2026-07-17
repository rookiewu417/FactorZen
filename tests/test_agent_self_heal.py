# tests/test_agent_self_heal.py
"""Workstream D：表达式层自愈循环（CoSTEER 轻量版，DSL 层无 exec 沙箱）。"""
from __future__ import annotations

import json

from factorzen.agents.self_heal import heal_expressions
from factorzen.discovery.expression import parse_expr


def test_heal_fixes_parse_error():
    """语法错（非未知算子）→ 报错回灌 LLM 修正 → 产出可解析表达式。

    W5a：``not_a_func(`` 实际是未知算子，默认直接丢弃不进 heal；
    本测改用 ``ts_mean()``（缺窗口参数）作为可修语法错。
    """
    def fake(_msgs):
        return json.dumps({"expressions": ["ts_mean(close, 5)"]})
    healed = heal_expressions(["ts_mean()"], "动量", fake, max_rounds=2)
    assert len(healed) >= 1
    for h in healed:
        parse_expr(h)  # 全部可解析


def test_heal_valid_expr_no_llm_call():
    """可解析表达式不触发 LLM（零额外成本）。"""
    def fake(_msgs):
        raise AssertionError("valid expr 不应触发 LLM 修正")
    healed = heal_expressions(["ts_mean(close, 5)"], "h", fake, max_rounds=2)
    assert len(healed) == 1
    parse_expr(healed[0])


def test_heal_gives_up_after_max_rounds():
    """LLM 持续产语法错 → max_rounds 耗尽后丢弃（不死循环）。

    用 ``add(close)``（arity 错）而非未知算子，确保走 heal 路径。
    """
    def fake(_msgs):
        return json.dumps({"expressions": ["add(close)"]})
    healed = heal_expressions(["add(close)"], "h", fake, max_rounds=2)
    assert healed == []


def test_heal_dedup_and_mixed():
    """有效 + 语法错混合：有效直通，语法错修正，结果去重。"""
    def fake(_msgs):
        return json.dumps({"expressions": ["rank(vol)"]})
    healed = heal_expressions(["ts_mean(close, 5)", "ts_mean()"], "h", fake, max_rounds=2)
    assert len(healed) == len(set(healed))
    for h in healed:
        parse_expr(h)


def test_node_generate_heal_rounds_zero_disables_healing():
    """heal_rounds=0 → 关闭自愈：非法表达式原样进入 pending，不触发 revise LLM 调用。

    原测试是 `assert "heal_rounds" in inspect.signature(node_generate).parameters`——
    形参存在不等于调用方传、也不等于它起作用，对接线缺口零判别力。改为观察 LLM 调用次数
    与 pending 内容这两个真实行为。（heal_rounds>0 的行为见 test_agent_health_check.py）
    """
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.nodes import node_generate
    from factorzen.agents.state import AgentState
    from factorzen.discovery.scoring import DataBundle

    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 90:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(6)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98, "vol": 1e6, "amount": 1e7})
    daily = pl.DataFrame(rows)

    calls: list[list] = []
    seq = [json.dumps({"hypothesis": "h", "expressions": ["bad("], "rationale": "r"}),
           json.dumps({"consistent": True, "reason": "ok"})]

    def fn(msgs):
        calls.append(msgs)
        return seq[min(len(calls) - 1, len(seq) - 1)]

    state = node_generate(AgentState(seed=1), fn, daily=daily,
                          bundle=DataBundle.build(daily), heal_rounds=0)

    assert len(calls) == 2, f"应只有 proposal + semantic_check 两次调用，实得 {len(calls)}"
    assert [p.expression for p in state._pending] == ["bad("]
