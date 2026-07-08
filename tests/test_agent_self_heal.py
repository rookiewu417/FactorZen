# tests/test_agent_self_heal.py
"""Workstream D：表达式层自愈循环（CoSTEER 轻量版，DSL 层无 exec 沙箱）。"""
from __future__ import annotations

import json

from factorzen.agents.self_heal import heal_expressions
from factorzen.discovery.expression import parse_expr


def test_heal_fixes_parse_error():
    """非法表达式 → 报错回灌 LLM 修正 → 产出可解析表达式。"""
    def fake(_msgs):
        return json.dumps({"expressions": ["ts_mean(close, 5)"]})
    healed = heal_expressions(["not_a_func("], "动量", fake, max_rounds=2)
    assert len(healed) >= 1
    for h in healed:
        parse_expr(h)  # 全部可解析
    assert all("not_a_func" not in h for h in healed)


def test_heal_valid_expr_no_llm_call():
    """可解析表达式不触发 LLM（零额外成本）。"""
    def fake(_msgs):
        raise AssertionError("valid expr 不应触发 LLM 修正")
    healed = heal_expressions(["ts_mean(close, 5)"], "h", fake, max_rounds=2)
    assert len(healed) == 1
    parse_expr(healed[0])


def test_heal_gives_up_after_max_rounds():
    """LLM 持续产非法 → max_rounds 耗尽后丢弃（不死循环）。"""
    def fake(_msgs):
        return json.dumps({"expressions": ["still_bad("]})
    healed = heal_expressions(["bad("], "h", fake, max_rounds=2)
    assert healed == []


def test_heal_dedup_and_mixed():
    """有效 + 无效混合：有效直通，无效修正，结果去重。"""
    def fake(_msgs):
        return json.dumps({"expressions": ["rank(vol)"]})
    healed = heal_expressions(["ts_mean(close, 5)", "bad("], "h", fake, max_rounds=2)
    assert len(healed) == len(set(healed))
    for h in healed:
        parse_expr(h)


def test_node_generate_heal_rounds_param():
    import inspect

    from factorzen.agents.nodes import node_generate
    assert "heal_rounds" in inspect.signature(node_generate).parameters
