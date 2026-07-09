# tests/test_agent_ashare_caveats.py
"""Workstream E：A股机制 + PIT 陷阱 Prompt 注入（研报优化方向①）。"""
from __future__ import annotations


def test_caveats_fragment_covers_key_mechanisms():
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS
    for kw in ["涨跌停", "停牌", "T+1", "PIT", "换手", "风险因子"]:
        assert kw in ASHARE_CAVEATS, f"缺少 {kw}"


def test_build_agent_messages_injects_caveats():
    from factorzen.llm.generation import build_agent_messages
    sys = build_agent_messages(["ts_mean"], ["close"], "", [])[0]["content"]
    assert "涨跌停" in sys and "T+1" in sys


def test_hypothesis_prompt_injects_caveats():
    from factorzen.agents.roles.hypothesis import propose_hypotheses
    cap: dict = {}

    def fake(msgs):
        cap["m"] = msgs
        return '{"hypotheses":["x"]}'
    propose_hypotheses(fake, known_invalid=[], known_valid=[])
    assert "涨跌停" in cap["m"][0]["content"]


def test_coder_prompt_injects_caveats():
    from factorzen.agents.roles.coder import write_expressions
    cap: dict = {}

    def fake(msgs):
        cap["m"] = msgs
        return '{"expressions":["ts_mean(close,5)"]}'
    write_expressions("动量", fake)
    sys = cap["m"][0]["content"]
    assert "PIT" in sys or "涨跌停" in sys
