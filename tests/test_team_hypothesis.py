"""Tests for Hypothesis 角色 propose_hypotheses。"""
import json

from factorzen.agents.roles.hypothesis import propose_hypotheses


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_propose_returns_directions():
    llm = FakeLLM([json.dumps({"hypotheses": ["小市值反转", "高换手动量"]})])
    out = propose_hypotheses(llm, known_invalid=[], known_valid=[], n=2)
    assert out == ["小市值反转", "高换手动量"]


def test_known_invalid_injected_into_prompt():
    llm = FakeLLM([json.dumps({"hypotheses": ["x"]})])
    propose_hypotheses(llm, known_invalid=["rank(vol)"], known_valid=["ts_mean(close, 5)"], n=1)
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "rank(vol)" in blob  # 已知无效注入(避开)
    assert "ts_mean(close, 5)" in blob  # 已知有效作方向参考


def test_propose_garbage_returns_empty():
    llm = FakeLLM(["非 JSON"])
    assert propose_hypotheses(llm, known_invalid=[], known_valid=[]) == []
