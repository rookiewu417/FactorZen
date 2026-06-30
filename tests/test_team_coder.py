import json

from factorzen.agents.roles.coder import revise_expressions, write_expressions


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self._r.pop(0) if self._r else "{}"


def test_write_expressions_lists_ops():
    llm = FakeLLM([json.dumps({"expressions": ["ts_mean(close,5)", "rank(vol)"]})])
    out = write_expressions("动量", llm)
    assert out == ["ts_mean(close,5)", "rank(vol)"]
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "ts_mean" in blob and "close" in blob  # 算子/特征清单进 prompt


def test_revise_uses_critic_reason():
    llm = FakeLLM([json.dumps({"expressions": ["ts_mean(close,20)"]})])
    out = revise_expressions("动量", ["ts_mean(close,5)"], "窗口太短", llm)
    assert out == ["ts_mean(close,20)"]
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "窗口太短" in blob and "ts_mean(close,5)" in blob  # 反馈+原表达式进 prompt


def test_write_garbage_returns_empty():
    llm = FakeLLM(["非 JSON"])
    assert write_expressions("动量", llm) == []
