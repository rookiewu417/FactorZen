import json

from factorzen.llm.generation import (
    build_agent_messages,
    generate_factor_proposal,
    semantic_check,
)


class FakeLLM:
    """确定性 LLMFn：按调用顺序返回预设字符串。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else "{}"


def test_generate_factor_proposal_parses_json():
    raw = json.dumps(
        {
            "hypothesis": "低换手反转",
            "expressions": ["rank(close)", "ts_mean(vol,5)"],
            "rationale": "...",
        }
    )
    llm = FakeLLM([raw])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm, n_hypotheses=1)
    assert len(props) == 1
    assert props[0].hypothesis == "低换手反转"
    assert props[0].expressions == ["rank(close)", "ts_mean(vol,5)"]


def test_generate_factor_proposal_tolerates_garbage():
    # 非 JSON → 返回空列表（降级，不抛）
    llm = FakeLLM(["这不是 JSON"])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm)
    assert props == []


def test_generate_extracts_json_substring():
    # JSON 嵌在自然语言里 → 提取首个 {...}
    raw = '好的，这是我的提议：{"hypothesis":"h","expressions":["rank(close)"],"rationale":"r"} 完毕'
    llm = FakeLLM([raw])
    props = generate_factor_proposal([{"role": "user", "content": "x"}], llm)
    assert props and props[0].expressions == ["rank(close)"]


def test_semantic_check_yes_no():
    llm = FakeLLM(
        [
            json.dumps({"consistent": True, "reason": "对齐"}),
            json.dumps({"consistent": False, "reason": "表达式与假设无关"}),
        ]
    )
    ok1, _ = semantic_check("动量", "ts_mean(close,20)", llm)
    ok2, reason2 = semantic_check("动量", "rank(pb)", llm)
    assert ok1 is True and ok2 is False and reason2


def test_build_agent_messages_lists_ops_and_leaves():
    msgs = build_agent_messages(
        op_names=["ts_mean", "rank", "div"],
        leaf_names=["close", "vol", "pb"],
        feedback="上轮 IC 偏低",
        negatives=["rank(close)"],
    )
    blob = " ".join(m["content"] for m in msgs)
    assert "ts_mean" in blob and "close" in blob  # 算子/特征清单进 prompt
    assert "rank(close)" in blob  # Negative RAG 负例进 prompt
    assert any(m["role"] == "system" for m in msgs)
