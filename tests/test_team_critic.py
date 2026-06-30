import json

from factorzen.agents.roles.critic import CriticVerdict, critique


class FakeLLM:
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
    llm = FakeLLM([json.dumps({"verdict": "keep", "reason": "稳健"})])
    v = critique(_cand(), llm)
    assert isinstance(v, CriticVerdict) and v.verdict == "keep"


def test_critique_drop_overfit():
    # DSR 不显著的候选 → drop
    llm = FakeLLM([json.dumps({"verdict": "drop", "reason": "DSR 不显著疑过拟合"})])
    v = critique(_cand(dsr=0.2, dsr_pvalue=0.4), llm)
    assert v.verdict == "drop" and v.reason


def test_critique_revise_variants():
    llm = FakeLLM([json.dumps({"verdict": "revise_expr", "reason": "窗口太短"}),
                   json.dumps({"verdict": "revise_hypothesis", "reason": "方向牵强"})])
    assert critique(_cand(), llm).verdict == "revise_expr"
    assert critique(_cand(), llm).verdict == "revise_hypothesis"


def test_critique_garbage_defaults_keep():
    # 解析失败 → 默认 keep（不误杀；与 M5 node_critic 容错一致）
    llm = FakeLLM(["不是 JSON"])
    assert critique(_cand(), llm).verdict == "keep"


def test_critique_unknown_verdict_defaults_keep():
    llm = FakeLLM([json.dumps({"verdict": "explode", "reason": "x"})])
    assert critique(_cand(), llm).verdict == "keep"   # 非法 verdict 归一到 keep
