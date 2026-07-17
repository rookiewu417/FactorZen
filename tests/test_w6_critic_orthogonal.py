# tests/test_w6_critic_orthogonal.py
"""W6：Critic prompt 注入残差 IC / 库相关（None-gating）。"""
from __future__ import annotations

import json

from factorzen.agents.roles.critic import critique


class FakeLLM:
    def __init__(self):
        self.calls: list = []

    def __call__(self, messages):
        self.calls.append(messages)
        return json.dumps({"verdict": "keep", "reason": "ok"})


def _base_cand(**kw):
    base = {
        "expression": "ts_mean(close,5)", "hypothesis": "动量",
        "ic_train": 0.05, "holdout_ic": 0.03, "dsr": 0.7, "dsr_pvalue": 0.01,
        "ir_train": 0.4, "turnover": 0.2,
    }
    base.update(kw)
    return base


def test_critique_injects_residual_and_library_corr():
    llm = FakeLLM()
    critique(_base_cand(
        residual_ic_train=0.012, residual_holdout_ic=0.008,
        max_corr_library=0.91,
    ), llm)
    assert llm.calls
    blob = "\n".join(m["content"] for m in llm.calls[0])
    assert "对库残差IC(train/holdout)" in blob
    assert "0.012" in blob and "0.008" in blob
    assert "与库最大相关" in blob
    assert "0.91" in blob
    # system 引导正交
    sys = llm.calls[0][0]["content"]
    assert "残差" in sys and "revise_hypothesis" in sys


def test_critique_omits_residual_when_missing():
    """字段缺失时不出现残差/库相关行（None-gating）。"""
    llm = FakeLLM()
    critique(_base_cand(), llm)
    blob = "\n".join(m["content"] for m in llm.calls[0])
    assert "对库残差IC" not in blob
    assert "与库最大相关" not in blob


def test_critique_partial_residual_only():
    """仅 residual_ic_train 有值时仍注入残差行，无库相关行。"""
    llm = FakeLLM()
    critique(_base_cand(residual_ic_train=0.02), llm)
    blob = "\n".join(m["content"] for m in llm.calls[0])
    assert "对库残差IC(train/holdout)" in blob
    assert "0.02" in blob
    assert "与库最大相关" not in blob
