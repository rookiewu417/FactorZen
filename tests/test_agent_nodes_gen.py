import datetime as dt
import json

import numpy as np
import polars as pl

from factorzen.agents.nodes import node_evaluate, node_generate
from factorzen.agents.state import AgentState
from factorzen.discovery.scoring import DataBundle


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _mock_daily(n_stocks=40, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def test_node_generate_then_evaluate_populates_attempts():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    raw = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)", "rank(vol)"],
                      "rationale": "r"})
    # semantic_check 也走 llm：两次 consistent=true
    sem = json.dumps({"consistent": True, "reason": "ok"})
    llm = FakeLLM([raw, sem, sem])
    state = AgentState(seed=42)
    state = node_generate(state, llm, daily=daily, bundle=bundle)
    state = node_evaluate(state, daily=daily, bundle=bundle)
    assert len(state.attempts) == 2
    assert all(a.compile_ok for a in state.attempts)
    assert all(a.ic_train is not None for a in state.attempts)
    # 验证归一化形式（带空格）在 seen_expressions 中
    assert "ts_mean(close, 5)" in state.seen_expressions


def test_node_generate_rejects_illegal_and_records_error():
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    raw = json.dumps({"hypothesis": "h", "expressions": ["bogus_op(close)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    llm = FakeLLM([raw, sem])
    state = AgentState(seed=1)
    state = node_generate(state, llm, daily=daily, bundle=bundle)
    state = node_evaluate(state, daily=daily, bundle=bundle)
    assert state.attempts[0].compile_ok is False and state.attempts[0].error


def test_node_generate_dedup_with_normalized_form():
    """验证去重用归一化形式：原始 vs 归一化两种写法不重复进 attempts。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    # 第一轮：无空格形式
    raw1 = json.dumps({"hypothesis": "h1", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem1 = json.dumps({"consistent": True, "reason": "ok"})
    # 第二轮：有空格形式（归一化后相同）
    raw2 = json.dumps({"hypothesis": "h2", "expressions": ["ts_mean(close, 5)"], "rationale": "r"})
    sem2 = json.dumps({"consistent": True, "reason": "ok"})
    llm = FakeLLM([raw1, sem1, raw2, sem2])
    state = AgentState(seed=42)
    # 第一轮
    state = node_generate(state, llm, daily=daily, bundle=bundle)
    state = node_evaluate(state, daily=daily, bundle=bundle)
    assert len(state.attempts) == 1
    assert "ts_mean(close, 5)" in state.seen_expressions
    # 第二轮：同一表达式的不同写法应被去重
    state = node_generate(state, llm, daily=daily, bundle=bundle)
    assert len(state._pending) == 0  # type: ignore[attr-defined]
    state = node_evaluate(state, daily=daily, bundle=bundle)
    assert len(state.attempts) == 1  # 仍为 1，没有增加
