"""合并自 agents 相关碎片测试（test_nodes_gen.py）。

test_agent_nodes_gen.py：生成侧节点：node_generate 填充 attempts、非法式拒绝、规范化去重；eval 缺 warmup 必失败
test_agent_structured.py：Workstream C：结构化假设（RD-Agent 步1）+ 任务分解（步2）
test_agent_self_heal.py：Workstream D：表达式层自愈循环（CoSTEER 轻量版，DSL 层无 exec 沙箱）
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_evaluate, node_generate
from factorzen.agents.roles.coder import decompose_tasks
from factorzen.agents.roles.hypothesis import format_structured, propose_structured
from factorzen.agents.self_heal import heal_expressions
from factorzen.agents.state import AgentState
from factorzen.discovery.expression import parse_expr
from factorzen.discovery.scoring import DataBundle


# ==== 来自 test_agent_nodes_gen.py ====
class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
    def __call__(self, messages):
        return self._r.pop(0) if self._r else "{}"


def _mock_daily__nodes_gen(n_stocks=40, n_days=120, seed=1):
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
    daily = _mock_daily__nodes_gen()
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


def test_node_evaluate_raises_when_eval_start_set_without_warmup_daily():
    """eval_start 已设但漏传 warmup_daily → 必须出声，而不是静默裸求值。

    静默退回 `evaluate_expressions(exprs, daily, bundle)` 会在已裁到 eval_start 的
    daily 上求值：预热裁剪与预热门（`warmup_bars`）双双失效，段首滚动算子用截断窗口
    出噪声值（`operators._MIN = 3` 不产 NaN）灌回 train IC——正是本任务要根除的 bug。
    """
    daily = _mock_daily__nodes_gen()
    bundle = DataBundle.build(daily)
    raw = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    llm = FakeLLM([raw, sem])
    state = AgentState(seed=1)
    state = node_generate(state, llm, daily=daily, bundle=bundle)

    with pytest.raises(ValueError, match="warmup_daily"):
        node_evaluate(state, daily=daily, bundle=bundle,
                      eval_start=dt.date(2022, 2, 1), warmup_daily=None)


def test_node_generate_rejects_illegal_and_records_error():
    daily = _mock_daily__nodes_gen()
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
    daily = _mock_daily__nodes_gen()
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

# ==== 来自 test_agent_structured.py ====
def test_propose_structured_returns_four_fields():
    def fake(_m):
        return json.dumps({"hypotheses": [{"direction": "高动量", "mechanism": "趋势延续",
                                           "expected_sign": 1, "falsification": "IC<0则证伪"}]})
    out = propose_structured(fake, known_invalid=[], known_valid=[])
    assert len(out) == 1
    for k in ["direction", "mechanism", "expected_sign", "falsification"]:
        assert k in out[0]
    assert out[0]["expected_sign"] == 1


def test_propose_structured_skips_malformed():
    def fake(_m):
        return json.dumps({"hypotheses": [{"no_direction": "x"}, {"direction": "ok"}]})
    out = propose_structured(fake, known_invalid=[], known_valid=[])
    assert len(out) == 1 and out[0]["direction"] == "ok"


def test_format_structured_renders_fields():
    h = {"direction": "高动量", "mechanism": "趋势延续", "expected_sign": 1, "falsification": "IC<0"}
    txt = format_structured(h)
    assert "高动量" in txt and "机制" in txt and "证伪" in txt


def test_decompose_tasks_returns_tasks():
    def fake(_m):
        return json.dumps({"tasks": [{"name": "mom20", "description": "20日动量", "rationale": "趋势"}]})
    tasks = decompose_tasks("高动量", fake)
    assert len(tasks) == 1
    assert tasks[0]["name"] == "mom20" and tasks[0]["rationale"] == "趋势"


def _mock_daily__structured():
    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(20)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def test_team_structured_opt_in_closes_loop(tmp_path):
    """structured=True 走结构化假设路径，闭环完成不崩。"""
    from factorzen.agents.team_orchestrator import run_team_agent
    seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                       "expected_sign": 1, "falsification": "f"}]}),
           json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})] * 20
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    res = run_team_agent(_mock_daily__structured(), fn, n_rounds=1, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), structured=True)
    assert res.state.iteration == 1

# ==== 来自 test_agent_self_heal.py ====
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
