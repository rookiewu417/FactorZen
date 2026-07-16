# tests/test_exception_and_accounting_hygiene.py
"""异常契约 + 记账卫生：静默吞异常、静默丢记录、静默多记 N。

独立但同源的缺陷——**都在「悄悄发生、看不出来」这条线上**：

1. `node_guardrails` / `evaluate_expressions` 的裸 `except Exception: continue`
   吞掉一切，无日志无计数。一个候选的 holdout 求值炸了，你永远不会知道。
2. `node_critic` 用裸 `json.loads` 而非容错的 `_extract_json`（`roles/critic.py` 用后者）。
   `request_chat` 显式关掉 json_object 模式且不剥围栏，LLM 加 markdown 围栏时解析必抛，
   被 `except Exception` 降级为 `"keep"`。双路径漂移。
3. `librarian.record()` 跳过编译失败的 attempt → 坏表达式**永不进长期记忆** →
   `seen_expressions()` 看不到它们 → 跨 session 反复生成同一语法坑，白烧 LLM 调用与自愈轮次。
4. team 的 `_evaluate_and_record` 的 `fresh` 列表**内部不去重**。`heal_rounds=0` 时（heal 的
   去重不生效）多个 task 翻译出同一表达式 → 评估两次 → N over-count。
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import polars as pl
import pytest


def _daily(n_stocks: int = 40, n_days: int = 200, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99, "vol": 1e6, "amount": 1e7})
    return pl.DataFrame(rows)


# ── 1. 静默吞异常 ───────────────────────────────────────────────────────────


def test_node_guardrails_logs_when_a_candidate_blows_up(caplog):
    """一个候选的 holdout 求值失败必须留下日志，而不是静默 continue。

    静默 continue 会让「这个候选炸了」与「这个候选没过护栏」在产物上不可区分。
    """
    import factorzen.validation.holdout as hmod
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _daily()
    bundle = DataBundle.build(daily)
    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="rank(close)", compile_ok=True,
        ic_train=0.05, passed_guardrails=False, critic_verdict=None, error=None,
        ir_train=0.4, n_train=150))

    orig = hmod.holdout_ic_result

    def boom(*_a, **_kw):
        raise RuntimeError("holdout 求值失败")

    hmod.holdout_ic_result = boom
    try:
        with caplog.at_level(logging.WARNING, logger="factorzen.agents.nodes"):
            node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                            ledger=TrialLedger(), top_k=5, warmup_daily=daily)
    finally:
        hmod.holdout_ic_result = orig

    assert not state.candidates
    assert any("rank(close)" in r.getMessage() for r in caplog.records), \
        "候选护栏计算失败必须记日志（含表达式），不得静默吞掉"


# ── 2. node_critic 的双路径漂移 ─────────────────────────────────────────────


def test_node_critic_parses_fenced_json_like_team_critic_does():
    """真实 LLM 常返回 markdown 围栏。单 Agent 与 team 的 Critic 必须给出同一裁决。"""
    from factorzen.agents.nodes import node_critic
    from factorzen.agents.roles.critic import critique
    from factorzen.agents.state import AgentState, AttemptRecord

    fenced = '```json\n{"verdict": "drop", "reason": "过拟合"}\n```'

    def llm(_msgs):
        return fenced

    state = AgentState(seed=0)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="close", compile_ok=True, ic_train=0.5,
        passed_guardrails=True, critic_verdict=None, error=None, ir_train=0.3))
    node_critic(state, llm)

    assert state.attempts[0].critic_verdict == critique({"expression": "close"}, llm).verdict
    assert state.attempts[0].critic_verdict == "drop", "LLM 说 drop，不该被静默降级为 keep"


def test_node_critic_still_defaults_to_keep_on_garbage():
    """完全无法解析时仍 fail-open（不误杀），但那是**解析失败**，不是格式差异。"""
    from factorzen.agents.nodes import node_critic
    from factorzen.agents.state import AgentState, AttemptRecord

    state = AgentState(seed=0)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="close", compile_ok=True, ic_train=0.5,
        passed_guardrails=True, critic_verdict=None, error=None, ir_train=0.3))
    node_critic(state, lambda _m: "彻底不是 JSON")

    assert state.attempts[0].critic_verdict == "keep"


def test_json_loads_would_have_raised_on_fenced_input():
    """判别性前置：证明围栏输入确实会让裸 json.loads 抛——否则上面的修复无从验证。"""
    with pytest.raises(json.JSONDecodeError):
        json.loads('```json\n{"verdict": "drop"}\n```')


# ── 3. 编译失败的表达式必须进长期记忆 ───────────────────────────────────────


def test_record_persists_compile_failures_so_they_are_not_retried(tmp_path):
    """坏表达式不进 index → `seen_expressions()` 看不到 → 跨 session 反复生成同一语法坑。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import record
    from factorzen.agents.state import AttemptRecord

    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    bad = AttemptRecord(iteration=0, hypothesis="h", expression="ts_mean(close)",
                        compile_ok=False, ic_train=None, passed_guardrails=False,
                        critic_verdict=None, error="缺少窗口参数", ir_train=None)
    good = AttemptRecord(iteration=0, hypothesis="h", expression="rank(close)",
                         compile_ok=True, ic_train=0.02, passed_guardrails=False,
                         critic_verdict="keep", error=None, ir_train=0.1, n_train=200)
    record(idx, [bad, good], run_id="r1")

    assert idx.seen_expressions() == {"ts_mean(close)", "rank(close)"}, \
        "编译失败的表达式必须进 seen，否则跨 session 会重复生成"
    stored = {r["expression"]: r for r in idx.load()}
    assert stored["ts_mean(close)"]["compile_ok"] is False
    assert stored["ts_mean(close)"]["error"] == "缺少窗口参数"


def test_known_invalid_excludes_compile_failures(tmp_path):
    """`known_invalid` 的语义是「能编译但无效」。语法坑 ic_train=None → 排序键 0.0 会排最前，
    把有信息的低 IC 负例全部挤出 top-k。它们的价值在 seen 去重，不在负例库。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import record
    from factorzen.agents.state import AttemptRecord

    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [
        AttemptRecord(0, "h", "bad_syntax", False, None, False, None, "boom", None),
        AttemptRecord(0, "h", "low_ic", True, 0.001, False, None, None, 0.01, n_train=200),
    ], run_id="r1")

    assert idx.known_invalid(k=5) == ["low_ic"], "语法坑不该占据「已验证无效」负例库"


def test_compile_failures_never_enter_the_deflation_pool(tmp_path):
    """编译失败记录的 ir_train=None，必须被 DeflationBasis 剔除——否则污染 N 与经验方差。"""
    from factorzen.discovery.guardrails import DeflationBasis

    basis = DeflationBasis.from_ir_pool([0.3, None, 0.1])
    assert basis.n_trials == 2


# ── 4. 同轮重复表达式的 N over-count ────────────────────────────────────────


def test_team_evaluate_deduplicates_within_the_batch():
    """`heal_rounds=0` 时 heal 的去重不生效；多个 task 翻译出同一表达式 → 不得评估两次。

    N 是多重检验的记账，多算一次就是记账不诚实（方向偏严，但仍是错的）。
    """
    from factorzen.agents.state import AgentState
    from factorzen.agents.team_orchestrator import _evaluate_and_record
    from factorzen.discovery.scoring import DataBundle

    daily = _daily(n_days=150, seed=7)
    bundle = DataBundle.build(daily)
    state = AgentState(seed=0)

    exprs = ["ts_mean(close, 5)", "ts_mean(close,5)", "rank(close)"]  # 前两个归一化后同一个
    results = _evaluate_and_record(state, exprs, "h", daily=daily, bundle=bundle, mem_seen=set())

    assert len(results) == 2, f"3 个输入含 1 个重复 → 只该评估 2 次，实得 {len(results)}"
    assert len(state.attempts) == 2
    uniq = {a.expression for a in state.attempts}
    assert len(uniq) == 2

    passed = [a for a in state.attempts if a.compile_ok and a.ic_train is not None]
    assert len(passed) == len(uniq), "node_guardrails 记的 N 必须等于唯一表达式数"


def test_team_evaluate_normalizes_before_dedup():
    """去重必须在**归一化后**比较：`ts_mean(close,5)` 与 `ts_mean(close, 5)` 是同一表达式。"""
    from factorzen.agents.team_orchestrator import _normalize

    assert _normalize("ts_mean(close,5)") == _normalize("ts_mean(close, 5)")


@pytest.mark.parametrize("garbage", ["", "{", "不是json", "[1,2]"])
def test_extract_json_never_raises_on_string_garbage(garbage):
    """回归守卫：`_extract_json` 对任何字符串都不抛（只返回 None 或 dict）。"""
    from factorzen.llm.generation import _extract_json

    out = _extract_json(garbage)
    assert out is None or isinstance(out, dict)
