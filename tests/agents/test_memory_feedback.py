"""合并自 agents 相关碎片测试（test_memory_feedback.py）。

test_agent_memory.py：negative_recall 与 family_groups（记忆侧负样本召回/相关簇）
test_agent_feedback.py：P1①：_summarize_feedback 必须真的报「上一轮的最佳」
test_w6_critic_orthogonal.py：W6 critic prompt 注入 residual_ic / library_corr
test_w3_lift_reject_index.py：W3 A2/A3: lift_rejected 写回 experiment_index 与召回通道
"""

from __future__ import annotations

import json
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.memory import family_groups, negative_recall
from factorzen.agents.orchestrator import _summarize_feedback
from factorzen.agents.roles.critic import critique
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIBRARY_CORRELATED,
    REJECT_CATEGORY_LIFT_REJECTED,
)


# ==== 来自 test_agent_memory.py ====
def test_negative_recall_picks_low_ic():
    seen = [("rank(close)", 0.001), ("ts_mean(vol,5)", 0.08), ("div(close,open)", -0.002)]
    neg = negative_recall(seen, k=2, ic_threshold=0.01)
    # 只召回 IC < 阈值的，按 |IC| 升序（最没用的优先），不含高 IC 的
    assert "ts_mean(vol,5)" not in neg
    assert "rank(close)" in neg
    assert len(neg) <= 2


def test_negative_recall_empty_when_all_good():
    seen = [("a", 0.1), ("b", 0.2)]
    assert negative_recall(seen, ic_threshold=0.01) == []


def test_family_groups_union_find():
    names = ["f1", "f2", "f3", "f4"]
    # f1-f2 高相关, f3-f4 高相关, 两组互不相关
    pairs = {("f1", "f2"): 0.9, ("f1", "f3"): 0.1, ("f3", "f4"): 0.85, ("f2", "f3"): 0.2}
    groups = family_groups(pairs, names, threshold=0.7)
    # 应分成两族 {f1,f2} 和 {f3,f4}
    assert {frozenset(g) for g in groups} == {frozenset({"f1", "f2"}), frozenset({"f3", "f4"})}


def test_family_groups_all_singletons_when_low_corr():
    names = ["a", "b", "c"]
    pairs = {("a", "b"): 0.1, ("b", "c"): 0.2, ("a", "c"): 0.05}
    groups = family_groups(pairs, names, threshold=0.7)
    assert len(groups) == 3  # 全独立

# ==== 来自 test_agent_feedback.py ====
def _rec(it: int, expr: str, ic: float | None, *, passed: bool = False) -> AttemptRecord:
    return AttemptRecord(iteration=it, hypothesis="h", expression=expr, compile_ok=ic is not None,
                         ic_train=ic, passed_guardrails=passed, critic_verdict=None, error=None)


def test_reports_best_of_last_round_not_last_appended():
    """同一轮内有多条 attempt 时，报 |IC| 最大的那条，而不是最后追加的那条。"""
    state = AgentState(seed=1)
    state.attempts += [
        _rec(0, "old", 0.09),
        _rec(1, "best_expr", 0.03, passed=True),
        _rec(1, "worst_expr", 0.01),          # 最后追加，但不是最佳
    ]
    state.iteration = 2                        # node_reflect 已 +1 → 上一轮是 1

    fb = _summarize_feedback(state)

    assert "best_expr" in fb
    assert "worst_expr" not in fb
    assert "0.03" in fb


def test_best_is_by_absolute_ic_negative_wins():
    """反向因子同样有效：|IC| 最大者胜，负 IC 不被歧视。

    neg 故意放在前面：取 `[-1]` 会得到 pos，取 max|IC| 才得到 neg —— 两种实现答案不同，
    这条断言才有判别力。
    """
    state = AgentState(seed=1)
    state.attempts += [_rec(1, "neg", -0.08), _rec(1, "pos", 0.03)]
    state.iteration = 2

    fb = _summarize_feedback(state)

    assert "neg" in fb and "-0.08" in fb
    assert "pos" not in fb


def test_does_not_fall_back_to_earlier_rounds():
    """上一轮无可评估 attempt 时，不得回退去报更早轮次的结果。"""
    state = AgentState(seed=1)
    state.attempts += [_rec(0, "stale_expr", 0.42)]     # 只有第 0 轮有结果
    state.iteration = 2                                  # 上一轮(=1)什么都没产出

    fb = _summarize_feedback(state)

    assert "stale_expr" not in fb
    assert "0.42" not in fb


def test_none_ic_never_rendered_into_prompt():
    """上一轮全部编译失败（ic_train=None）→ 反馈说明情况，绝不出现字面量 None。"""
    state = AgentState(seed=1)
    state.attempts += [_rec(1, "bad_a", None), _rec(1, "bad_b", None)]
    state.iteration = 2

    fb = _summarize_feedback(state)

    assert "None" not in fb
    assert fb != ""                    # 必须给 LLM 一个可用信号，而不是静默空串


def test_none_ic_mixed_with_valid_picks_the_valid_one():
    """good 故意放在前面：取 `[-1]` 会拿到 ic=None 的 bad，取「最佳有效」才拿到 good。"""
    state = AgentState(seed=1)
    state.attempts += [_rec(1, "good", 0.02), _rec(1, "bad", None)]
    state.iteration = 2

    fb = _summarize_feedback(state)

    assert "good" in fb and "None" not in fb


def test_empty_attempts_returns_empty_string():
    """回归守卫：首轮无历史 → 空反馈（既有行为，不得改变）。"""
    assert _summarize_feedback(AgentState(seed=1)) == ""

# ==== 来自 test_w6_critic_orthogonal.py ====
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

# ==== 来自 test_w3_lift_reject_index.py ====
_DW = {
    "start": "20200101",
    "end": "20201231",
    "universe": "csi300",
    "market": "ashare",
}
_DW_OTHER = {
    "start": "20210101",
    "end": "20211231",
    "universe": "csi300",
    "market": "ashare",
}


def _lift_reject(
    expr: str,
    *,
    lift: float | None = 0.0005,
    reason: str = "below_bar",
    ts: str | None = "2026-01-02T00:00:00",
    data_window: dict | None = None,
    ic_train: float | None = 0.02,
) -> dict:
    rec: dict = {
        "expression": expr,
        "data_window": data_window if data_window is not None else dict(_DW),
        "reject_category": REJECT_CATEGORY_LIFT_REJECTED,
        "passed": False,
        "compile_ok": True,
        "ic_train": ic_train,
        "residual_ic_train": 0.008,
        "lift": lift,
        "lift_se": 0.001,
        "lift_reason": reason,
        "baseline_rank_ic": 0.03,
        "admission_start": "2020-10-01",
        "admission_end": "2020-12-31",
        "source": "session_auto_lift",
    }
    if ts is not None:
        rec["ts"] = ts
    return rec



def test_known_lift_rejects_recalls_and_scopes(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        _lift_reject("rank(vol)", ts="2026-01-01T00:00:00", lift=0.0001),
        _lift_reject("ts_mean(close, 5)", ts="2026-01-03T00:00:00", lift=0.0002),
        _lift_reject(
            "rank(amount)",
            ts="2026-01-04T00:00:00",
            lift=0.0003,
            data_window=_DW_OTHER,
        ),
        # 非 lift_rejected 不应召回
        {
            "expression": "rank(open)",
            "data_window": dict(_DW),
            "reject_category": REJECT_CATEGORY_LIBRARY_CORRELATED,
            "passed": False,
            "compile_ok": True,
            "ic_train": 0.01,
            "ts": "2026-01-05T00:00:00",
        },
    ])
    out = idx.known_lift_rejects(k=5, data_window=_DW)
    exprs = [r["expression"] for r in out]
    assert "ts_mean(close, 5)" in exprs or any("ts_mean" in e for e in exprs)
    assert "rank(vol)" in exprs
    # 跨窗口不召回
    assert not any("amount" in e for e in exprs)
    # 形状
    for r in out:
        assert set(r.keys()) == {"expression", "lift", "lift_reason"}
    # ts 降序：ts_mean 更新
    assert out[0]["expression"].startswith("ts_mean") or "close" in out[0]["expression"]
    assert out[0]["lift"] == 0.0002


def test_known_lift_rejects_last_wins(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        _lift_reject("rank(vol)", reason="group_gate_fail", lift=None, ts="2026-01-01T00:00:00"),
        _lift_reject("rank(vol)", reason="below_bar", lift=0.0004, ts="2026-01-02T00:00:00"),
    ])
    out = idx.known_lift_rejects(k=5, data_window=_DW)
    assert len(out) == 1
    assert out[0]["lift_reason"] == "below_bar"
    assert out[0]["lift"] == 0.0004


def test_known_invalid_excludes_lift_rejected(tmp_path: Path):
    """lift_rejected 不进 known_invalid，进 known_lift_rejects。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        _lift_reject("rank(vol)", lift=0.0001),
        {
            "expression": "rank(open)",
            "data_window": dict(_DW),
            "passed": False,
            "compile_ok": True,
            "ic_train": 0.001,
        },
    ])
    invalid = idx.known_invalid(k=10, data_window=_DW)
    lift_r = idx.known_lift_rejects(k=10, data_window=_DW)
    assert "rank(vol)" not in invalid
    assert any(r["expression"] == "rank(vol)" for r in lift_r)
    assert "rank(open)" in invalid


def test_leaf_stats_counts_lift_rejected(tmp_path: Path):
    """lift_rejected 仍计 n_exprs（compile_ok=True）。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([_lift_reject("ts_mean(holder_num_chg, 5)", ic_train=0.01)])
    stats = idx.leaf_stats(["holder_num_chg"], data_window=_DW)
    assert stats["holder_num_chg"]["n_exprs"] == 1
    assert stats["holder_num_chg"]["n_passed"] == 0
