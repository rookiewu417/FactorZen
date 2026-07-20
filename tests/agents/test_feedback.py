"""
test_memory_feedback.py：合并自 agents 相关碎片测试（test_memory_feedback.py）。
test_feedback_prompt_exhausted.py：W3 B/C/D: lift 拒绝 prompt 注入、exhausted 硬过滤、library 族聚类。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.memory import family_groups, negative_recall
from factorzen.agents.orchestrator import _summarize_feedback
from factorzen.agents.roles.critic import critique
from factorzen.agents.roles.librarian import recall
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIBRARY_CORRELATED,
    REJECT_CATEGORY_LIFT_REJECTED,
)
from factorzen.llm.prompt_fragments import (
    format_library_covered,
    format_library_crowded,
    format_lift_rejected,
)


# ==== 来自 test_memory_feedback.py ====
# ==== 来自 test_agent_memory.py ====
def test_negative_recall_and_family_suite():
    """test_negative_recall_picks_low_ic；test_negative_recall_empty_when_all_good；test_family_groups_union_find；test_family_groups_all_singletons_when_low_corr"""
    # -- 原 test_negative_recall_picks_low_ic --
    def _section_0_test_negative_recall_picks_low_ic():
        seen = [("rank(close)", 0.001), ("ts_mean(vol,5)", 0.08), ("div(close,open)", -0.002)]
        neg = negative_recall(seen, k=2, ic_threshold=0.01)
        # 只召回 IC < 阈值的，按 |IC| 升序（最没用的优先），不含高 IC 的
        assert "ts_mean(vol,5)" not in neg
        assert "rank(close)" in neg
        assert len(neg) <= 2

    _section_0_test_negative_recall_picks_low_ic()

    # -- 原 test_negative_recall_empty_when_all_good --
    def _section_1_test_negative_recall_empty_when_all_good():
        seen = [("a", 0.1), ("b", 0.2)]
        assert negative_recall(seen, ic_threshold=0.01) == []

    _section_1_test_negative_recall_empty_when_all_good()

    # -- 原 test_family_groups_union_find --
    def _section_2_test_family_groups_union_find():
        names = ["f1", "f2", "f3", "f4"]
        # f1-f2 高相关, f3-f4 高相关, 两组互不相关
        pairs = {("f1", "f2"): 0.9, ("f1", "f3"): 0.1, ("f3", "f4"): 0.85, ("f2", "f3"): 0.2}
        groups = family_groups(pairs, names, threshold=0.7)
        # 应分成两族 {f1,f2} 和 {f3,f4}
        assert {frozenset(g) for g in groups} == {frozenset({"f1", "f2"}), frozenset({"f3", "f4"})}

    _section_2_test_family_groups_union_find()

    # -- 原 test_family_groups_all_singletons_when_low_corr --
    def _section_3_test_family_groups_all_singletons_when_low_corr():
        names = ["a", "b", "c"]
        pairs = {("a", "b"): 0.1, ("b", "c"): 0.2, ("a", "c"): 0.05}
        groups = family_groups(pairs, names, threshold=0.7)
        assert len(groups) == 3  # 全独立

    _section_3_test_family_groups_all_singletons_when_low_corr()


# ==== 来自 test_agent_feedback.py ====
def _rec(it: int, expr: str, ic: float | None, *, passed: bool = False) -> AttemptRecord:
    return AttemptRecord(iteration=it, hypothesis="h", expression=expr, compile_ok=ic is not None,
                         ic_train=ic, passed_guardrails=passed, critic_verdict=None, error=None)


def test_summarize_feedback_best_suite():
    """同一轮内有多条 attempt 时，报 |IC| 最大的那条，而不是最后追加的那条。；反向因子同样有效：|IC| 最大者胜，负 IC 不被歧视。；上一轮无可评估 attempt 时，不得回退去报更早轮次的结果。；上一轮全部编译失败（ic_train=None）→ 反馈说明情况，绝不出现字面量 None。；good 故意放在前面：取 `[-1]` 会拿到 ic=None 的 bad，取「最佳有效」才拿到 good。；回归守卫：首轮无历史 → 空反馈（既有行为，不得改变）。"""
    # -- 原 test_reports_best_of_last_round_not_last_appended --
    def _section_0_test_reports_best_of_last_round_not_last_appended():
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

    _section_0_test_reports_best_of_last_round_not_last_appended()

    # -- 原 test_best_is_by_absolute_ic_negative_wins --
    def _section_1_test_best_is_by_absolute_ic_negative_wins():
        state = AgentState(seed=1)
        state.attempts += [_rec(1, "neg", -0.08), _rec(1, "pos", 0.03)]
        state.iteration = 2

        fb = _summarize_feedback(state)

        assert "neg" in fb and "-0.08" in fb
        assert "pos" not in fb

    _section_1_test_best_is_by_absolute_ic_negative_wins()

    # -- 原 test_does_not_fall_back_to_earlier_rounds --
    def _section_2_test_does_not_fall_back_to_earlier_rounds():
        state = AgentState(seed=1)
        state.attempts += [_rec(0, "stale_expr", 0.42)]     # 只有第 0 轮有结果
        state.iteration = 2                                  # 上一轮(=1)什么都没产出

        fb = _summarize_feedback(state)

        assert "stale_expr" not in fb
        assert "0.42" not in fb

    _section_2_test_does_not_fall_back_to_earlier_rounds()

    # -- 原 test_none_ic_never_rendered_into_prompt --
    def _section_3_test_none_ic_never_rendered_into_prompt():
        state = AgentState(seed=1)
        state.attempts += [_rec(1, "bad_a", None), _rec(1, "bad_b", None)]
        state.iteration = 2

        fb = _summarize_feedback(state)

        assert "None" not in fb
        assert fb != ""                    # 必须给 LLM 一个可用信号，而不是静默空串

    _section_3_test_none_ic_never_rendered_into_prompt()

    # -- 原 test_none_ic_mixed_with_valid_picks_the_valid_one --
    def _section_4_test_none_ic_mixed_with_valid_picks_the_valid_one():
        state = AgentState(seed=1)
        state.attempts += [_rec(1, "good", 0.02), _rec(1, "bad", None)]
        state.iteration = 2

        fb = _summarize_feedback(state)

        assert "good" in fb and "None" not in fb

    _section_4_test_none_ic_mixed_with_valid_picks_the_valid_one()

    # -- 原 test_empty_attempts_returns_empty_string --
    def _section_5_test_empty_attempts_returns_empty_string():
        assert _summarize_feedback(AgentState(seed=1)) == ""

    _section_5_test_empty_attempts_returns_empty_string()


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


def test_critique_residual_inject_suite():
    """test_critique_injects_residual_and_library_corr；字段缺失时不出现残差/库相关行（None-gating）。；仅 residual_ic_train 有值时仍注入残差行，无库相关行。"""
    # -- 原 test_critique_injects_residual_and_library_corr --
    def _section_0_test_critique_injects_residual_and_library_corr():
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

    _section_0_test_critique_injects_residual_and_library_corr()

    # -- 原 test_critique_omits_residual_when_missing --
    def _section_1_test_critique_omits_residual_when_missing():
        llm = FakeLLM()
        critique(_base_cand(), llm)
        blob = "\n".join(m["content"] for m in llm.calls[0])
        assert "对库残差IC" not in blob
        assert "与库最大相关" not in blob

    _section_1_test_critique_omits_residual_when_missing()

    # -- 原 test_critique_partial_residual_only --
    def _section_2_test_critique_partial_residual_only():
        llm = FakeLLM()
        critique(_base_cand(residual_ic_train=0.02), llm)
        blob = "\n".join(m["content"] for m in llm.calls[0])
        assert "对库残差IC(train/holdout)" in blob
        assert "0.02" in blob
        assert "与库最大相关" not in blob

    _section_2_test_critique_partial_residual_only()


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


def test_lift_reject_index_suite(tmp_path):
    """test_known_lift_rejects_recalls_and_scopes；test_known_lift_rejects_last_wins；lift_rejected 不进 known_invalid，进 known_lift_rejects。；lift_rejected 仍计 n_exprs（compile_ok=True）。"""
    # -- 原 test_known_lift_rejects_recalls_and_scopes --
    def _section_0_test_known_lift_rejects_recalls_and_scopes(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_known_lift_rejects_recalls_and_scopes(_tp0)

    # -- 原 test_known_lift_rejects_last_wins --
    def _section_1_test_known_lift_rejects_last_wins(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        idx.append([
            _lift_reject("rank(vol)", reason="group_gate_fail", lift=None, ts="2026-01-01T00:00:00"),
            _lift_reject("rank(vol)", reason="below_bar", lift=0.0004, ts="2026-01-02T00:00:00"),
        ])
        out = idx.known_lift_rejects(k=5, data_window=_DW)
        assert len(out) == 1
        assert out[0]["lift_reason"] == "below_bar"
        assert out[0]["lift"] == 0.0004

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_known_lift_rejects_last_wins(_tp1)

    # -- 原 test_known_invalid_excludes_lift_rejected --
    def _section_2_test_known_invalid_excludes_lift_rejected(tmp_path):
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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_known_invalid_excludes_lift_rejected(_tp2)

    # -- 原 test_leaf_stats_counts_lift_rejected --
    def _section_3_test_leaf_stats_counts_lift_rejected(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        idx.append([_lift_reject("ts_mean(holder_num_chg, 5)", ic_train=0.01)])
        stats = idx.leaf_stats(["holder_num_chg"], data_window=_DW)
        assert stats["holder_num_chg"]["n_exprs"] == 1
        assert stats["holder_num_chg"]["n_passed"] == 0

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_leaf_stats_counts_lift_rejected(_tp3)


# ==== 来自 test_feedback_prompt_exhausted.py ====
# ── B1 fragments ─────────────────────────────────────────────────────────────


def test_lift_reject_prompt_format_suite():
    """test_format_lift_rejected_text；test_format_library_crowded_text；test_propose_structured_injects_lift_rejected；test_propose_hypotheses_lift_rejected_none_zero_regression；test_critic_optional_lift_rejected"""
    # -- 原 test_format_lift_rejected_text --
    def _section_0_test_format_lift_rejected_text():
        text = format_lift_rejected([
            {"expression": "rank(vol)", "lift": 0.0005, "lift_reason": "below_bar"},
            {"expression": "ts_mean(close,5)", "lift": None, "lift_reason": "group_gate_fail"},
        ])
        assert "组合层证明" in text or "无增量" in text
        assert "rank(vol)" in text
        assert "组合增量不足" in text
        assert "组门整体无增量" in text
        assert "lift=" in text

    _section_0_test_format_lift_rejected_text()

    # -- 原 test_format_library_crowded_text --
    def _section_1_test_format_library_crowded_text():
        text = format_library_crowded([("holder_num_chg", 9), ("roe", 7)])
        assert "拥挤" in text
        assert "holder_num_chg(9)" in text
        assert "roe(7)" in text

    _section_1_test_format_library_crowded_text()

    # -- 原 test_propose_structured_injects_lift_rejected --
    def _section_2_test_propose_structured_injects_lift_rejected():
        from factorzen.agents.roles.hypothesis import propose_structured

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, messages):
                self.calls.append(messages)
                return json.dumps({
                    "hypotheses": [{
                        "direction": "x", "mechanism": "m",
                        "expected_sign": 1, "falsification": "f",
                    }],
                })

        llm = FakeLLM()
        propose_structured(
            llm,
            known_invalid=[], known_valid=[], n=1,
            lift_rejected=[{"expression": "rank(vol)", "lift": 0.0001, "lift_reason": "below_bar"}],
        )
        blob = " ".join(m["content"] for m in llm.calls[0])
        assert "rank(vol)" in blob
        assert "组合" in blob or "lift" in blob.lower() or "增量" in blob

    _section_2_test_propose_structured_injects_lift_rejected()

    # -- 原 test_propose_hypotheses_lift_rejected_none_zero_regression --
    def _section_3_test_propose_hypotheses_lift_rejected_none_zero_regression():
        from factorzen.agents.roles.hypothesis import propose_hypotheses

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, messages):
                self.calls.append(messages)
                return json.dumps({"hypotheses": ["dir"]})

        llm = FakeLLM()
        propose_hypotheses(llm, known_invalid=[], known_valid=[], n=1, lift_rejected=None)
        blob = " ".join(m["content"] for m in llm.calls[0])
        assert "组合层" not in blob and "lift 拒绝" not in blob

    _section_3_test_propose_hypotheses_lift_rejected_none_zero_regression()

    # -- 原 test_critic_optional_lift_rejected --
    def _section_4_test_critic_optional_lift_rejected():
        from factorzen.agents.roles.critic import critique

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, messages):
                self.calls.append(messages)
                return json.dumps({"verdict": "keep", "reason": "ok"})

        llm = FakeLLM()
        critique(
            {"expression": "rank(close)", "ic_train": 0.02},
            llm,
            lift_rejected=[{"expression": "rank(vol)", "lift": 0.0, "lift_reason": "below_bar"}],
        )
        blob = " ".join(m["content"] for m in llm.calls[0])
        assert "rank(vol)" in blob or "lift" in blob.lower() or "组合" in blob

        llm2 = FakeLLM()
        critique({"expression": "rank(close)"}, llm2)  # 默认 None
        blob2 = " ".join(m["content"] for m in llm2.calls[0])
        assert "组合层" not in blob2

    _section_4_test_critic_optional_lift_rejected()


# ── B2 hypothesis / critic ───────────────────────────────────────────────────


# ── B3 librarian ─────────────────────────────────────────────────────────────


def test_recall_fills_lift_rejected(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    dw = {"start": "20200101", "end": "20201231", "universe": "csi300", "market": "ashare"}
    idx.append([{
        "expression": "rank(vol)",
        "data_window": dw,
        "reject_category": REJECT_CATEGORY_LIFT_REJECTED,
        "passed": False,
        "compile_ok": True,
        "lift": 0.0001,
        "lift_reason": "below_bar",
        "ts": "2026-01-01T00:00:00",
    }])
    r = recall(idx, k=5, data_window=dw)
    assert r.lift_rejected is not None
    assert any(x["expression"] == "rank(vol)" for x in r.lift_rejected)

    r2 = recall(idx, k=5, data_window={
        "start": "20990101", "end": "20991231", "universe": "x", "market": "ashare",
    })
    assert r2.lift_rejected is None  # 空 → None


def test_recall_exhausted_leaves_raw_names(tmp_path: Path, monkeypatch):
    """C1: RecallResult 带原始 exhausted 叶名，非格式化字符串。"""
    from factorzen.agents.roles import librarian as lib_mod

    monkeypatch.setattr(lib_mod, "EXHAUSTED_MIN_TRIES", 2)
    dw = {"start": "20200101", "end": "20201231", "universe": "csi300", "market": "ashare"}
    idx = ExperimentIndex(str(tmp_path / "e2.jsonl"))
    for i, expr in enumerate([
        "rank(holder_num_chg)",
        "ts_mean(holder_num_chg, 5)",
        "ts_mean(holder_num_chg, 10)",
    ]):
        idx.append([{
            "expression": expr,
            "data_window": dw,
            "passed": False,
            "compile_ok": True,
            "ic_train": 0.01 * (i + 1),
        }])
    r = recall(idx, k=5, data_window=dw, leaf_names=["holder_num_chg", "roe"])
    assert r.exhausted_leaves is not None
    assert "holder_num_chg" in r.exhausted_leaves
    # 格式化文案仍在 leaf_guidance
    assert r.leaf_guidance is not None
    assert any("holder_num_chg" in s for s in (r.leaf_guidance.get("exhausted") or []))


# ── C filter ─────────────────────────────────────────────────────────────────


def test_filter_exhausted_suite():
    """test_filter_exhausted_all_exhausted_drop；test_filter_exhausted_mixed_family_quota；test_filter_exhausted_parse_fail_keep；test_filter_exhausted_none_passthrough"""
    # -- 原 test_filter_exhausted_all_exhausted_drop --
    def _section_0_test_filter_exhausted_all_exhausted_drop():
        from factorzen.agents.scout_support import filter_exhausted_expressions

        kept, n_drop = filter_exhausted_expressions(
            ["rank(holder_num_chg)", "ts_mean(holder_num_chg, 5)"],
            exhausted={"holder_num_chg"},
            leaf_map=None,
            quota_used={},
            per_leaf_quota=2,
        )
        assert kept == []
        assert n_drop == 2

    _section_0_test_filter_exhausted_all_exhausted_drop()

    # -- 原 test_filter_exhausted_mixed_family_quota --
    def _section_1_test_filter_exhausted_mixed_family_quota():
        from factorzen.agents.scout_support import filter_exhausted_expressions

        quota: dict[str, int] = {}
        # 混族：含 exhausted 叶 + 非 exhausted 叶 → 配额内放行
        kept, n_drop = filter_exhausted_expressions(
            ["div(rank(holder_num_chg), rank(roe))"],
            exhausted={"holder_num_chg"},
            leaf_map=None,
            quota_used=quota,
            per_leaf_quota=2,
        )
        assert kept == ["div(rank(holder_num_chg), rank(roe))"]
        assert n_drop == 0
        assert quota.get("holder_num_chg") == 1

        # 再两条后配额满
        kept2, n2 = filter_exhausted_expressions(
            [
                "div(ts_mean(holder_num_chg, 5), rank(close))",
                "div(ts_mean(holder_num_chg, 10), rank(open))",
            ],
            exhausted={"holder_num_chg"},
            leaf_map=None,
            quota_used=quota,
            per_leaf_quota=2,
        )
        assert n2 == 1  # 第 3 条超配额
        assert len(kept2) == 1
        assert quota["holder_num_chg"] == 2

    _section_1_test_filter_exhausted_mixed_family_quota()

    # -- 原 test_filter_exhausted_parse_fail_keep --
    def _section_2_test_filter_exhausted_parse_fail_keep():
        from factorzen.agents.scout_support import filter_exhausted_expressions

        kept, n_drop = filter_exhausted_expressions(
            ["this_is_not(valid"],
            exhausted={"holder_num_chg"},
            leaf_map=None,
            quota_used={},
        )
        assert kept == ["this_is_not(valid"]
        assert n_drop == 0

    _section_2_test_filter_exhausted_parse_fail_keep()

    # -- 原 test_filter_exhausted_none_passthrough --
    def _section_3_test_filter_exhausted_none_passthrough():
        from factorzen.agents.scout_support import filter_exhausted_expressions

        exprs = ["rank(vol)", "rank(close)"]
        kept, n = filter_exhausted_expressions(
            exprs, exhausted=None, leaf_map=None, quota_used={},
        )
        assert kept == exprs and n == 0
        kept2, n2 = filter_exhausted_expressions(
            exprs, exhausted=set(), leaf_map=None, quota_used={},
        )
        assert kept2 == exprs and n2 == 0

    _section_3_test_filter_exhausted_none_passthrough()


# ── D library family ─────────────────────────────────────────────────────────


def test_library_covered_family_suite(tmp_path, monkeypatch):
    """test_library_covered_by_family；旧 fragment 零回归。；轮内接线：exhausted 非空 → rounds_log 有 n_exhausted_filtered；None 直通。；双路径：M5 外层 scripted-llm 行为测试——library_crowded 进 prompt（非仅 inspect）。"""
    # -- 原 test_library_covered_by_family --
    def _section_0_test_library_covered_by_family(tmp_path):
        from factorzen.discovery.factor_library import (
            FactorRecord,
            library_covered_by_family,
        )

        root = str(tmp_path / "lib")
        recs = [
            FactorRecord(expression="rank(holder_num_chg)", market="ashare",
                         ic_train=0.05, status="active"),
            FactorRecord(expression="ts_mean(holder_num_chg, 5)", market="ashare",
                         ic_train=0.04, status="active"),
            FactorRecord(expression="ts_mean(holder_num_chg, 10)", market="ashare",
                         ic_train=0.03, status="active"),
            FactorRecord(expression="rank(roe)", market="ashare",
                         ic_train=0.06, status="active"),
            FactorRecord(expression="ts_mean(roe, 5)", market="ashare",
                         ic_train=0.02, status="active"),
            FactorRecord(expression="rank(close)", market="ashare",
                         ic_train=0.01, status="active"),
        ]
        # write via save if available, else raw jsonl
        path = Path(root) / "ashare.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in recs) + "\n",
            encoding="utf-8",
        )
        covered, crowded = library_covered_by_family(
            "ashare", per_family=2, max_total=12, crowded_min=3, root=root,
        )
        # holder 族 3 条只留 2；roe 2；close 1
        assert len(covered) <= 5
        # 同叶集 holder 最多 2
        holder_exprs = [e for e in covered if "holder_num_chg" in e]
        assert len(holder_exprs) == 2
        # 最佳 |ic| 的 rank(holder) 应在
        assert any("rank(holder_num_chg)" in e for e in holder_exprs)
        # crowded: holder 出现 3 次 ≥3
        crowded_map = dict(crowded)
        assert crowded_map.get("holder_num_chg") == 3
        # roe 只 2 次 < crowded_min=3 → 不进
        assert "roe" not in crowded_map

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_library_covered_by_family(_tp0)

    # -- 原 test_format_library_covered_unchanged --
    def _section_1_test_format_library_covered_unchanged():
        assert format_library_covered(None) == ""
        assert format_library_covered(["a", "b"]) == "库内已有(追求与其正交,换方向): a；b"

    _section_1_test_format_library_covered_unchanged()

    # -- 原 test_round_exhausted_filter_in_rounds_log --
    def _section_2_test_round_exhausted_filter_in_rounds_log(tmp_path, mp):
        import datetime as dt

        import numpy as np
        import polars as pl

        from factorzen.agents.roles import librarian as lib_mod
        from factorzen.agents.team_orchestrator import run_team_agent

        mp.setattr(lib_mod, "EXHAUSTED_MIN_TRIES", 2)

        # 预置 index：holder_num_chg 挖穿
        idx_path = tmp_path / "experiment_index.jsonl"
        idx = ExperimentIndex(str(idx_path))
        dw = {"start": "20220101", "end": "20220630", "universe": "csi300", "market": "ashare"}
        for expr in [
            "rank(holder_num_chg)",
            "ts_mean(holder_num_chg, 5)",
            "ts_mean(holder_num_chg, 10)",
        ]:
            idx.append([{
                "expression": expr, "data_window": dw,
                "passed": False, "compile_ok": True, "ic_train": 0.01,
            }])

        # 极简日频帧
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < 80:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        rows = []
        rng = np.random.default_rng(0)
        for c in [f"{i:06d}.SZ" for i in range(20)]:
            px = 10.0
            for dd in days:
                px *= 1 + rng.standard_normal() * 0.01
                rows.append({
                    "trade_date": dd, "ts_code": c, "close": px,
                    "open": px, "high": px, "low": px,
                    "vol": 1e6, "amount": 1e7,
                })
        daily = pl.DataFrame(rows)

        # scripted：propose → write 产出纯 exhausted 表达式（与 test_team_lift_hook 同款分支）
        def llm_fn(messages):
            text = "\n".join(m["content"] for m in messages)
            if "风控审计员" in text:
                return json.dumps({"verdict": "keep", "reason": "ok"})
            if "翻译成" in text:
                return json.dumps({
                    "expressions": [
                        "rank(holder_num_chg)",
                        "ts_mean(holder_num_chg, 20)",
                    ],
                })
            return json.dumps({"hypotheses": ["筹码集中"]})

        result = run_team_agent(
            daily, llm_fn,
            n_rounds=1, seed=1, top_k=3,
            index_path=str(idx_path),
            data_window=dw,
            library_orthogonal=False,
            auto_lift=False,
            heal_rounds=0,
        )
        assert result.rounds_log
        assert "n_exhausted_filtered" in result.rounds_log[0]
        # 两条纯 exhausted 应被过滤
        assert result.rounds_log[0]["n_exhausted_filtered"] >= 1

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_round_exhausted_filter_in_rounds_log(_tp2, mp)

    # -- 原 test_m5_library_crowded_injected_via_orchestrator --
    def _section_3_test_m5_library_crowded_injected_via_orchestrator(tmp_path):
        import datetime as dt

        import numpy as np
        import polars as pl

        from factorzen.agents.orchestrator import run_llm_agent
        from factorzen.discovery.factor_library import FactorRecord

        root = tmp_path / "lib"
        root.mkdir()
        recs = []
        for i in range(4):
            # 同叶 close 变体 4 条 → crowded
            recs.append(FactorRecord(
                expression=f"ts_mean(close, {5 + i * 5})",
                market="ashare", ic_train=0.05 - i * 0.005, status="active",
            ))
        (root / "ashare.jsonl").write_text(
            "\n".join(json.dumps(r.__dict__, default=str) for r in recs) + "\n",
            encoding="utf-8",
        )

        days, d = [], dt.date(2022, 1, 3)
        while len(days) < 60:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        rows = []
        rng = np.random.default_rng(1)
        for c in [f"{i:06d}.SZ" for i in range(15)]:
            px = 10.0
            for dd in days:
                px *= 1 + rng.standard_normal() * 0.01
                rows.append({
                    "trade_date": dd, "ts_code": c, "close": px,
                    "open": px, "high": px, "low": px, "vol": 1e6, "amount": 1e7,
                })
        daily = pl.DataFrame(rows)

        captured: list[str] = []

        def llm_fn(messages):
            blob = "\n".join(m["content"] for m in messages)
            captured.append(blob)
            if "风控" in blob or "审计" in blob:
                return json.dumps({"verdict": "keep", "reason": "ok"})
            return json.dumps({
                "hypothesis": "动量",
                "expressions": ["rank(close)"],
                "rationale": "x",
            })

        run_llm_agent(
            daily, llm_fn,
            n_rounds=1, seed=1, top_k=2,
            library_orthogonal=True,
            library_root=str(root),
            heal_rounds=0,
        )
        all_text = "\n".join(captured)
        # 拥挤叶子文案应出现在生成 prompt
        assert "拥挤" in all_text or "close(" in all_text

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_m5_library_crowded_injected_via_orchestrator(_tp3)


# ── C wiring + D M5 dual-path behavioral ─────────────────────────────────────


