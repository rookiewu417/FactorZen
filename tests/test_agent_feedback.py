# tests/test_agent_feedback.py
"""P1①：_summarize_feedback 必须真的报「上一轮的最佳」。

原实现 `last = state.attempts[-1]` 取的是**最后追加**的那条 attempt：
  - 它不是最佳（同轮里 IC 更高的会被后来者覆盖）；
  - 上一轮若无新 attempt（表达式全被去重/语义检查拦下），它属于**更早的轮次**；
  - 它的 ic_train 可能是 None（编译失败），会渲染成「上轮最佳 train_IC=None」喂给 LLM。
文案却写着「上轮最佳」。这是喂进下一轮 prompt 的反馈，错了会污染假设生成。

调用时机：orchestrator 里 node_reflect(iteration += 1) 之后才调 _summarize_feedback，
故「上一轮」= state.iteration - 1。
"""
from __future__ import annotations

from factorzen.agents.orchestrator import _summarize_feedback
from factorzen.agents.state import AgentState, AttemptRecord


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
