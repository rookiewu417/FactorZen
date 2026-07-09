# tests/test_mining_loop_resilience.py
"""P1: 挖掘循环韧性 —— 一次 LLM 故障不该让整轮挖掘全损。

修复前：所有 LLM 调用点都不 catch `LLMClientError`（除 `node_critic` 的宽 except），
且 manifest 只在 `run_llm_agent` **返回之后**才落盘。于是多轮挖掘跑到第 N 轮遇一次
不可恢复的 LLM 故障 → 异常一路冒泡 → 进程崩溃 → **前 N-1 轮找到的候选全部丢失，
无 manifest、无从续跑**。在「无人值守运营」目标下，例行网络抖动即静默全损。

client 层的有限重试（PR #63）只挡得住瞬时故障；重试耗尽、或遇 422 这类不可重试错误时，
异常仍会冒泡。本文件覆盖编排层：**该轮跳过而非崩**，且**每轮增量落盘**。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from factorzen.agents.orchestrator import run_llm_agent
from factorzen.llm.client import LLMClientError


def _mock_daily(n_stocks=40, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(n_stocks)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _smart_llm(*, fail_rounds: frozenset[int] = frozenset(),
               exc: type[Exception] = LLMClientError):
    """按 system prompt 分辨调用类型的 fake LLM。

    按**调用序号**注入失败是不可靠的：每轮的 LLM 调用次数并不固定——若某轮的表达式已在
    `seen_expressions` 里，`node_generate` 会跳过 `semantic_check`，该轮只发 1 次调用。
    故这里按 proposal 轮次（0-based）注入，并让每轮产出**不同**表达式以避开去重。

    ``fail_rounds`` 指定哪几轮的 proposal 调用抛 ``exc``。
    """
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "consistent" in system:
            return json.dumps({"consistent": True, "reason": "ok"})
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        st["round"] += 1
        if st["round"] in fail_rounds:
            raise exc(f"round#{st['round']}: 上游不可用（重试已耗尽）")
        w = 4 + st["round"]          # 每轮不同窗口 → 表达式不同 → 不触发去重
        return json.dumps({"hypothesis": f"动量{w}",
                           "expressions": [f"ts_mean(close,{w})"], "rationale": "r"})
    return fn


# ── 轮层容错 ────────────────────────────────────────────────────────────────


def test_agent_loop_survives_llm_error_and_continues():
    """某轮 LLM 不可用 → 跳过该轮继续，而非冒泡崩掉整个 session。"""
    llm = _smart_llm(fail_rounds=frozenset({0}))
    res = run_llm_agent(_mock_daily(), llm, n_rounds=3, seed=42)

    assert res.state.iteration == 3, "失败轮应跳过而非崩溃，循环跑满"
    assert len(res.state.attempts) >= 1, "轮 1/2 应正常产出 attempts"


def test_agent_loop_aborts_after_consecutive_llm_failures():
    """LLM 持续不可用时提前终止，不空转跑满 n_rounds。"""
    llm = _smart_llm(fail_rounds=frozenset(range(20)))
    res = run_llm_agent(_mock_daily(), llm, n_rounds=10, seed=42, llm_failure_patience=2)

    assert res.state.iteration == 2, f"连续 2 轮失败即终止，实得 {res.state.iteration}"
    assert not res.state.attempts


def test_consecutive_failure_counter_resets_on_success():
    """失败计数器必须在成功轮重置——否则零散的抖动会被累计成「持续不可用」。"""
    llm = _smart_llm(fail_rounds=frozenset({0, 2}))   # 轮 0、2 失败；轮 1、3 成功
    res = run_llm_agent(_mock_daily(), llm, n_rounds=4, seed=42, llm_failure_patience=2)

    assert res.state.iteration == 4, "两次孤立失败不该触发 patience=2 的提前终止"


def test_non_llm_exception_still_propagates():
    """只吞 LLMClientError。别的异常（代码 bug、磁盘满）必须冒泡，不许静默吞掉。"""
    llm = _smart_llm(fail_rounds=frozenset({0}), exc=RuntimeError)
    with pytest.raises(RuntimeError):
        run_llm_agent(_mock_daily(), llm, n_rounds=3, seed=42)


# ── 增量落盘 ────────────────────────────────────────────────────────────────


def test_on_round_end_called_after_each_successful_round():
    seen: list[int] = []
    run_llm_agent(_mock_daily(), _smart_llm(), n_rounds=3, seed=42,
                  on_round_end=lambda r: seen.append(len(r.state.attempts)))

    assert len(seen) == 3, f"每轮末应回调一次，实得 {len(seen)}"
    assert seen == sorted(seen), "attempts 应单调不减"


def test_on_round_end_not_called_for_failed_round():
    """失败轮没有产出，不该触发落盘回调。"""
    seen: list[int] = []
    run_llm_agent(_mock_daily(), _smart_llm(fail_rounds=frozenset({1})), n_rounds=3, seed=42,
                  on_round_end=lambda r: seen.append(r.n_trials))

    assert len(seen) == 2, f"3 轮中 1 轮失败 → 回调 2 次，实得 {len(seen)}"


def test_manifest_survives_mid_loop_crash(tmp_path):
    """不可恢复的崩溃发生在第 3 轮 → 前两轮的成果必须已经落盘。"""
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    llm = _smart_llm(fail_rounds=frozenset({2}), exc=RuntimeError)
    with pytest.raises(RuntimeError):
        run_agent_mine(_mock_daily(), n_rounds=3, seed=1, out_dir=str(tmp_path),
                       llm_fn=llm, export=False)

    mf = tmp_path / "agent_1_3r" / "manifest.json"
    assert mf.exists(), "崩溃前应已增量落盘，而非全损"
    m = json.loads(mf.read_text())
    assert m["partial"] is True, "中途崩溃留下的 manifest 必须自标 partial"
    assert len(m["attempts"]) >= 1, "应含崩溃前轮次的 attempts"
    assert m["iterations"] == 2, "崩溃在第 3 轮 → 落盘的是前 2 轮"


def test_completed_run_marks_manifest_not_partial(tmp_path):
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    run_agent_mine(_mock_daily(), n_rounds=2, seed=1, out_dir=str(tmp_path),
                   llm_fn=_smart_llm(), export=False)

    m = json.loads((tmp_path / "agent_1_2r" / "manifest.json").read_text())
    assert m["partial"] is False, "正常跑完的 manifest 不应标 partial"
    assert m["iterations"] == 2


# ── team 路径（双路径登记簿：改一侧必查另一侧）────────────────────────────────


def _team_llm(*, fail_rounds: frozenset[int] = frozenset(),
              exc: type[Exception] = LLMClientError):
    """team 角色链的 fake：Hypothesis→Coder→Critic，按轮次注入失败。

    用各角色 system prompt 里的 **JSON key** 分辨调用方——按中文措辞判断不可靠：
    Coder 的 `_syntax_prompt()` 里根本没有「表达式」三个字。
    """
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "verdict" in system:                          # Critic
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if '"expressions"' in system:                    # Coder（含 heal 的 revise_from_error）
            w = 4 + max(st["round"], 0)
            return json.dumps({"expressions": [f"ts_mean(close,{w})"]})
        st["round"] += 1                                 # Hypothesis：每轮恰好一次
        if st["round"] in fail_rounds:
            raise exc(f"round#{st['round']}: 上游不可用")
        return json.dumps({"hypotheses": [f"动量假设{st['round']}"]})
    return fn


def test_team_loop_survives_llm_error_and_continues(tmp_path):
    from factorzen.agents.team_orchestrator import run_team_agent

    res = run_team_agent(_mock_daily(), _team_llm(fail_rounds=frozenset({0})),
                         n_rounds=3, seed=42, index_path=str(tmp_path / "idx.jsonl"))

    assert res.state.iteration == 3, "失败轮应跳过而非崩溃"


def test_team_loop_aborts_after_consecutive_llm_failures(tmp_path):
    from factorzen.agents.team_orchestrator import run_team_agent

    res = run_team_agent(_mock_daily(), _team_llm(fail_rounds=frozenset(range(20))),
                         n_rounds=10, seed=42, index_path=str(tmp_path / "idx.jsonl"),
                         llm_failure_patience=2)

    assert res.state.iteration == 2, f"连续 2 轮失败即终止，实得 {res.state.iteration}"


def test_team_manifest_survives_mid_loop_crash(tmp_path):
    from factorzen.pipelines.factor_mine_team import run_team_mine

    llm = _team_llm(fail_rounds=frozenset({2}), exc=RuntimeError)
    with pytest.raises(RuntimeError):
        run_team_mine(_mock_daily(), n_rounds=3, seed=1, out_dir=str(tmp_path / "out"),
                      index_path=str(tmp_path / "idx.jsonl"), llm_fn=llm, export=False)

    m = json.loads((tmp_path / "out" / "team_1_3r" / "manifest.json").read_text())
    assert m["partial"] is True
    assert m["iterations"] == 2, "崩溃在第 3 轮 → 落盘的是前 2 轮"
