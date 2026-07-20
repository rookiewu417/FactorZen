"""
test_loop_resilience.py：合并自 agents 相关碎片测试（test_loop_resilience.py）。
test_eval_hygiene.py：合并自 agents 相关碎片测试（test_eval_hygiene.py）。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from factorzen.agents.orchestrator import run_llm_agent
from factorzen.agents.self_heal import heal_expressions
from factorzen.agents.team_orchestrator import run_team_agent
from factorzen.discovery.evaluation import make_health_check
from factorzen.discovery.expression import clamp_window_literals, parse_expr
from factorzen.llm.client import LLMClientError


# ==== 来自 test_loop_resilience.py ====
# ==== 来自 test_team_llm_parallel.py ====
def _mock_daily__llm_parallel(n_stocks=40, n_days=180, seed=1):
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


# 路由词用不可能出现在 prompt 样板里的独特假设名——「反转」曾被 write 样板误触发假绿。
_HYP_A = "HYP_ALPHA_ZZ9K"
_HYP_B = "HYP_BETA_QQ7M"
_HYP_FAIL = "HYP_FAIL_XX1P"
_HYP_OK = "HYP_OK_YY2R"


def _threadsafe_routed_llm(*, delays: dict[str, float] | None = None):
    """内容路由 + Lock 计数；可选按路由 sleep，制造完成序 ≠ 提交序。"""
    lock = threading.Lock()
    counts = {"n": 0}
    delays = delays or {}

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        with lock:
            counts["n"] += 1
        if "风控审计员" in text:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in text:  # write_expressions
            if _HYP_A in text:
                time.sleep(delays.get(_HYP_A, 0.0))
                return json.dumps({"expressions": ["ts_mean(close,5)"]})
            if _HYP_B in text:
                time.sleep(delays.get(_HYP_B, 0.0))
                return json.dumps({"expressions": ["ts_std(close,10)"]})
            return json.dumps({"expressions": ["rank(vol)"]})
        # propose
        time.sleep(delays.get("propose", 0.0))
        return json.dumps({"hypotheses": [_HYP_A, _HYP_B]})

    return fn, counts


def _attempt_exprs(res) -> list[str]:
    return [a.expression for a in res.state.attempts]


def test_parallel_serial_parity_same_seed(tmp_path: Path):
    """同 seed 下 llm_workers=4 与 =1 的 attempts 表达式集合与顺序完全一致。"""
    daily = _mock_daily__llm_parallel()
    fn1, _ = _threadsafe_routed_llm()
    fn4, _ = _threadsafe_routed_llm()
    kwargs = dict(
        n_rounds=1, seed=7, heal_rounds=0, hypotheses_per_round=2,
    )
    serial = run_team_agent(
        daily, fn1, index_path=str(tmp_path / "s.jsonl"), llm_workers=1, **kwargs
    )
    parallel = run_team_agent(
        daily, fn4, index_path=str(tmp_path / "p.jsonl"), llm_workers=4, **kwargs
    )
    assert _attempt_exprs(serial) == _attempt_exprs(parallel), (
        f"串行={_attempt_exprs(serial)} 并行={_attempt_exprs(parallel)}"
    )
    # 两个假设的产物都在，且提交序：A 先于 B
    exprs = _attempt_exprs(serial)
    assert "ts_mean(close, 5)" in exprs and "ts_std(close, 10)" in exprs
    assert exprs.index("ts_mean(close, 5)") < exprs.index("ts_std(close, 10)")


def test_parallel_deterministic_despite_completion_order(tmp_path: Path):
    """llm_workers=4 同 seed 跑两次：人为延迟让 B 先完成，attempts 仍逐位一致。"""
    daily = _mock_daily__llm_parallel()
    # A 慢、B 快 → 并行完成序 B 先于 A；装配必须仍按提交序 A→B
    delays = {_HYP_A: 0.08, _HYP_B: 0.01}
    kwargs = dict(
        n_rounds=1, seed=11, heal_rounds=0, hypotheses_per_round=2, llm_workers=4,
    )
    r1 = run_team_agent(
        daily, _threadsafe_routed_llm(delays=delays)[0],
        index_path=str(tmp_path / "d1.jsonl"), **kwargs,
    )
    r2 = run_team_agent(
        daily, _threadsafe_routed_llm(delays=delays)[0],
        index_path=str(tmp_path / "d2.jsonl"), **kwargs,
    )
    assert _attempt_exprs(r1) == _attempt_exprs(r2)
    exprs = _attempt_exprs(r1)
    assert exprs.index("ts_mean(close, 5)") < exprs.index("ts_std(close, 10)"), (
        f"装配须按提交序而非完成序: {exprs}"
    )


def test_parallel_llm_error_counts_as_round_failure(tmp_path: Path):
    """某假设链 write 抛 LLMClientError → 该轮按 LLM 失败计（与串行一致）。"""
    lock = threading.Lock()

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        with lock:
            pass
        if "风控审计员" in text:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in text:
            if _HYP_FAIL in text:
                raise LLMClientError("simulated write failure")
            if _HYP_OK in text:
                return json.dumps({"expressions": ["ts_mean(close,5)"]})
            return json.dumps({"expressions": ["rank(vol)"]})
        return json.dumps({"hypotheses": [_HYP_FAIL, _HYP_OK]})

    # llm_failure_patience=1 → 首轮 LLM 失败即终止；iteration 推进 1
    res = run_team_agent(
        _mock_daily__llm_parallel(), fn, n_rounds=3, seed=3, heal_rounds=0,
        hypotheses_per_round=2, llm_workers=4,
        llm_failure_patience=1,
        index_path=str(tmp_path / "e.jsonl"),
    )
    assert res.state.iteration == 1, (
        f"LLM 失败轮应跳过并计失败，patience=1 应终止: iteration={res.state.iteration}"
    )
    # 失败轮不落有效评估 attempts（护栏/评估未跑完）
    assert all(
        a.expression != "ts_mean(close, 5)" for a in res.state.attempts
    ) or res.n_trials == 0


def test_llm_workers_one_never_constructs_executor(tmp_path: Path, monkeypatch):
    """llm_workers=1 必须不实例化 ThreadPoolExecutor。"""
    created: list[int] = []

    real = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor

    class TrackingPool(real):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **k):
            created.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(
        "factorzen.agents.team_orchestrator.ThreadPoolExecutor", TrackingPool
    )
    fn, _ = _threadsafe_routed_llm()
    run_team_agent(
        _mock_daily__llm_parallel(), fn, n_rounds=1, seed=1, heal_rounds=0,
        hypotheses_per_round=2, llm_workers=1,
        index_path=str(tmp_path / "z.jsonl"),
    )
    assert created == [], f"workers=1 不应进 executor, created={created}"


def test_parser_mine_team_llm_workers_default_is_four():
    from factorzen.cli.main import build_parser

    args = build_parser().parse_args(
        ["mine", "team", "--start", "20220101", "--end", "20231231"]
    )
    assert args.llm_workers == 4


def test_cmd_mine_team_forwards_llm_workers(monkeypatch):
    from factorzen.cli import main as cli

    captured: dict = {}

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        return pl.DataFrame({"ts_code": ["000001.SZ"]})

    def fake_run_team_mine(daily, **kw):
        captured.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine)

    rc = cli.main([
        "mine", "team", "--start", "20220101", "--end", "20231231",
        "--llm-workers", "8",
    ])
    assert rc == 0
    assert captured["llm_workers"] == 8


def test_run_team_mine_forwards_llm_workers_and_records_manifest(
    monkeypatch, tmp_path: Path
):
    """pipeline 透传 llm_workers 到 orchestrator，并写入 manifest params。"""
    from factorzen.agents.state import AgentState
    from factorzen.agents.team_orchestrator import TeamResult
    from factorzen.pipelines import factor_mine_team as fmt

    captured: dict = {}

    def fake_run_team_agent(daily, llm_fn, **kw):
        captured.update(kw)
        return TeamResult(state=AgentState(seed=1), candidates=[], n_trials=0)

    monkeypatch.setattr(fmt, "run_team_agent", fake_run_team_agent)
    fmt.run_team_mine(
        _mock_daily__llm_parallel(), n_rounds=1, seed=1, index_path=str(tmp_path / "e.jsonl"),
        llm_fn=lambda _m: "{}", out_dir=str(tmp_path), run_id="r",
        export=False, llm_workers=6,
    )
    assert captured["llm_workers"] == 6
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["params"]["llm_workers"] == 6


def test_run_team_agent_default_llm_workers_is_one():
    """API 缺省 llm_workers=1（零回归）。"""
    import inspect

    from factorzen.agents.team_orchestrator import run_team_agent as rta

    sig = inspect.signature(rta)
    assert sig.parameters["llm_workers"].default == 1

# ==== 来自 test_mining_loop_resilience.py ====
def _mock_daily__loop_resilience(n_stocks=40, n_days=180, seed=1):
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
    res = run_llm_agent(_mock_daily__loop_resilience(), llm, n_rounds=3, seed=42, library_orthogonal=False)

    assert res.state.iteration == 3, "失败轮应跳过而非崩溃，循环跑满"
    assert len(res.state.attempts) >= 1, "轮 1/2 应正常产出 attempts"


def test_agent_loop_aborts_after_consecutive_llm_failures():
    """LLM 持续不可用时提前终止，不空转跑满 n_rounds。"""
    llm = _smart_llm(fail_rounds=frozenset(range(20)))
    res = run_llm_agent(_mock_daily__loop_resilience(), llm, n_rounds=10, seed=42, llm_failure_patience=2, library_orthogonal=False)

    assert res.state.iteration == 2, f"连续 2 轮失败即终止，实得 {res.state.iteration}"
    assert not res.state.attempts


def test_consecutive_failure_counter_resets_on_success():
    """失败计数器必须在成功轮重置——否则零散的抖动会被累计成「持续不可用」。"""
    llm = _smart_llm(fail_rounds=frozenset({0, 2}))   # 轮 0、2 失败；轮 1、3 成功
    res = run_llm_agent(_mock_daily__loop_resilience(), llm, n_rounds=4, seed=42, llm_failure_patience=2, library_orthogonal=False)

    assert res.state.iteration == 4, "两次孤立失败不该触发 patience=2 的提前终止"


def test_non_llm_exception_still_propagates():
    """只吞 LLMClientError。别的异常（代码 bug、磁盘满）必须冒泡，不许静默吞掉。"""
    llm = _smart_llm(fail_rounds=frozenset({0}), exc=RuntimeError)
    with pytest.raises(RuntimeError):
        run_llm_agent(_mock_daily__loop_resilience(), llm, n_rounds=3, seed=42, library_orthogonal=False)


# ── 增量落盘 ────────────────────────────────────────────────────────────────


def test_on_round_end_called_after_each_successful_round():
    seen: list[int] = []
    run_llm_agent(_mock_daily__loop_resilience(), _smart_llm(), n_rounds=3, seed=42,
                  on_round_end=lambda r: seen.append(len(r.state.attempts)))

    assert len(seen) == 3, f"每轮末应回调一次，实得 {len(seen)}"
    assert seen == sorted(seen), "attempts 应单调不减"


def test_on_round_end_not_called_for_failed_round():
    """失败轮没有产出，不该触发落盘回调。"""
    seen: list[int] = []
    run_llm_agent(_mock_daily__loop_resilience(), _smart_llm(fail_rounds=frozenset({1})), n_rounds=3, seed=42,
                  on_round_end=lambda r: seen.append(r.n_trials))

    assert len(seen) == 2, f"3 轮中 1 轮失败 → 回调 2 次，实得 {len(seen)}"


def test_manifest_survives_mid_loop_crash(tmp_path):
    """不可恢复的崩溃发生在第 3 轮 → 前两轮的成果必须已经落盘。"""
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    llm = _smart_llm(fail_rounds=frozenset({2}), exc=RuntimeError)
    with pytest.raises(RuntimeError):
        run_agent_mine(_mock_daily__loop_resilience(), n_rounds=3, seed=1, out_dir=str(tmp_path),
                       llm_fn=llm, export=False, run_id="crash")

    mf = tmp_path / "crash" / "manifest.json"
    assert mf.exists(), "崩溃前应已增量落盘，而非全损"
    m = json.loads(mf.read_text())
    assert m["partial"] is True, "中途崩溃留下的 manifest 必须自标 partial"
    assert len(m["attempts"]) >= 1, "应含崩溃前轮次的 attempts"
    assert m["iterations"] == 2, "崩溃在第 3 轮 → 落盘的是前 2 轮"


def test_completed_run_marks_manifest_not_partial(tmp_path):
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    run_agent_mine(_mock_daily__loop_resilience(), n_rounds=2, seed=1, out_dir=str(tmp_path),
                   llm_fn=_smart_llm(), export=False, run_id="done")

    m = json.loads((tmp_path / "done" / "manifest.json").read_text())
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

    res = run_team_agent(_mock_daily__loop_resilience(), _team_llm(fail_rounds=frozenset({0})),
                         n_rounds=3, seed=42, index_path=str(tmp_path / "idx.jsonl"))

    assert res.state.iteration == 3, "失败轮应跳过而非崩溃"


def test_team_loop_aborts_after_consecutive_llm_failures(tmp_path):
    from factorzen.agents.team_orchestrator import run_team_agent

    res = run_team_agent(_mock_daily__loop_resilience(), _team_llm(fail_rounds=frozenset(range(20))),
                         n_rounds=10, seed=42, index_path=str(tmp_path / "idx.jsonl"),
                         llm_failure_patience=2)

    assert res.state.iteration == 2, f"连续 2 轮失败即终止，实得 {res.state.iteration}"


def test_team_manifest_survives_mid_loop_crash(tmp_path):
    from factorzen.pipelines.factor_mine_team import run_team_mine

    llm = _team_llm(fail_rounds=frozenset({2}), exc=RuntimeError)
    with pytest.raises(RuntimeError):
        run_team_mine(_mock_daily__loop_resilience(), n_rounds=3, seed=1, out_dir=str(tmp_path / "out"),
                      index_path=str(tmp_path / "idx.jsonl"), llm_fn=llm, export=False,
                      run_id="team_crash")

    m = json.loads((tmp_path / "out" / "team_crash" / "manifest.json").read_text())
    assert m["partial"] is True
    assert m["iterations"] == 2, "崩溃在第 3 轮 → 落盘的是前 2 轮"

# ==== 来自 test_eval_hygiene.py ====
# ==== 来自 test_exception_and_accounting_hygiene.py ====
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



@pytest.mark.parametrize("garbage", ["", "{", "不是json", "[1,2]"])
def test_extract_json_never_raises_on_string_garbage(garbage):
    """回归守卫：`_extract_json` 对任何字符串都不抛（只返回 None 或 dict）。"""
    from factorzen.llm.generation import _extract_json

    out = _extract_json(garbage)
    assert out is None or isinstance(out, dict)

# ==== 来自 test_w5_llm_waste.py ====
# ── W5a ──────────────────────────────────────────────────────────────────────

def test_heal_drops_unknown_op_no_llm():
    """ts_delta 等未知算子不触发 revise_from_error，计数=1。"""
    calls = {"n": 0}

    def fake(_msgs):
        calls["n"] += 1
        raise AssertionError("未知算子不应触发 LLM")

    stats: dict = {}
    healed = heal_expressions(
        ["ts_delta(close, 5)"], "动量", fake, max_rounds=2, stats=stats,
    )
    assert healed == []
    assert calls["n"] == 0
    assert stats.get("n_unknown_op_dropped") == 1


def test_heal_syntax_error_still_enters_heal():
    """普通语法错（非未知算子）仍进 heal 调 LLM。"""
    calls = {"n": 0}

    def fake(_msgs):
        calls["n"] += 1
        return json.dumps({"expressions": ["ts_mean(close, 5)"]})

    healed = heal_expressions(["ts_mean()"], "动量", fake, max_rounds=2)
    assert calls["n"] >= 1
    assert any("ts_mean" in h for h in healed)
    for h in healed:
        parse_expr(h)


def test_heal_drop_unknown_ops_false_restores_old():
    """drop_unknown_ops=False 时未知算子仍进 heal（兼容开关）。"""
    calls = {"n": 0}

    def fake(_msgs):
        calls["n"] += 1
        return json.dumps({"expressions": ["ts_mean(close, 5)"]})

    healed = heal_expressions(
        ["ts_delta(close, 5)"], "h", fake, max_rounds=2, drop_unknown_ops=False,
    )
    assert calls["n"] >= 1
    assert healed


# ── W5b ──────────────────────────────────────────────────────────────────────

def test_clamp_window_over_budget():
    """504 窗 + 预算 400 → 钳到 400。"""
    out, did = clamp_window_literals(
        "ts_mean(amount, 504)", {"amount": 400}, None,
    )
    assert did is True
    assert "400" in out
    assert "504" not in out
    node = parse_expr(out)
    assert getattr(node, "window", None) == 400


def test_clamp_window_budget_sufficient_unchanged():
    """预算充足 → 不动。"""
    expr = "ts_mean(amount, 20)"
    out, did = clamp_window_literals(expr, {"amount": 400}, None)
    assert did is False
    assert out == expr


def test_clamp_window_no_window_literal():
    """无窗口字面量（纯截面）→ 原样。"""
    expr = "rank(amount)"
    out, did = clamp_window_literals(expr, {"amount": 400}, None)
    assert did is False
    assert out == expr


def test_clamp_window_parse_fail_passthrough():
    expr = "not_parseable!!!"
    out, did = clamp_window_literals(expr, {"amount": 400}, None)
    assert out == expr and did is False


def test_clamp_nested_windows():
    """嵌套：仅超 cap 的窗口被钳。"""
    out, did = clamp_window_literals(
        "ts_mean(delta(amount, 20), 504)", {"amount": 400}, None,
    )
    assert did is True
    node = parse_expr(out)
    assert node.window == 400
    assert node.children[0].window == 20  # type: ignore[attr-defined]


# ── W5c ──────────────────────────────────────────────────────────────────────

def _mock_daily__llm_waste(n_stocks=30, n_days=120, seed=3):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def test_empty_round_skips_critic_llm(tmp_path: Path):
    """new_cands=[] 时 critic 零调用，verdict=revise_hypothesis，critic_skipped=True。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    critic_calls = {"n": 0}

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            critic_calls["n"] += 1
            return json.dumps({"verdict": "keep", "reason": "should_not_run"})
        if "翻译成" in text:
            # 故意产重复/已评估表达式 → 评估后可能无新候选进护栏
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    # 让护栏拒绝所有候选 → new_cands 空
    def _guard_reject(state, **kw):
        return state  # 不注入 candidates

    daily = _mock_daily__llm_waste()
    with patch("factorzen.agents.team_orchestrator.node_guardrails", side_effect=_guard_reject):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"),
        )
    assert critic_calls["n"] == 0, f"空轮不应调 critic，实得 {critic_calls['n']}"
    assert res.rounds_log
    r0 = res.rounds_log[0]
    assert r0.get("critic_skipped") is True
    assert r0["verdict"] == "revise_hypothesis"
    assert "无新候选" in r0["reason"]


def test_nonempty_round_still_calls_critic(tmp_path: Path):
    """有新候选时 critic 仍被调用。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    critic_calls = {"n": 0}

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            critic_calls["n"] += 1
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in text:
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    def _inject(state, *, ledger, **_kw):
        for a in state.attempts:
            if a.iteration != state.iteration:
                continue
            a.passed_guardrails = True
            state.candidates.append({
                "expression": a.expression, "hypothesis": a.hypothesis,
                "ic_train": 0.05, "ir_train": 0.4, "turnover": 0.1,
                "holdout_ic": 0.04, "holdout_ir": 0.3,
                "dsr": 0.7, "dsr_pvalue": 0.05, "n_train": 100,
                "n_holdout_days": 80, "ic_ci_low": 0.01, "ic_ci_high": 0.08,
            })
            ledger.record(1)
        return state

    daily = _mock_daily__llm_waste()
    with patch("factorzen.agents.team_orchestrator.node_guardrails", side_effect=_inject):
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1, heal_rounds=0,
            index_path=str(tmp_path / "e.jsonl"),
        )
    assert critic_calls["n"] >= 1
    assert res.rounds_log[0].get("critic_skipped") is False

# ==== 来自 test_agent_health_check.py ====
_ALL_NULL = "div(close, sub(close, close))"   # 分母恒 0 → _safe_div 全列 null
_HEALTHY = "ts_mean(close, 5)"


def _mock_daily__health_check(n_days: int = 60, n_codes: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(n_codes)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98, "vol": 1e6, "amount": 1e7})
    return pl.DataFrame(rows)


# ─────────────────────────── make_health_check ───────────────────────────

def test_healthy_expression_reports_no_diagnosis():
    check = make_health_check(_mock_daily__health_check())
    assert check(_HEALTHY) is None


def test_all_null_factor_is_diagnosed():
    """全 null = 静默失明，必须被抓出来（PR #61 那类 bug 的兜底）。"""
    check = make_health_check(_mock_daily__health_check())
    diag = check(_ALL_NULL)
    assert diag is not None
    assert "null" in diag.lower() or "NaN" in diag


def test_null_ratio_threshold_is_configurable_and_respected():
    """健康表达式在极严阈值下也会被判不健康 —— 证明判据真的是比例而非硬编码。"""
    daily = _mock_daily__health_check()
    assert make_health_check(daily, max_null_ratio=0.5)(_HEALTHY) is None
    assert make_health_check(daily, max_null_ratio=0.001)(_HEALTHY) is not None


def test_parse_error_is_diagnosed():
    check = make_health_check(_mock_daily__health_check())
    diag = check("not_a_func(")
    assert diag is not None and "解析" in diag


def test_eval_error_is_diagnosed(monkeypatch):
    """求值抛异常 → 诊断带上异常类型与消息（供 LLM 修正）。"""
    from factorzen.discovery import evaluation as ev

    check = ev.make_health_check(_mock_daily__health_check())

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ev, "evaluate_materialized", boom)
    diag = check(_HEALTHY)
    assert diag is not None
    assert "求值失败" in diag and "RuntimeError" in diag and "boom" in diag


# ─────────────────── heal_expressions × health_check 集成 ───────────────────

def test_heal_revises_expression_that_parses_but_evaluates_all_null():
    """parse 通过但全 null → 诊断回灌 LLM → 换成健康表达式。"""
    prompts: list[str] = []

    def fake(msgs):
        prompts.append(msgs[1]["content"])
        return json.dumps({"expressions": [_HEALTHY]})

    healed = heal_expressions([_ALL_NULL], "动量", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily__health_check()))

    assert healed == [_HEALTHY]
    assert len(prompts) == 1, "健康表达式不应再触发第二次修正"
    assert _ALL_NULL in prompts[0]
    assert "null" in prompts[0].lower() or "NaN" in prompts[0]


def test_heal_gives_up_when_llm_keeps_producing_unhealthy_expressions():
    """LLM 持续产全 null → max_rounds 耗尽后丢弃，不死循环、不放行病态因子。"""
    def fake(_msgs):
        return json.dumps({"expressions": [_ALL_NULL]})

    healed = heal_expressions([_ALL_NULL], "h", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily__health_check()))
    assert healed == []


def test_healthy_expression_never_triggers_llm_even_with_health_check():
    """零额外成本不变量：健康表达式不调用 LLM。"""
    def fake(_msgs):
        raise AssertionError("健康表达式不应触发 LLM 修正")

    healed = heal_expressions([_HEALTHY], "h", fake, max_rounds=2,
                              health_check=make_health_check(_mock_daily__health_check()))
    assert len(healed) == 1


def test_no_health_check_is_zero_regression():
    """health_check=None（默认）→ 只查 parse，全 null 表达式照旧放行（既有行为）。"""
    def fake(_msgs):
        raise AssertionError("parse 通过的表达式不应触发 LLM")

    healed = heal_expressions([_ALL_NULL], "h", fake, max_rounds=2)
    assert len(healed) == 1
    assert "div" in healed[0]


# ─────────────────── 接线：health_check 必须真的抵达自愈循环 ───────────────────
# 能力层实现完 ≠ 用户用得上。这两条从 node_generate / run_team_agent 这一层出发，
# 断言全 null 表达式在真实闭环里被诊断并修正掉，而不是靠 inspect.signature 看形参。

def test_node_generate_heals_all_null_expression_end_to_end():
    """M5 单 Agent：LLM 产出全 null 表达式 → 求值诊断回灌 → pending 里只剩健康表达式。"""
    from factorzen.agents.nodes import node_generate
    from factorzen.agents.state import AgentState
    from factorzen.discovery.scoring import DataBundle

    daily = _mock_daily__health_check(n_days=120, n_codes=10)
    seq = [
        json.dumps({"hypothesis": "动量", "expressions": [_ALL_NULL], "rationale": "r"}),
        json.dumps({"expressions": [_HEALTHY]}),        # revise_from_error 的修正
        json.dumps({"consistent": True, "reason": "ok"}),  # semantic_check
    ]
    i = {"k": 0}

    def fn(_m):
        v = seq[min(i["k"], len(seq) - 1)]
        i["k"] += 1
        return v

    state = node_generate(AgentState(seed=1), fn, daily=daily,
                          bundle=DataBundle.build(daily), heal_rounds=2)
    pending = [p.expression for p in state._pending]
    assert pending == [_HEALTHY], f"全 null 表达式未被自愈: {pending}"


def test_team_agent_heals_all_null_expression_end_to_end(tmp_path):
    """M6 团队：Coder 写出全 null 表达式 → 求值诊断回灌 → 落库的是健康表达式。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    # n_codes≥30：叶子 holdout 覆盖门用 _MIN_CROSS_SAMPLES=30，截面太薄会被整批摘叶导致 0 评估。
    daily = _mock_daily__health_check(n_days=200, n_codes=40)
    seq = [
        json.dumps({"hypotheses": ["动量"]}),           # propose_hypotheses
        json.dumps({"expressions": [_ALL_NULL]}),       # write_expressions
        json.dumps({"expressions": [_HEALTHY]}),        # revise_from_error
        json.dumps({"verdict": "keep", "reason": "ok"}),  # critique
    ]
    i = {"k": 0}

    def fn(_m):
        v = seq[min(i["k"], len(seq) - 1)]
        i["k"] += 1
        return v

    res = run_team_agent(daily, fn, n_rounds=1, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), heal_rounds=2)
    exprs = [a.expression for a in res.state.attempts]
    assert exprs, "本轮应有被评估的表达式"
    assert all("div" not in e for e in exprs), f"全 null 表达式未被自愈就进了评估: {exprs}"
