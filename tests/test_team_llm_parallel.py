# tests/test_team_llm_parallel.py
"""轮内 LLM 调用并行化：parity / 确定性 / 异常契约 / workers=1 零回归 / CLI 透传。

硬约束：llm_workers=1 不进 ThreadPoolExecutor（既有有状态 scripted llm 依赖调用序）。
并行路径 futures 按提交序装配，产物与完成序无关。
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.agents.team_orchestrator import run_team_agent
from factorzen.llm.client import LLMClientError


def _mock_daily(n_stocks=40, n_days=180, seed=1):
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
    daily = _mock_daily()
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
    daily = _mock_daily()
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
        _mock_daily(), fn, n_rounds=3, seed=3, heal_rounds=0,
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
        _mock_daily(), fn, n_rounds=1, seed=1, heal_rounds=0,
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

    def fake_prepare(start, end, universe=None, lookback_days=None):
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
        _mock_daily(), n_rounds=1, seed=1, index_path=str(tmp_path / "e.jsonl"),
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
