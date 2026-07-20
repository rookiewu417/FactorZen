"""C2：team 全量残差选槽 + session 末 lift 钩子 + CLI 透传。

全部 mock 离线；upsert_lift_admissions 一律 monkeypatch（不依赖任务 D 真实现）。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.agents.team_orchestrator import (
    _session_end_auto_lift,
    run_team_agent,
    write_team_manifest,
)
from factorzen.discovery.guardrails import (
    DEFAULT_GRAY_IC_FLOOR,
    REJECT_CATEGORY_LIFT_QUEUE,
)

# ── fixtures ────────────────────────────────────────────────────────────────


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
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _scripted_exprs(exprs_by_round: list[list[str]]):
    """每轮固定表达式列表；Critic 一律 keep。"""
    round_i = {"k": 0}
    hyp = json.dumps({"hypotheses": ["HYP_LIFT"]})
    keep = json.dumps({"verdict": "keep", "reason": "ok"})

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            return keep
        if "翻译成" in text:
            ri = min(round_i["k"], len(exprs_by_round) - 1)
            return json.dumps({"expressions": exprs_by_round[ri]})
        # propose：推进轮次计数（每轮一次）
        ri = round_i["k"]
        round_i["k"] += 1
        del ri
        return hyp

    return fn


def _panel(n_dates: int, n_stocks: int = 10, start=dt.date(2022, 1, 3)):
    days, d = [], start
    while len(days) < n_dates:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for dd in days:
            rows.append({
                "trade_date": dd, "ts_code": c, "factor_value": float(hash((c, dd)) % 100) / 100.0,
            })
    return pl.DataFrame(rows)


# ── 1. 全量残差标记（非 top-K → lift_queue + residual_ic_train）─────────────


def test_full_residual_marks_nontopk_lift_queue(monkeypatch):
    """多表达式 top_k=1：非 top-K 且 |residual|≥gray floor → lift_queue；全体带 residual_ic_train。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.discovery.residual import ResidualICResult
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mock_daily()
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)
    state = AgentState(seed=1, iteration=0)
    # 3 候选；全量残差按 attempts 序写入
    rows = [
        ("ts_mean(close, 5)", 0.08),
        ("ts_mean(close, 10)", 0.02),
        ("ts_std(close, 5)", 0.015),
    ]
    for expr, ic in rows:
        state.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=expr,
            compile_ok=True, ic_train=ic, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=ic * 10, n_train=100,
        ))

    call_i = {"n": 0}
    # 全量 train 三次均 ≥ DEFAULT_GRAY_IC_FLOOR(0.008)；按 |residual| 排序后
    # top_k=1 取 0.02，非 top-K 为 0.009 / 0.0085（均过 train floor，需补 holdout）
    order_ric = [0.009, 0.02, 0.0085]

    def fake_ric(candidate, lib_panel, fwd_returns, *, ret_col="fwd_ret_1d",
                 projector=None):
        i = call_i["n"]
        call_i["n"] += 1
        if i < 3:
            return ResidualICResult(order_ric[i], n_days=80)
        # holdout 残差补算：覆盖充足
        return ResidualICResult(0.01, n_days=80)

    monkeypatch.setattr(
        "factorzen.discovery.residual.compute_residual_ic", fake_ric,
    )
    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda *a, **k: type("H", (), {
            "ic_mean": 0.01, "ir": 0.5, "ci": (0.0, 0.02), "n_days": 80,
        })(),
    )
    monkeypatch.setattr(
        "factorzen.discovery.scoring.library_orthogonal_check",
        lambda *a, **k: (True, 0.1, None),
    )
    # 主门一律不过 → top-K 与非 top-K 均走 is_lift_queue（统一后缀待组合裁决）
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.acceptance_reasons",
        lambda **kw: ["残差IC太弱"],
    )

    lib_pool = {"lib_f": _panel(120)}
    ledger = TrialLedger()
    node_guardrails(
        state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
        ledger=ledger, top_k=1, lib_pool=lib_pool, objective="residual",
    )

    for a in state.attempts:
        assert a.residual_ic_train is not None, f"{a.expression} 缺 residual_ic_train"
        assert abs(a.residual_ic_train) >= DEFAULT_GRAY_IC_FLOOR

    nontop_marks = [
        a for a in state.attempts
        if a.reject_category == REJECT_CATEGORY_LIFT_QUEUE
        and a.reject_reason
        and "待组合裁决" in a.reject_reason
    ]
    assert len(nontop_marks) >= 2, (
        f"非 top-K 应≥2 个 lift 队列(待组合裁决): "
        f"{[(a.expression, a.reject_category, a.reject_reason) for a in state.attempts]}"
    )
    assert all(
        "覆盖待lift验" not in (a.reject_reason or "")
        for a in state.attempts
    )


def test_nontopk_holdout_coverage_shortfall_not_queued(monkeypatch):
    """W1b：非 top-K train residual ≥ floor 但 holdout 覆盖 <60 天 → 不入队。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.discovery.residual import ResidualICResult
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mock_daily()
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)
    state = AgentState(seed=1, iteration=0)
    rows = [
        ("ts_mean(close, 5)", 0.08),
        ("ts_mean(close, 10)", 0.02),
    ]
    for expr, ic in rows:
        state.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=expr,
            compile_ok=True, ic_train=ic, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=ic * 10, n_train=100,
        ))

    call_i = {"n": 0}

    def fake_ric(candidate, lib_panel, fwd_returns, *, ret_col="fwd_ret_1d",
                 projector=None):
        i = call_i["n"]
        call_i["n"] += 1
        if i < 2:
            # train：均 ≥ floor
            return ResidualICResult(0.009 if i == 0 else 0.02, n_days=80)
        # holdout 补算：覆盖不足
        return ResidualICResult(0.01, n_days=30)

    monkeypatch.setattr(
        "factorzen.discovery.residual.compute_residual_ic", fake_ric,
    )
    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda *a, **k: type("H", (), {
            "ic_mean": 0.01, "ir": 0.5, "ci": (0.0, 0.02), "n_days": 80,
        })(),
    )
    monkeypatch.setattr(
        "factorzen.discovery.scoring.library_orthogonal_check",
        lambda *a, **k: (True, 0.1, None),
    )
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.acceptance_reasons",
        lambda **kw: ["残差IC太弱"],
    )

    lib_pool = {"lib_f": _panel(120)}
    ledger = TrialLedger()
    node_guardrails(
        state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
        ledger=ledger, top_k=1, lib_pool=lib_pool, objective="residual",
    )

    # 排序后 top 是 0.02（ts_mean close 10）；非 top 是 0.009（close 5）
    low = next(a for a in state.attempts if a.expression == "ts_mean(close, 5)")
    assert low.reject_category != REJECT_CATEGORY_LIFT_QUEUE, (
        f"覆盖不足应不入队: {low.reject_category=} {low.reject_reason=}"
    )


def test_slot_key_prefers_high_residual_over_high_raw_ic(monkeypatch):
    """泄漏 A 回归：裸 IC 高残差低 vs 裸 IC 低残差高 → 后者进验收槽（top_k=1）。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.discovery.residual import ResidualICResult
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mock_daily(n_days=200, seed=2)
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)

    high_raw_low_res = "ts_mean(close, 5)"
    low_raw_high_res = "ts_std(close, 10)"

    state = AgentState(seed=7, iteration=0)
    state.attempts = [
        AttemptRecord(
            0, "h", high_raw_low_res, True, 0.09, False, None, None,
            ir_train=3.0, n_train=100,
        ),
        AttemptRecord(
            0, "h", low_raw_high_res, True, 0.012, False, None, None,
            ir_train=0.5, n_train=100,
        ),
    ]

    # 全量残差：按 expression 返回固定值
    # 残差设定：high_raw_low_res→0.001（低残差）、low_raw_high_res→0.025（高残差,应进槽）
    # compute_residual_ic 看不到 expression；用调用次数：passed 保持 attempts 序
    # 第一次 train 对应 high_raw，第二次 low_raw；之后 holdout 等
    train_calls = {"n": 0}
    train_order = [0.001, 0.025]

    def fake_ric(candidate, lib_panel, fwd_returns, *, ret_col="fwd_ret_1d", projector=None):
        # 前两次是全量 train；后续是 top-K holdout（或 train 补算）
        n = train_calls["n"]
        train_calls["n"] += 1
        if n < 2:
            return ResidualICResult(train_order[n], n_days=90)
        # holdout residual for the selected slot candidate
        return ResidualICResult(0.02, n_days=90)

    monkeypatch.setattr(
        "factorzen.discovery.residual.compute_residual_ic", fake_ric,
    )

    lib_pool = {"lib_f": _panel(150)}
    ledger = TrialLedger()

    # 让 acceptance 尽量过：mock acceptance_reasons → [] 仅对高残差候选
    # 更简单：检查 top_k 循环处理的第一个——通过 spy 谁先被 holdout 求值
    evaluated: list[str] = []
    from factorzen.discovery import expression as expr_mod
    _orig_parse = expr_mod.parse_expr

    def tracking_parse(expression, leaf_map=None):
        # only track during guardrails holdout loop roughly
        if expression in (high_raw_low_res, low_raw_high_res) and (
            expression not in evaluated or evaluated[-1] != expression
        ):
            evaluated.append(expression)
        return _orig_parse(expression, leaf_map)

    monkeypatch.setattr("factorzen.agents.nodes.parse_expr", tracking_parse)

    # 避免 holdout/护栏炸：mock holdout_ic_result + library check + acceptance

    class _HRes:
        ic_mean = 0.02
        ir = 1.0
        ci = (0.01, 0.03)
        n_days = 80

    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda *a, **k: _HRes(),
    )
    monkeypatch.setattr(
        "factorzen.discovery.scoring.library_orthogonal_check",
        lambda *a, **k: (True, 0.1, None),
    )
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.acceptance_reasons",
        lambda **kw: [],
    )
    monkeypatch.setattr(
        "factorzen.discovery.scoring.max_correlation",
        lambda *a, **k: 0.0,
    )

    node_guardrails(
        state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
        ledger=ledger, top_k=1, lib_pool=lib_pool, objective="residual",
    )

    # 验收槽唯一候选应是高残差者
    assert len(state.candidates) == 1, state.candidates
    assert state.candidates[0]["expression"] == low_raw_high_res, (
        f"槽位应按 |residual| 排序，期望 {low_raw_high_res}，"
        f"得到 {state.candidates[0]['expression']}"
    )
    # 低残差高裸 IC 不应进候选（top_k=1 且按 residual 排序）
    assert all(c["expression"] != high_raw_low_res for c in state.candidates)


# ── 2. session 末 lift 钩子 ─────────────────────────────────────────────────


def _state_with_lift_queue(exprs: list[str]) -> AgentState:
    st = AgentState(seed=1)
    for e in exprs:
        st.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=e,
            compile_ok=True, ic_train=0.02, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=1.0, n_train=100,
            residual_ic_train=0.01,
            reject_category=REJECT_CATEGORY_LIFT_QUEUE,
            reject_reason="x(lift队列,待组合裁决)",
        ))
    st.n_gray_zone = len(exprs)
    return st


class _FakeCtx:
    leaf_map = None


def _holdout_and_mat(n_days=120):
    """构造 holdout + 覆盖充足的 materialize（从 holdout 起点起 80 日）。"""
    daily = _mock_daily(n_days=n_days)
    dates = daily["trade_date"].unique().sort()
    cut = dates[int(len(dates) * 0.8)]
    holdout = daily.filter(pl.col("trade_date") >= cut)

    def mat(expr):
        return _panel(80, start=cut)

    return daily, holdout, mat


def test_lift_hook_group_fail_skips_per_candidate(monkeypatch):
    """组门不过 → 不跑逐候选且 upsert 未被调用。"""
    state = _state_with_lift_queue(["ts_mean(close, 5)", "rank(vol)"])
    daily, holdout, mat = _holdout_and_mat()

    calls = {"group": 0, "per": 0, "upsert": 0}

    def fake_group(*a, **k):
        calls["group"] += 1
        return {
            "lift": 0.0001, "lift_se": 0.01, "error": None,
            "n_candidates": 2, "expressions": ["ts_mean(close, 5)", "rank(vol)"],
        }

    def fake_per(*a, **k):
        calls["per"] += 1
        return []

    def fake_upsert(*a, **k):
        calls["upsert"] += 1
        return {"added_active": 0, "added_probation": 0, "rejected": 0, "errors": []}

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        fake_upsert, raising=False,
    )

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root=str(Path("/tmp/lib")), seed=1,
        auto_lift=True, lift_se_mult=1.0,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        ret_df=_panel(100).rename({"factor_value": "ret"}),
        horizon=1,
    )
    assert calls["group"] == 1
    assert calls["per"] == 0
    assert calls["upsert"] == 0
    assert meta["lift_group"] is not None
    assert meta["lift_results"] == []
    assert meta["n_lift_evaluated"] == 1


def test_lift_hook_group_se_not_finite_fails_gate(monkeypatch):
    """组门 SE 缺失/非有限 = 区间证据不完整 → 拒，不跑逐候选（不再按 0 处理）。

    旧行为：SE=None/NaN 按 0 → bar 退化为裸 threshold，lift 高就放行——统计上把
    「无 SE」当「零方差」。与 lift_admission 的 SE 契约对齐。
    """
    for bad_se in (None, float("nan")):
        state = _state_with_lift_queue(["ts_mean(close, 5)", "rank(vol)"])
        daily, holdout, mat = _holdout_and_mat()
        calls = {"per": 0}

        def fake_group(*a, _se=bad_se, **k):
            return {
                "lift": 0.5, "lift_se": _se, "error": None,  # lift 远超门槛
                "n_candidates": 2, "expressions": ["ts_mean(close, 5)", "rank(vol)"],
            }

        def fake_per(*a, _c=calls, **k):
            _c["per"] += 1
            return []

        monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
        monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)

        meta = _session_end_auto_lift(
            state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
            market="ashare", library_root=str(Path("/tmp/lib")), seed=1,
            auto_lift=True, lift_se_mult=1.0,
            materialize_candidate=mat,
            active_factor_dfs={"base": _panel(100)},
            ret_df=_panel(100).rename({"factor_value": "ret"}),
            horizon=1,
        )
        assert calls["per"] == 0, f"SE={bad_se!r} 时组门应拒、不跑逐候选"
        assert meta["lift_results"] == []


def test_lift_hook_group_pass_runs_per_and_upsert(monkeypatch):
    """组门过 → 逐候选进 manifest、upsert 收到正确行。"""
    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily, holdout, mat = _holdout_and_mat()

    upsert_rows = []

    def fake_group(*a, **k):
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "n_candidates": 1, "expressions": ["ts_mean(close, 5)"],
        }

    def fake_per(queue, **k):
        return [{
            "expression": "ts_mean(close, 5)",
            "lift": 0.008, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
        }]

    def fake_upsert(rows, *, market, **kw):
        upsert_rows.extend(rows)
        return {"added_active": 1, "added_probation": 0, "rejected": 0, "errors": []}

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        fake_upsert, raising=False,
    )

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp/lib", seed=1,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        horizon=1,
    )
    assert len(meta["lift_results"]) == 1
    assert meta["lift_admissions"]["added_active"] == 1
    assert meta["n_lift_evaluated"] == 2  # group + 1 per
    assert upsert_rows and upsert_rows[0]["expression"] == "ts_mean(close, 5)"


def test_lift_hook_exception_does_not_kill_session(monkeypatch, tmp_path: Path):
    """钩子内部异常 → 返回 lift_error，不向外抛；run_team_agent 仍完成。"""
    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily, holdout, mat = _holdout_and_mat()

    def boom_group(*a, **k):
        raise RuntimeError("lift exploded")

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", boom_group)

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root=str(tmp_path), seed=1,
        materialize_candidate=mat,
        horizon=1,
    )
    assert meta["lift_error"] is not None
    assert "RuntimeError" in meta["lift_error"]
    assert meta["n_lift_queue"] == 1

    # 端到端：空队列 auto_lift 不炸；有队列时 hook 异常也被吞
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 10
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    def boom_hook(*a, **k):
        # 模拟钩子顶层被包住前的异常——真正路径应内部消化；
        # 这里验证 run_team_agent 对 hook 返回值的容错：改用返回 lift_error
        return {
            "n_lift_queue": 1, "lift_group": None, "lift_results": [],
            "lift_admissions": {"added_active": 0, "added_probation": 0},
            "n_lift_evaluated": 0, "lift_dropped_coverage": [],
            "lift_error": "RuntimeError: simulated",
        }

    monkeypatch.setattr(
        "factorzen.agents.team_orchestrator._session_end_auto_lift", boom_hook,
    )
    res = run_team_agent(
        _mock_daily(), fn, n_rounds=1, seed=1,
        index_path=str(tmp_path / "e.jsonl"), heal_rounds=0,
        auto_lift=True,
    )
    assert res.state.iteration == 1
    assert res.lift_error == "RuntimeError: simulated"


def test_lift_hook_drops_low_oos_coverage(monkeypatch):
    """物化后 OOS 天数不足 → lift_dropped_coverage，不进 lift。"""
    state = _state_with_lift_queue(["ts_mean(close, 5)", "rank(vol)"])
    daily = _mock_daily(n_days=120)
    # holdout 起点靠后
    dates = daily["trade_date"].unique().sort()
    cut = dates[int(len(dates) * 0.8)]
    holdout = daily.filter(pl.col("trade_date") >= cut)

    group_calls = []

    def fake_group(queue, **k):
        group_calls.append(list(queue))
        return {"lift": 0.01, "lift_se": 0.001, "error": None, "expressions": []}

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr(
        "factorzen.discovery.lift_test.run_lift_tests",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        lambda *a, **k: {"added_active": 0, "added_probation": 0, "rejected": 0, "errors": []},
        raising=False,
    )

    cut_d = cut

    def mat(expr):
        if expr == "ts_mean(close, 5)":
            # 仅 5 个 OOS 日（<< 60）
            return _panel(5, start=cut_d)
        # 从 holdout 起点起 80 日 → 覆盖充足
        return _panel(80, start=cut_d)

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp/lib", seed=1,
        materialize_candidate=mat,
        active_factor_dfs={"b": _panel(100)},
        horizon=1,
    )
    dropped_exprs = {d["expression"] for d in meta["lift_dropped_coverage"]}
    assert "ts_mean(close, 5)" in dropped_exprs
    assert group_calls, "覆盖充足的候选应进组测"
    assert all(c["expression"] != "ts_mean(close, 5)" for c in group_calls[0])
    assert any(c["expression"] == "rank(vol)" for c in group_calls[0])


def test_lift_hook_dedupes_expressions(monkeypatch):
    """同一 expression 多 attempt → 队列去重。"""
    state = _state_with_lift_queue(["ts_mean(close, 5)", "ts_mean(close, 5)"])
    assert len(state.attempts) == 2
    daily, holdout, mat = _holdout_and_mat()
    seen_n = []

    def fake_group(queue, **k):
        seen_n.append(len(queue))
        return {"lift": 0.0, "lift_se": 0.1, "error": None}

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp", seed=0,
        materialize_candidate=mat,
        active_factor_dfs={"b": _panel(100)},
        horizon=1,
    )
    assert meta["n_lift_queue"] == 1
    assert seen_n == [1]


def test_write_team_manifest_includes_lift_fields(tmp_path: Path):
    from factorzen.agents.team_orchestrator import TeamResult

    state = AgentState(seed=1)
    res = TeamResult(
        state=state, candidates=[], n_trials=3,
        n_lift_queue=2,
        lift_group={"lift": 0.01},
        lift_results=[{"expression": "x", "lift": 0.01}],
        lift_admissions={"added_active": 1, "added_probation": 0},
        n_lift_evaluated=3,
        lift_dropped_coverage=[{"expression": "y", "n_oos_days": 3}],
        lift_error=None,
    )
    path = write_team_manifest(res, out_dir=str(tmp_path), run_id="r1", params={})
    man = json.loads(path.read_text(encoding="utf-8"))
    assert man["n_lift_queue"] == 2
    assert man["lift_group"]["lift"] == 0.01
    assert man["n_lift_evaluated"] == 3
    assert man["lift_admissions"]["added_active"] == 1
    assert man["n_gray_zone"] == 0  # 旧字段仍在


# ── 3. CLI 透传（parser 最外层，禁止 inspect.signature）─────────────────────


def test_cli_no_auto_lift_forwards_to_run_team_agent(monkeypatch, tmp_path):
    """从 CLI parser 最外层出发：--no-auto-lift/--lift-se-mult 两段透传。

    段1：CLI → run_team_mine 收到 auto_lift=False / lift_se_mult=1.5；
    段2：真 run_team_mine → run_team_agent 同名转发（mock agent 捕获）。
    """
    from factorzen.cli import main as cli

    fake_daily = pl.DataFrame({
        "trade_date": [dt.date(2022, 1, 4)],
        "ts_code": ["600000.SH"],
        "close": [10.0], "open": [10.0], "high": [10.0], "low": [10.0],
        "vol": [1e6], "amount": [1e7],
    })
    monkeypatch.setattr(
        "factorzen.cli.main._prepare_agent_mining_data",
        lambda args: (fake_daily, None, {}),
    )

    # parser 契约
    p = cli.build_parser()
    args = p.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20231231",
        "--no-auto-lift", "--lift-se-mult", "1.5",
    ])
    assert args.no_auto_lift is True
    assert args.lift_se_mult == 1.5

    # 段1：CLI → run_team_mine kwargs
    mine_kw: dict = {}

    def fake_mine(daily, **kw):
        mine_kw.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "tmp"}

    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_mine", fake_mine,
    )
    rc = cli.main([
        "mine", "team", "--start", "20220101", "--end", "20231231",
        "--no-auto-lift", "--lift-se-mult", "1.5",
    ])
    assert rc == 0
    assert mine_kw.get("auto_lift") is False
    assert mine_kw.get("lift_se_mult") == 1.5

    # 段2：真 run_team_mine → run_team_agent 同名转发（段1 patch 了 run_team_mine，
    # 这里必须先撤销再取真函数，否则调到 fake_mine 上）
    monkeypatch.undo()
    import factorzen.pipelines.factor_mine_team as pmt

    agent_kw: dict = {}

    def recording_agent(*a, **kw):
        agent_kw.update(kw)
        from factorzen.agents.team_orchestrator import TeamResult
        return TeamResult(state=AgentState(seed=0), candidates=[], n_trials=0)

    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine_team.run_team_agent", recording_agent,
    )
    # 测试封闭性：CI 无 FACTORZEN_LLM_*——注入 llm_fn 绕开配置加载,并主动清 env
    # 证明不依赖本地 .env（此前本地绿/CI 红的正是这条泄漏）。
    for _k in [k for k in os.environ if k.startswith("FACTORZEN_LLM")]:
        monkeypatch.delenv(_k, raising=False)
    pmt.run_team_mine(
        fake_daily, n_rounds=1, seed=1, index_path=str(tmp_path / "x.jsonl"),
        out_dir=str(tmp_path / "mt"), export=False,
        llm_fn=lambda messages: "{}",
        auto_lift=False, lift_se_mult=1.5,
    )
    assert agent_kw.get("auto_lift") is False
    assert agent_kw.get("lift_se_mult") == 1.5

def test_cli_auto_lift_default_is_on():
    from factorzen.cli.main import build_parser

    p = build_parser()
    args = p.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20231231",
    ])
    assert getattr(args, "no_auto_lift", False) is False
    assert args.lift_se_mult == 1.0



def test_auto_lift_false_skips_expensive_path(monkeypatch):
    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily = _mock_daily(n_days=60)
    holdout = daily.tail(30)
    called = {"g": 0}

    monkeypatch.setattr(
        "factorzen.discovery.lift_test.run_group_lift",
        lambda *a, **k: called.__setitem__("g", called["g"] + 1) or {},
    )
    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp", seed=0, auto_lift=False,
        horizon=1,
    )
    assert meta["n_lift_queue"] == 1
    assert called["g"] == 0
    assert meta["n_lift_evaluated"] == 0


def test_session_end_auto_lift_uses_explicit_horizon(monkeypatch):
    """P5：auto-lift 必须用传入的 mining horizon，禁止硬编码 DEFAULT_HORIZON=5。

    旧实现 make_lift_context(..., horizon=DEFAULT_HORIZON) → meta/ctx.horizon==5。
    """
    state = _state_with_lift_queue(["ts_mean(close, 5)"])
    daily, holdout, mat = _holdout_and_mat()
    captured: dict = {}

    def fake_group(*a, **k):
        captured["group_ctx"] = k.get("ctx")
        return {
            "lift": 0.01, "lift_se": 0.001, "error": None,
            "n_candidates": 1, "expressions": ["ts_mean(close, 5)"],
        }

    def fake_per(*a, **k):
        captured["per_ctx"] = k.get("ctx")
        return [{
            "expression": "ts_mean(close, 5)",
            "lift": 0.008, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
        }]

    def fake_upsert(rows, *, market, **kw):
        captured["upsert_meta"] = kw.get("meta") or {}
        return {"added_active": 1, "added_probation": 0, "rejected": 0, "errors": []}

    monkeypatch.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
    monkeypatch.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        fake_upsert, raising=False,
    )

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp/lib", seed=1,
        auto_lift=True, lift_se_mult=1.0,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        ret_df=_panel(100).rename({"factor_value": "ret"}),
        horizon=1,
    )
    assert meta.get("horizon") == 1, f"期望 mining horizon=1，got {meta.get('horizon')}"
    assert captured["group_ctx"].horizon == 1
    assert captured["per_ctx"].horizon == 1
    assert captured["upsert_meta"].get("horizon") == 1

    # 另一组：horizon=3 不走默认 5
    meta3 = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
        market="ashare", library_root="/tmp/lib", seed=1,
        auto_lift=True, lift_se_mult=1.0,
        materialize_candidate=mat,
        active_factor_dfs={"base": _panel(100)},
        ret_df=_panel(100).rename({"factor_value": "ret"}),
        horizon=3,
    )
    assert meta3.get("horizon") == 3


def _lift_baseline_spy_meta() -> dict:
    return {
        "n_lift_queue": 0, "lift_group": None, "lift_results": [],
        "lift_admissions": {"added_active": 0, "added_probation": 0},
        "n_lift_evaluated": 0, "lift_dropped_coverage": [],
        "lift_error": None,
    }


def test_lift_baseline_reuses_session_pool_when_active_set_unchanged(
    monkeypatch, tmp_path: Path,
):
    """库 active 集与 session lib_pool 键集一致 → 钩子收到 lib_pool（免重物化）。"""
    from tests.daily.test_factor_library import _write_lib

    lib_root = tmp_path / "lib"
    _write_lib(lib_root, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
    ])

    captured: dict = {}

    def spy_hook(*a, **k):
        captured.update(k)
        return _lift_baseline_spy_meta()

    monkeypatch.setattr(
        "factorzen.agents.team_orchestrator._session_end_auto_lift", spy_hook,
    )

    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 10
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    run_team_agent(
        _mock_daily(), fn, n_rounds=1, seed=1,
        index_path=str(tmp_path / "e.jsonl"), heal_rounds=0,
        auto_lift=True, update_library=False,
        library_root=str(lib_root),
    )
    baseline = captured.get("active_factor_dfs")
    assert baseline is not None, "库 active 集未变时应复用 session lib_pool"
    assert set(baseline.keys()) == {"rank(close)"}


def test_lift_baseline_reuses_even_with_unmaterializable_active(
    monkeypatch, tmp_path: Path,
):
    """库含无法物化的 active(记录 87→物化 84 类比)且文件未变 → 仍复用。

    判据是库文件 hash 而非记录级键集:lift 自建走同函数同库,skip 相同,
    自建结果 ≡ lib_pool。键集比较在真实库上恒不成立(v25 探针实证)。
    """
    from tests.daily.test_factor_library import _write_lib

    lib_root = tmp_path / "lib"
    _write_lib(lib_root, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "rank(bogus_leaf_xyz)", "market": "ashare",
         "status": "active", "ic_train": 0.04},
    ])

    captured: dict = {}

    def spy_hook(*a, **k):
        captured.update(k)
        return _lift_baseline_spy_meta()

    monkeypatch.setattr(
        "factorzen.agents.team_orchestrator._session_end_auto_lift", spy_hook,
    )

    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 10
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    run_team_agent(
        _mock_daily(), fn, n_rounds=1, seed=1,
        index_path=str(tmp_path / "e.jsonl"), heal_rounds=0,
        auto_lift=True, update_library=False,
        library_root=str(lib_root),
    )
    baseline = captured.get("active_factor_dfs")
    assert baseline is not None, "库文件未变时应复用(skip 因子 lift 自建同样 skip)"
    assert set(baseline.keys()) == {"rank(close)"}


def test_lift_baseline_rebuilds_when_library_file_changed(
    monkeypatch, tmp_path: Path,
):
    """本 session upsert 改了库文件(hash 变)→ 钩子收到 None(基线须含新 active)。"""
    from tests.daily.test_factor_library import _write_lib

    lib_root = tmp_path / "lib"
    _write_lib(lib_root, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
    ])

    captured: dict = {}

    def spy_hook(*a, **k):
        captured.update(k)
        return _lift_baseline_spy_meta()

    def fake_upsert(*a, **k):
        # 模拟收尾 upsert 新增一条 active(改库文件内容 → hash 变)
        lib_path = lib_root / "ashare.jsonl"
        with lib_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "expression": "rank(vol)", "market": "ashare",
                "status": "active", "ic_train": 0.06,
            }, ensure_ascii=False) + "\n")

    monkeypatch.setattr(
        "factorzen.agents.team_orchestrator._session_end_auto_lift", spy_hook,
    )
    monkeypatch.setattr(
        "factorzen.agents.team_orchestrator._library_upsert_team", fake_upsert,
    )

    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 10
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    run_team_agent(
        _mock_daily(), fn, n_rounds=1, seed=1,
        index_path=str(tmp_path / "e.jsonl"), heal_rounds=0,
        auto_lift=True, update_library=True,
        library_root=str(lib_root),
    )
    assert captured.get("active_factor_dfs") is None, (
        "库文件已变(upsert 新增 active)时必须回退 lift 自建基线"
    )
