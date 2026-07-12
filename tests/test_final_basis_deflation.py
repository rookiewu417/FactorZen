"""收尾复核：候选的 DSR 必须按**全 session 的 N** 定，而非「它被找到那一轮的 N」。

`node_guardrails` 每轮调用，`basis` 建自截至当轮的累积 attempts。于是 round 0 的候选
用 N@轮=3 定 p、round 5 的候选用 N=18 定 p —— **门槛取决于候选碰巧在第几轮被找到**。
但候选集是从全部 18 次试验里选出来的，多重检验记账必须覆盖整个搜索。

真实实证（`team_51_6r`）：round-0 候选记录 `p=0.0011`，按最终 N=18 复算是 `p=0.0212`，
差 19 倍。且 manifest 报 `n_trials=18` —— **拿着 manifest 复算不出产物里的 p 值**，
违反「一切产物可复现」铁律。

修法不是「每轮重跑护栏」（那会让 N 三角和 over-count），而是收尾时用最终 basis
统一重算一次 DSR 并复核 `passed`。`holdout_ic`/CI/`ic_train` 与 N 无关，无需重跑。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_finalize_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue

_N_OBS = 303


def _mk_daily(n_days: int = 300, n_stocks: int = 40, seed: int = 7) -> pl.DataFrame:
    """≥40 只：`_MIN_CROSS_SAMPLES=30` 会把不足 30 只的截面整天丢掉，IC 序列变空。"""
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        px = rng.uniform(8, 15)
        for dd in days:
            px = float(max(px * (1 + rng.standard_normal() * 0.02), 0.1))
            vol = float(abs(rng.standard_normal()) * 1e6 + 1e5)
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98, "pre_close": px,
                         "close_adj": px, "open_adj": px * 0.99,
                         "high_adj": px * 1.01, "low_adj": px * 0.98,
                         "vol": vol, "amount": px * vol})
    return pl.DataFrame(rows)


def _attempt(it: int, ir: float, expr: str) -> AttemptRecord:
    return AttemptRecord(
        iteration=it, hypothesis="h", expression=expr, compile_ok=True,
        ic_train=ir / 10.0, passed_guardrails=False, critic_verdict=None, error=None,
        ir_train=ir, turnover=0.3, n_train=_N_OBS,
    )


def _candidate(ir: float, expr: str) -> dict:
    """一个已被早轮护栏放行的候选：holdout 同号、CI 方向正确，只有 DSR 依赖 N。"""
    return {"expression": expr, "hypothesis": "h", "ic_train": ir / 10.0, "ir_train": ir,
            "turnover": 0.3, "holdout_ic": 0.05, "holdout_ir": 0.5,
            "ic_ci_low": 0.01, "ic_ci_high": 0.09, "n_train": _N_OBS,
            "dsr": 0.99, "dsr_pvalue": 0.001}


_CAND_EXPR = "rank(neg(pb))"


def _state_with_pool(cand_ir: float, pool_irs: list[float]) -> AgentState:
    state = AgentState(seed=1)
    for i, ir in enumerate(pool_irs):
        state.attempts.append(_attempt(i // 3, ir, f"rank(neg(ts_min(low, {5 + i})))"))
    a = _attempt(0, cand_ir, _CAND_EXPR)
    a.passed_guardrails = True          # 早轮护栏已放行（事实，按当轮 N 计）
    state.attempts.append(a)
    state.candidates.append(_candidate(cand_ir, _CAND_EXPR))
    return state


# ── 核心：早轮候选必须被最终 N 重新审判 ──────────────────────────────────


def test_early_round_candidate_is_rejudged_against_final_n():
    """IR 落在分歧带：小 N 下过关、最终 N 下不显著 ⇒ 必须被剔除。"""
    pool = [0.02, -0.05, 0.08, 0.11, -0.09, 0.03, 0.06, -0.02, 0.12,
            0.01, -0.07, 0.09, 0.04, -0.03, 0.10, 0.05, -0.06]
    cand_ir = 0.172
    state = _state_with_pool(cand_ir, pool)

    final_basis = DeflationBasis.from_ir_pool(
        [a.ir_train for a in state.attempts if a.compile_ok], two_sided=True)
    p_final = deflated_pvalue(cand_ir, final_basis, _N_OBS)[1]
    assert p_final >= 0.05, (
        f"测试前提：该 IR 在最终 N={final_basis.n_trials} 下必须不显著（实得 p={p_final:.4f}），"
        "否则本测试没有判别力"
    )

    node_finalize_guardrails(state, gate="strict")  # N 惩罚是 strict 专属机制

    assert state.candidates == [], f"最终 N 下 p={p_final:.4f} ≥ 0.05，候选应被剔除"


def test_significant_candidate_survives_final_rejudgement():
    """反向断言：真正显著的候选不该被误杀。没有这条，「无脑清空 candidates」也能过上一个测试。"""
    pool = [0.02, -0.05, 0.08, 0.11, -0.09, 0.03, 0.06, -0.02, 0.12]
    state = _state_with_pool(0.45, pool)

    node_finalize_guardrails(state)

    assert len(state.candidates) == 1
    assert state.candidates[0]["ir_train"] == pytest.approx(0.45)


def test_surviving_candidate_p_equals_final_basis_recipe():
    """存活候选的 dsr_pvalue 必须由最终 basis 逐位算出——否则 manifest 与产物不自洽。"""
    pool = [0.02, -0.05, 0.08, 0.11, -0.09, 0.03]
    state = _state_with_pool(0.45, pool)

    basis = node_finalize_guardrails(state)

    c = state.candidates[0]
    want = deflated_pvalue(c["ir_train"], basis, c["n_train"])[1]
    assert c["dsr_pvalue"] == pytest.approx(want, abs=1e-12)


def test_demoted_candidate_syncs_the_attempt_fact():
    """`passed_guardrails` 是「过了定量护栏」这个事实。最终 N 说没过，事实就得改。

    不同步的话，Librarian 会把它当「已验证有效」写进长期记忆。
    """
    pool = [0.02, -0.05, 0.08, 0.11, -0.09, 0.03, 0.06, -0.02, 0.12,
            0.01, -0.07, 0.09, 0.04, -0.03, 0.10, 0.05, -0.06]
    state = _state_with_pool(0.172, pool)
    cand_expr = state.candidates[0]["expression"]

    node_finalize_guardrails(state, gate="strict")  # N 惩罚是 strict 专属机制

    a = next(a for a in state.attempts if a.expression == cand_expr)
    assert a.passed_guardrails is False, "被最终 N 否掉的候选，其 passed_guardrails 必须回落"


def test_final_basis_counts_unique_attempts_not_round_sums():
    """N = 唯一评估过的表达式数，不是逐轮 len(passed) 的三角和。"""
    pool = [0.02, -0.05, 0.08, 0.11, -0.09, 0.03]
    state = _state_with_pool(0.45, pool)
    n_compile_ok = sum(1 for a in state.attempts if a.compile_ok)

    basis = node_finalize_guardrails(state)

    assert basis.n_trials == n_compile_ok == 7
    assert basis.effective_trials == 14, "Agent 路径双边 ⇒ effective = 2N"


def test_dead_expressions_do_not_inflate_final_n():
    """死表达式（ir_train=None）不得计入 N——它们没有产生可比较的 IR。"""
    pool = [0.02, -0.05, 0.08]
    state = _state_with_pool(0.45, pool)
    dead = _attempt(1, 0.0, "rank(dead)")
    dead.ir_train = None
    dead.ic_train = None
    state.attempts.append(dead)

    basis = node_finalize_guardrails(state)

    assert basis.n_trials == 4, "3 个池成员 + 1 个候选；死表达式不计入"


# ── 可复现：光靠 manifest 就能复算出产物里的 p ────────────────────────────


def test_candidate_carries_every_field_needed_to_recompute_p():
    """`n_train` / `ic_ci_low` / `ic_ci_high` 必须落进候选，否则拿 manifest 复算不出 p。

    反解真实 run 时正是被这个卡住：候选里没有 n_train，只能回连 attempts 才算得出。
    """
    pool = [0.02, -0.05, 0.08]
    state = _state_with_pool(0.45, pool)
    node_finalize_guardrails(state)

    c = state.candidates[0]
    for key in ("n_train", "ic_ci_low", "ic_ci_high", "ir_train", "dsr_pvalue"):
        assert key in c, f"候选缺字段 {key}，manifest 无法自证 p 值"


def test_real_node_guardrails_records_ci_and_n_train(monkeypatch):
    """字段得由真实 `node_guardrails` 写入，不能只在 finalize 里补。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.multiple_testing import TrialLedger

    monkeypatch.setattr("factorzen.validation.holdout.holdout_ic",
                        lambda fdf, hdf: (0.05, 0.5, (0.01, 0.09)))
    monkeypatch.setattr("factorzen.discovery.scoring.max_correlation", lambda fdf, pool: 0.0)

    daily = _mk_daily()
    state = AgentState(seed=1)
    for i, ir in enumerate([0.45, 0.1048, -0.1285]):
        state.attempts.append(_attempt(0, ir, f"rank(neg(ts_min(low, {5 + i})))"))

    node_guardrails(state, daily=daily, holdout_df=daily, bundle=DataBundle.build(daily),
                    ledger=TrialLedger(), top_k=5)

    assert state.candidates, "IR=0.45 应过关，否则本测试失去判别力"
    c = state.candidates[0]
    assert c["n_train"] == _N_OBS
    assert c["ic_ci_low"] == pytest.approx(0.01)
    assert c["ic_ci_high"] == pytest.approx(0.09)


# ── manifest 自证：光靠产物就能复算出 p ────────────────────────────────────


def test_manifest_alone_reproduces_every_candidate_pvalue(tmp_path):
    """铁律「一切产物可复现」：不回连任何外部状态，只读 manifest 就能复算出 dsr_pvalue。

    修复前做不到——`sharpe_variance` 没落盘，且候选里没有 `n_train`。反解真实 run 时
    只能回连 attempts 才凑齐入参。
    """
    import json

    from factorzen.agents.manifest import write_session_manifest
    from factorzen.agents.orchestrator import AgentResult

    pool = [0.02, -0.05, 0.08, 0.11]
    state = _state_with_pool(0.45, pool)
    basis = node_finalize_guardrails(state)
    result = AgentResult(state=state, candidates=state.candidates, n_trials=basis.n_trials,
                         sharpe_variance=basis.sharpe_variance)
    path = write_session_manifest(result, out_dir=str(tmp_path), run_id="r", params={})

    m = json.loads(path.read_text())
    assert m["sharpe_variance"] is not None, "缺 sharpe_variance → 复算不出 p"
    assert m["deflation_two_sided"] is True

    recovered = DeflationBasis(n_trials=m["n_trials"], sharpe_variance=m["sharpe_variance"],
                               two_sided=m["deflation_two_sided"])
    assert m["candidates"], "本测试需要至少一个候选"
    for c in m["candidates"]:
        want = deflated_pvalue(c["ir_train"], recovered, c["n_train"])[1]
        assert c["dsr_pvalue"] == pytest.approx(want, abs=1e-12), (
            f"manifest 自证失败：{c['expression']} 记录 {c['dsr_pvalue']}，复算 {want}"
        )


def test_partial_checkpoint_records_null_sharpe_variance(tmp_path):
    """中途快照没有最终 basis —— 写 null，而不是写一个看似可用的假值。"""
    import json

    from factorzen.agents.manifest import write_session_manifest
    from factorzen.agents.orchestrator import AgentResult

    state = _state_with_pool(0.45, [0.02, -0.05])
    result = AgentResult(state=state, candidates=state.candidates, n_trials=3)  # 未传 basis
    path = write_session_manifest(result, out_dir=str(tmp_path), run_id="r", params={},
                                  partial=True)

    m = json.loads(path.read_text())
    assert m["partial"] is True
    assert m["sharpe_variance"] is None


# ── 接线守卫：能力实现了 ≠ 编排器真的调了它 ──────────────────────────────
#
# 变异实证：把 orchestrator 里的 `node_finalize_guardrails(...)` 换成一个假 basis，
# 上面 10 个测试**全部照绿**——它们都在直接调这个函数。能力层与接线层的漂移，
# 只有从编排器最外层出发的测试才抓得住。


def _mk_signal_daily(n_days: int = 300, n_stocks: int = 40, seed: int = 11):
    """带真实截面 alpha 的合成数据——否则 DSR 关卡拦住一切，接线测试拿不到候选。

    每只股票一个持久的 `alpha_i`：既驱动次日收益，也体现在成交量上。
    于是 `ts_mean(vol, k)` 的截面秩 ≈ alpha 的秩 ⇒ 正 IC。
    """
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    alphas = rng.standard_normal(n_stocks)
    rows = []
    for i, code in enumerate([f"{600000 + i:06d}.SH" for i in range(n_stocks)]):
        px, a = 10.0, float(alphas[i])
        for dd in days:
            px = float(max(px * (1 + 0.004 * a + 0.02 * rng.standard_normal()), 0.1))
            vol = float(1e5 * np.exp(a) * (1 + 0.1 * abs(rng.standard_normal())))
            rows.append({"trade_date": dd, "ts_code": code, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98, "pre_close": px,
                         "close_adj": px, "open_adj": px * 0.99,
                         "high_adj": px * 1.01, "low_adj": px * 0.98,
                         "vol": vol, "amount": px * vol})
    return pl.DataFrame(rows)


def _fake_llm():
    """proposal → semantic → critic；每轮表达式不同以避开去重。"""
    import json
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "consistent" in system:
            return json.dumps({"consistent": True, "reason": "ok"})
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        st["round"] += 1
        return json.dumps({"hypothesis": f"h{st['round']}",
                           "expressions": [f"ts_mean(vol,{4 + st['round']})"],
                           "rationale": "r"})
    return fn


def test_run_llm_agent_actually_finalizes(monkeypatch):
    """单 Agent 编排器返回的每个候选，其 p 必须由**最终** basis 算出，且 basis 已落进 result。"""
    from factorzen.agents.orchestrator import run_llm_agent

    monkeypatch.setattr("factorzen.validation.holdout.holdout_ic",
                        lambda fdf, hdf: (0.05, 0.5, (0.01, 0.09)))
    monkeypatch.setattr("factorzen.discovery.scoring.max_correlation", lambda fdf, pool: 0.0)

    res = run_llm_agent(_mk_signal_daily(), _fake_llm(), n_rounds=3, seed=1,
                        heal_rounds=0)

    assert res.sharpe_variance == res.sharpe_variance, "sharpe_variance 未落进 AgentResult（nan）"
    assert res.candidates, "本测试需要至少一个候选，否则无判别力"

    pool = [a.ir_train for a in res.state.attempts if a.compile_ok]
    final = DeflationBasis.from_ir_pool(pool, two_sided=True)
    assert res.sharpe_variance == pytest.approx(final.sharpe_variance)
    for c in res.candidates:
        want = deflated_pvalue(c["ir_train"], final, c["n_train"])[1]
        assert c["dsr_pvalue"] == pytest.approx(want, abs=1e-12), (
            "候选的 p 不是最终 basis 算的 —— run_llm_agent 没有调 node_finalize_guardrails"
        )


def test_run_team_agent_actually_finalizes(tmp_path, monkeypatch):
    """team 编排器同上。两条路径各自接线，缺一不可（双路径登记簿）。"""
    import json

    from factorzen.agents.team_orchestrator import run_team_agent

    monkeypatch.setattr("factorzen.validation.holdout.holdout_ic",
                        lambda fdf, hdf: (0.05, 0.5, (0.01, 0.09)))
    monkeypatch.setattr("factorzen.discovery.scoring.max_correlation", lambda fdf, pool: 0.0)

    st = {"k": -1}

    def fn(_m):
        st["k"] += 1
        r = st["k"] % 3
        if r == 0:
            return json.dumps({"hypotheses": [f"h{st['k']}"]})
        if r == 1:
            return json.dumps({"expressions": [f"ts_mean(vol,{4 + st['k']})"]})
        return json.dumps({"verdict": "keep", "reason": "ok"})

    res = run_team_agent(_mk_signal_daily(), fn, n_rounds=3, seed=1,
                         index_path=str(tmp_path / "e.jsonl"), heal_rounds=0)

    assert res.sharpe_variance == res.sharpe_variance, "sharpe_variance 未落进 TeamResult（nan）"
    assert res.candidates, "本测试需要至少一个候选，否则无判别力"

    pool = [a.ir_train for a in res.state.attempts if a.compile_ok]
    final = DeflationBasis.from_ir_pool(pool, two_sided=True)
    for c in res.candidates:
        want = deflated_pvalue(c["ir_train"], final, c["n_train"])[1]
        assert c["dsr_pvalue"] == pytest.approx(want, abs=1e-12), (
            "候选的 p 不是最终 basis 算的 —— run_team_agent 没有调 node_finalize_guardrails"
        )


def test_pbo_is_recomputed_after_demotion(caplog):
    """候选集变了 → PBO 必须跟着重算，否则 manifest 里的 pbo 描述的是另一个池。

    必须留下 **≥1 个存活候选**：`[f(c) for c in []]` 不执行循环体，候选全剔时
    连 `_node_to_factor_df` 这个名字都不会被查找——第一版测试正是这样，
    把「未定义名」的变异体判成了绿。（该名字只在 `node_guardrails` 内局部导入过；
    lint 抓到了它在 finalize 里未定义，测试没有。）
    """
    import logging

    from factorzen.discovery.scoring import DataBundle

    pool = [0.02, -0.05, 0.08, 0.11, -0.09, 0.03, 0.06, -0.02, 0.12,
            0.01, -0.07, 0.09, 0.04, -0.03, 0.10, 0.05, -0.06]
    # 表达式必须在 _mk_daily 的列上可求值（无 pb/pe_ttm），否则求值异常会被同一个
    # except 吞掉，测试就分不清「NameError」与「列缺失」了。
    specs = [(0.172, "rank(neg(close))"),      # 分歧带 → 应被剔除
             (0.45, "ts_mean(vol, 5)"),        # 强显著 → 存活
             (0.50, "ts_mean(vol, 10)")]       # 强显著 → 存活
    state = AgentState(seed=1)
    for i, ir in enumerate(pool):
        state.attempts.append(_attempt(i // 3, ir, f"rank(neg(ts_min(low, {5 + i})))"))
    for ir, expr in specs:
        a = _attempt(0, ir, expr)
        a.passed_guardrails = True
        state.attempts.append(a)
        state.candidates.append(_candidate(ir, expr))
    state.pbo = 0.42                            # 旧池（3 个候选）的 PBO
    daily = _mk_daily()

    with caplog.at_level(logging.WARNING, logger="factorzen.agents.nodes"):
        node_finalize_guardrails(state, gate="strict",  # N 惩罚是 strict 专属机制
                                 daily=daily, bundle=DataBundle.build(daily))

    assert len(state.candidates) == 2, (
        f"前提：应剔除 1 个、留下 2 个（实得 {len(state.candidates)}）"
    )
    assert "收尾 PBO 重算失败" not in caplog.text, f"PBO 重算路径抛异常被吞：{caplog.text}"
    assert state.pbo == state.pbo, "pbo 是 nan —— 重算路径没能算出真值"
    assert state.pbo != 0.42, "候选集变了，pbo 仍是旧池的值"
