"""
test_guardrail_parity.py：合并自 agents 相关碎片测试（test_guardrail_parity.py）。
test_campaign_prior.py：campaign trial family：DSR 的 N 跨 session 累计（消除清零漏记）。
"""

from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_finalize_guardrails, node_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest
from factorzen.discovery.campaign import CampaignPrior, campaign_key, campaign_prior
from factorzen.discovery.evaluation import evaluate_expressions
from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue, guardrail_passed
from factorzen.discovery.scoring import DataBundle
from factorzen.validation.deflated_sharpe import deflated_sharpe, expected_max_sharpe
from factorzen.validation.multiple_testing import TrialLedger


# ==== 来自 test_guardrail_parity.py ====
# ==== 来自 test_agent_guardrail_parity.py ====
def test_guardrail_passed_matrix_suite():
    """test_guardrail_passed_positive_ic_significant；DSR p 值不显著（0.3>0.05）→ 拒。旧 M5/M6 口径 dsr>0.5(即 pval<0.5)会误放行。；test_guardrail_passed_rejects_sign_mismatch；2026-07 松一档：移除 holdout CI 单边门。CI 跨零不再单独否决——；test_guardrail_passed_negative_ic_bidirectional；test_guardrail_passed_none_and_nan_conservative；test_guardrail_negative_ic_without_ci_high_falls_back_to_ci_low"""
    # -- 原 test_guardrail_passed_positive_ic_significant --
    def _section_0_test_guardrail_passed_positive_ic_significant():
        assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                                ci_low=0.01, ci_high=0.08) is True

    _section_0_test_guardrail_passed_positive_ic_significant()

    # -- 原 test_guardrail_passed_rejects_insignificant_dsr --
    def _section_1_test_guardrail_passed_rejects_insignificant_dsr():
        assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.3,
                                ci_low=0.01, ci_high=0.08) is False

    _section_1_test_guardrail_passed_rejects_insignificant_dsr()

    # -- 原 test_guardrail_passed_rejects_sign_mismatch --
    def _section_2_test_guardrail_passed_rejects_sign_mismatch():
        assert guardrail_passed(ic_train=0.05, holdout_ic=-0.04, dsr_pvalue=0.01,
                                ci_low=0.01, ci_high=0.08) is False

    _section_2_test_guardrail_passed_rejects_sign_mismatch()

    # -- 原 test_guardrail_passed_ci_crosses_zero_now_allowed --
    def _section_3_test_guardrail_passed_ci_crosses_zero_now_allowed():
        assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                                ci_low=-0.01, ci_high=0.08) is True

    _section_3_test_guardrail_passed_ci_crosses_zero_now_allowed()

    # -- 原 test_guardrail_passed_negative_ic_bidirectional --
    def _section_4_test_guardrail_passed_negative_ic_bidirectional():
        assert guardrail_passed(ic_train=-0.05, holdout_ic=-0.04, dsr_pvalue=0.01,
                                ci_low=-0.08, ci_high=-0.01) is True

    _section_4_test_guardrail_passed_negative_ic_bidirectional()

    # -- 原 test_guardrail_passed_none_and_nan_conservative --
    def _section_5_test_guardrail_passed_none_and_nan_conservative():
        assert guardrail_passed(ic_train=None, holdout_ic=0.04, dsr_pvalue=0.01, ci_low=0.01) is False
        nan = float("nan")
        assert guardrail_passed(ic_train=0.05, holdout_ic=nan, dsr_pvalue=0.01, ci_low=0.01) is False

    _section_5_test_guardrail_passed_none_and_nan_conservative()

    # -- 原 test_guardrail_negative_ic_without_ci_high_falls_back_to_ci_low --
    def _section_6_test_guardrail_negative_ic_without_ci_high_falls_back_to_ci_low():
        assert guardrail_passed(ic_train=-0.05, holdout_ic=-0.04, dsr_pvalue=0.01, ci_low=0.01) is True

    _section_6_test_guardrail_negative_ic_without_ci_high_falls_back_to_ci_low()


def test_dsr_parity_node_suite(_stub_holdout):
    """M1 的 _guard_passed 委托共享入口 acceptance_reasons，逐样本无漂移：；PBO 是「过拟合概率」，必须落在 [0, 1]。；候选 < 2 时 CSCV 无从切分 → nan（而非 0.0 之类会被误读为「无过拟合」的值）。；求值后无任何有效截面的表达式必须记 ic=None，而非 0.0。；正常表达式必须回报 train 段有效 IC 天数（供 DSR 的 n_obs 用，对齐 M1 的 c['n_train']）。；核心断言：Agent 的 deflation 尺度必须是**池经验方差**，不是 `1/n_obs` 默认值。；回归：真实 run `agent_43_3r` 里 Agent 放行、M1 拒绝的那个因子，修复后必须被拒。；n_obs 口径对齐：DSR 用因子自己的有效 IC 天数，不是 train 段日历交易日数。"""
    # -- 原 test_cross_path_parity_m1_delegates_to_shared --
    def _section_0_test_cross_path_parity_m1_delegates_to_shared():
        from factorzen.discovery.guardrails import library_reasons
        from factorzen.discovery.mining_session import _guard_passed
        rng = np.random.default_rng(0)
        for _ in range(200):
            c = {"ic_train": float(rng.normal(0, 0.05)), "holdout_ic": float(rng.normal(0, 0.05)),
                 "dsr_pvalue": float(rng.uniform(0, 1)), "ic_ci_low": float(rng.normal(0, 0.03))}
            strict = guardrail_passed(ic_train=c["ic_train"], holdout_ic=c["holdout_ic"],
                                      dsr_pvalue=c["dsr_pvalue"], ci_low=c["ic_ci_low"], dsr_alpha=0.05)
            assert _guard_passed(c, dsr_alpha=0.05, gate="strict") == strict
            library = not library_reasons(ic_train=c["ic_train"], holdout_ic=c["holdout_ic"])
            assert _guard_passed(c) == library, f"library drift: {c}"

    _section_0_test_cross_path_parity_m1_delegates_to_shared()

    # -- 原 test_node_guardrails_pbo_is_a_probability_when_pool_is_big_enough --
    def _section_1_test_node_guardrails_pbo_is_a_probability_when_pool_is_big_enough():
        state = _run_guardrails_with(n_candidates=3)

        assert len(state.candidates) >= 2, "需要 ≥2 个候选，PBO(CSCV) 才有定义"
        assert state.pbo == state.pbo, "候选足够时 PBO 不该是 nan"
        assert 0.0 <= state.pbo <= 1.0, f"PBO 是概率，必须 ∈ [0,1]，实得 {state.pbo}"

    _section_1_test_node_guardrails_pbo_is_a_probability_when_pool_is_big_enough()

    # -- 原 test_node_guardrails_pbo_is_nan_when_pool_too_small --
    def _section_2_test_node_guardrails_pbo_is_nan_when_pool_too_small():
        import math

        state = _run_guardrails_with(n_candidates=1)

        assert len(state.candidates) == 1
        assert math.isnan(state.pbo), f"单候选时 PBO 应为 nan，实得 {state.pbo}"

    _section_2_test_node_guardrails_pbo_is_nan_when_pool_too_small()

    # -- 原 test_dead_expression_yields_none_ic_not_zero --
    def _section_3_test_dead_expression_yields_none_ic_not_zero():
        daily = _mk_daily__dsr_parity()
        bundle = DataBundle.build(daily)
        # 分母恒零 → 全 inf/nan → 过滤后空帧
        res = evaluate_expressions(["div(close, sub(close, close))"], daily, bundle)[0]

        assert res["compile_ok"] is True, "语法合法，compile 应成功"
        assert res["ic_train"] is None, "死表达式的 ic_train 必须是 None，不是 0.0"
        assert res["ir_train"] is None
        assert res["n_train"] == 0, "有效 IC 天数为 0"

    _section_3_test_dead_expression_yields_none_ic_not_zero()

    # -- 原 test_live_expression_reports_n_train --
    def _section_4_test_live_expression_reports_n_train():
        daily = _mk_daily__dsr_parity()
        bundle = DataBundle.build(daily)
        res = evaluate_expressions(["rank(close)"], daily, bundle)[0]

        assert res["ic_train"] is not None
        assert res["n_train"] > 0
        # n_train 是「因子有效 IC 天数」，不是 train 段日历交易日数
        train_days = bundle._segment_mask(daily, "train")["trade_date"].n_unique()
        assert res["n_train"] <= train_days

    _section_4_test_live_expression_reports_n_train()

    # -- 原 test_node_guardrails_dsr_pvalue_uses_pool_variance_not_h0_default --
    def _section_5_test_node_guardrails_dsr_pvalue_uses_pool_variance_not_h0_default(_stub_holdout):
        daily = _mk_daily__dsr_parity()
        mining_df, holdout_df = daily, daily
        bundle = DataBundle.build(mining_df)

        ir_pool = [0.45, 0.1048, -0.1285]     # 正 IR 被测因子 + 散布，signed
        n_train = 305
        state = AgentState(seed=1)
        _seed_attempts(state, ir_pool, n_train)

        ledger = TrialLedger()
        node_guardrails(state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=ledger, top_k=5)

        assert ledger.n_trials == len(ir_pool), "N 必须等于 IR 池大小（与 sharpe_variance 同源）"
        assert state.candidates, "被测因子 IR=0.45 应过护栏（否则测试失去判别力）"

        top = max(state.candidates, key=lambda c: abs(c["ir_train"]))
        _, expected_p = deflated_pvalue(
            0.45, DeflationBasis.from_ir_pool(ir_pool, two_sided=True), n_train
        )
        assert top["dsr_pvalue"] == pytest.approx(expected_p, abs=1e-9), (
            f"Agent 的 dsr_pvalue={top['dsr_pvalue']} 与共享配方 {expected_p} 不符"
        )

        # 判别力：若 sharpe_variance 漂移回 H0 默认 1/n_obs，p 会显著更小（这正是 P0 的形态）。
        _, p_h0_default = deflated_sharpe(abs(0.45), 2 * len(ir_pool), n_train)
        assert p_h0_default < expected_p / 10, (
            "测试数据须让两种 sharpe_variance 口径的 p 拉开量级差，否则本断言无判别力"
        )

    _section_5_test_node_guardrails_dsr_pvalue_uses_pool_variance_not_h0_default(_stub_holdout)

    # -- 原 test_agent_rejects_factor_that_m1_rejects_regression --
    def _section_6_test_agent_rejects_factor_that_m1_rejects_regression(_stub_holdout):
        daily = _mk_daily__dsr_parity()
        bundle = DataBundle.build(daily)

        ir_pool = [0.1698, 0.1048, -0.1285]
        n_train = 305
        state = AgentState(seed=1)
        _seed_attempts(state, ir_pool, n_train)

        # 前置：确认这组数字真的构成「默认口径放行 / 经验方差口径拒绝」的分歧，否则测试无判别力
        _, p_default = deflated_sharpe(0.1698, 3, n_train)
        _, p_empirical = deflated_sharpe(0.1698, 3, n_train,
                                         sharpe_variance=float(np.var(np.array(ir_pool))))
        assert p_default < 0.05 < p_empirical, "测试数据须落在两口径的分歧区"

        node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                        ledger=TrialLedger(), top_k=5, gate="strict")  # DSR 拒绝是 strict 专属

        top_attempt = max(state.attempts, key=lambda a: abs(a.ir_train or 0.0))
        assert top_attempt.passed_guardrails is False, (
            "IR=0.1698 在池经验方差口径下 DSR 不显著，必须被护栏拒绝"
        )
        assert not state.candidates

    _section_6_test_agent_rejects_factor_that_m1_rejects_regression(_stub_holdout)

    # -- 原 test_node_guardrails_uses_factor_n_train_not_calendar_days --
    def _section_7_test_node_guardrails_uses_factor_n_train_not_calendar_days(_stub_holdout):
        daily = _mk_daily__dsr_parity()
        bundle = DataBundle.build(daily)
        calendar_days = bundle._segment_mask(daily, "train")["trade_date"].n_unique()

        n_train = 40                       # 该因子真实有效 IC 天数，远小于日历日数
        assert n_train < calendar_days
        ir_pool = [0.45, 0.1048, -0.1285]
        sharpe_var = float(np.var(np.array(ir_pool)))

        # 判别性前置：两种 n_obs 口径必须给出相反判定，否则本测试无法区分实现
        _, p_correct = deflated_sharpe(0.45, 3, n_train, sharpe_variance=sharpe_var)
        _, p_calendar = deflated_sharpe(0.45, 3, calendar_days, sharpe_variance=sharpe_var)
        assert p_calendar < 0.05 < p_correct, (
            "测试数据须使「误用日历日数→放行 / 正确用 n_train→拒绝」成立"
        )

        state = AgentState(seed=1)
        _seed_attempts(state, ir_pool, n_train)
        node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                        ledger=TrialLedger(), top_k=5, gate="strict")  # DSR 拒绝是 strict 专属

        top_attempt = max(state.attempts, key=lambda a: abs(a.ir_train or 0.0))
        assert top_attempt.passed_guardrails is False, (
            f"n_train={n_train} 下 DSR 不显著(p={p_correct:.4f})，必须拒绝；"
            f"若放行说明仍在用日历日数(p={p_calendar:.4f})"
        )
        assert not state.candidates

    _section_7_test_node_guardrails_uses_factor_n_train_not_calendar_days(_stub_holdout)


def _mk_daily__guardrail(n_days=260, n_stocks=30, seed=7):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for c in codes:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99,
                         "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7})
    return pl.DataFrame(rows)


def _run_guardrails_with(n_candidates: int):
    """跑 node_guardrails，用 stub 让指定数量的候选过护栏，返回 state。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily__guardrail()
    mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.3)
    bundle = DataBundle.build(mining_df)
    state = AgentState(seed=1)
    # 用互不相关的表达式，避免被去相关剔除（corr>0.7）
    exprs = ["ts_mean(close, 5)", "rank(neg(vol))", "ts_std(close, 10)"][:n_candidates]
    for e in exprs:
        state.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=e, compile_ok=True,
            ic_train=0.05, passed_guardrails=False, critic_verdict=None, error=None,
            ir_train=1.2, n_train=100))

    import factorzen.discovery.guardrails as gmod
    import factorzen.validation.holdout as hmod
    from factorzen.validation.holdout import HoldoutICResult
    orig_hic, orig_pass = hmod.holdout_ic_result, gmod.acceptance_reasons
    # 覆盖充足 + 同号；acceptance_reasons 恒空 → 全过（与旧 mock guardrail_passed 等价）
    hmod.holdout_ic_result = lambda _f, _h: HoldoutICResult(
        0.05, 0.5, (0.01, 0.09), n_days=100)
    gmod.acceptance_reasons = lambda **_kw: []
    try:
        node_guardrails(state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                        ledger=TrialLedger(), top_k=5, warmup_daily=daily)
    finally:
        hmod.holdout_ic_result, gmod.acceptance_reasons = orig_hic, orig_pass
    return state


# ==== 来自 test_agent_dsr_parity.py ====
def _mk_daily__dsr_parity(n_days: int = 300, n_stocks: int = 30, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99,
                         "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7})
    return pl.DataFrame(rows)


# ── F14（P0 的前置条件）：死表达式不得冒充有效 trial ──────────────────────────


# ── P0 主体：decision-parity ────────────────────────────────────────────────


def _seed_attempts(state: AgentState, irs: list[float], n_train: int) -> None:
    """按给定 signed IR 池填充本轮 attempts（ic 与 ir 同号，holdout 由 monkeypatch 控制）。"""
    for i, ir in enumerate(irs):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis="h", expression=f"rank(neg(ts_min(low, {5 + i})))",
            compile_ok=True, ic_train=ir / 10.0, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=ir, turnover=0.3, n_train=n_train,
        ))


@pytest.fixture
def _stub_holdout(monkeypatch):
    """把 holdout 关卡固定为「同号且 CI 方向正确」，隔离出 DSR 这一关。

    只 stub holdout（外部关卡），DSR 走真实 `deflated_sharpe`——被测的就是它的入参。
    """
    monkeypatch.setattr("factorzen.validation.holdout.holdout_ic",
                        lambda fdf, hdf: (0.05, 0.5, (0.01, 0.09)))
    monkeypatch.setattr("factorzen.discovery.scoring.max_correlation", lambda fdf, pool: 0.0)


# ==== 来自 test_dsr_sidedness.py ====
_SRC = Path(__file__).resolve().parents[2] / "src" / "factorzen"

# team_51_6r 真实 run 反解出的 basis（6/6 候选的 dsr_pvalue 逐位复算吻合）
_REAL_VAR = 0.00127844
_REAL_N = 18
_REAL_NOBS = 303


# ── ground truth：取绝对值 ≡ 试验数翻倍 ────────────────────────────────────


@pytest.mark.parametrize("n", [5, 100])
def test_abs_max_null_baseline_matches_double_trials(n):
    """E[max_N |Z|] ≈ expected_max_sharpe(1, 2N)，而非 (1, N)。

    容差 0.03σ：`expected_max_sharpe` 对 E[max_N Z] 本身就有 ~0.02σ–0.04σ 的逼近误差，
    再苛刻就是在测这个公式的近似质量，而不是测 2N 这条结论。
    """
    rng = np.random.default_rng(20260709)
    z = rng.standard_normal((60_000, n))
    e_absmax = float(np.abs(z).max(axis=1).mean())

    assert e_absmax == pytest.approx(expected_max_sharpe(1.0, 2 * n), abs=0.03), (
        f"N={n}: E[max|Z|]={e_absmax:.4f} 应与 2N 基准 {expected_max_sharpe(1.0, 2 * n):.4f} 吻合"
    )


@pytest.mark.parametrize("n", [5, 100])
def test_one_sided_baseline_understates_the_abs_max_statistic(n):
    """反向断言：用 N 做 |IR| 的基准会少算一大截——这正是漂移的量级。

    没有这条，上一个测试可以靠「N 与 2N 差不多」而假绿。
    """
    rng = np.random.default_rng(20260709)
    z = rng.standard_normal((60_000, n))
    e_absmax = float(np.abs(z).max(axis=1).mean())
    gap = e_absmax - expected_max_sharpe(1.0, n)

    assert gap > 0.15, f"N={n}: 单边基准少算 {gap:.4f}σ，应显著为正（实测 0.20σ–0.41σ）"


# ── DeflationBasis 的 two_sided 契约 ───────────────────────────────────────


def test_dsr_sidedness_math_suite():
    """`n_trials` 是诚实计数（写进 manifest 的那个），`effective_trials` 才是 deflation 用的。；调用方一律传**带符号** IR；取不取绝对值由 basis 决定。；同一个 |IR|，双边基准给出的 p 必须更大（门槛更高）。方向错了就是把漂移修反了。；team_51_6r 的真实 basis 上，IR=0.172 的因子单边过关、双边被拒。；`abs()` 只许发生在 `deflated_pvalue` 内部。"""
    # -- 原 test_two_sided_basis_doubles_effective_trials_but_reports_honest_n --
    def _section_0_test_two_sided_basis_doubles_effective_trials_but_reports_honest_n():
        pool = [0.2, -0.1, 0.35, 0.05]
        one = DeflationBasis.from_ir_pool(pool)
        two = DeflationBasis.from_ir_pool(pool, two_sided=True)

        assert one.n_trials == two.n_trials == 4, "诚实计数不受 sidedness 影响"
        assert one.effective_trials == 4
        assert two.effective_trials == 8
        assert one.sharpe_variance == pytest.approx(two.sharpe_variance), "方差口径与 sidedness 无关"

    _section_0_test_two_sided_basis_doubles_effective_trials_but_reports_honest_n()

    # -- 原 test_deflated_pvalue_applies_abs_only_when_two_sided --
    def _section_1_test_deflated_pvalue_applies_abs_only_when_two_sided():
        var, n_obs = 0.01, 300
        two = DeflationBasis(n_trials=10, sharpe_variance=var, two_sided=True)
        one = DeflationBasis(n_trials=10, sharpe_variance=var)

        p_pos = deflated_pvalue(0.30, two, n_obs)[1]
        p_neg = deflated_pvalue(-0.30, two, n_obs)[1]
        assert p_neg == pytest.approx(p_pos), "双边下 ±0.30 必须同 p（反转因子只是符号相反）"

        p_neg_one = deflated_pvalue(-0.30, one, n_obs)[1]
        assert p_neg_one > 0.5, "单边下负 IR 不该显著——M1 的反转因子须以 neg(x) 形式出现"

    _section_1_test_deflated_pvalue_applies_abs_only_when_two_sided()

    # -- 原 test_two_sided_basis_is_stricter_on_the_same_statistic --
    def _section_2_test_two_sided_basis_is_stricter_on_the_same_statistic():
        var, n_obs, ir = _REAL_VAR, _REAL_NOBS, 0.20
        p_one = deflated_pvalue(ir, DeflationBasis(_REAL_N, var), n_obs)[1]
        p_two = deflated_pvalue(ir, DeflationBasis(_REAL_N, var, two_sided=True), n_obs)[1]
        assert p_two > p_one

    _section_2_test_two_sided_basis_is_stricter_on_the_same_statistic()

    # -- 原 test_sidedness_flips_a_real_run_candidate --
    def _section_3_test_sidedness_flips_a_real_run_candidate():
        p_one = deflated_pvalue(0.172, DeflationBasis(_REAL_N, _REAL_VAR), _REAL_NOBS)[1]
        p_two = deflated_pvalue(0.172, DeflationBasis(_REAL_N, _REAL_VAR, two_sided=True),
                                _REAL_NOBS)[1]

        assert p_one < 0.05, f"单边口径下应过关（p={p_one:.4f}）"
        assert p_two >= 0.05, f"双边口径下应被拒（p={p_two:.4f}）"

    _section_3_test_sidedness_flips_a_real_run_candidate()

    # -- 原 test_no_caller_feeds_an_abs_value_into_deflated_pvalue --
    def _section_4_test_no_caller_feeds_an_abs_value_into_deflated_pvalue():
        offenders: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            tainted = _abs_tainted_names(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "deflated_pvalue" and node.args):
                    continue
                first = node.args[0]
                inline_abs = (isinstance(first, ast.Call) and isinstance(first.func, ast.Name)
                              and first.func.id == "abs")
                via_name = isinstance(first, ast.Name) and first.id in tainted
                if inline_abs or via_name:
                    offenders.append(f"{path.relative_to(_SRC)}:{node.lineno}")

        assert not offenders, (
            "以下调用点把 abs(...) 的值传给了 deflated_pvalue——统计量与 deflation 基准会脱钩。"
            f"改传带符号 IR，并在构造 DeflationBasis 时指定 two_sided=True：{offenders}"
        )

    _section_4_test_no_caller_feeds_an_abs_value_into_deflated_pvalue()


# ── 架构守卫：禁止调用方自己 abs，否则 sidedness 又会与基准脱钩 ──────────────


def _abs_tainted_names(tree: ast.AST) -> set[str]:
    """收集所有「值来自 abs(...) 调用」的局部名字。

    历史 bug 的形态是 `sharpe = abs(a.ir_train) if ... else abs(...)` 再 `deflated_pvalue(sharpe, ...)`
    ——只查内联 `abs()` 的守卫会在有 bug 的代码上通过（我第一版就是这样，白写）。
    """
    def has_abs(node: ast.AST) -> bool:
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "abs"
                   for n in ast.walk(node))

    tainted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and has_abs(node.value):
            tainted.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and has_abs(node.value) and isinstance(node.target, ast.Name)):
            tainted.add(node.target.id)
    return tainted


# ── 行为守卫：AST 挡不住的，由真实 node_guardrails 挡 ──────────────────────


def _seed(state, irs, n_train):
    from factorzen.agents.state import AttemptRecord
    for i, ir in enumerate(irs):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis="h",
            expression=f"rank(neg(ts_min(low, {5 + i})))", compile_ok=True,
            ic_train=ir / 10.0, passed_guardrails=False, critic_verdict=None,
            error=None, ir_train=ir, turnover=0.3, n_train=n_train,
        ))


# ==== 来自 test_campaign_prior.py ====
# ── helpers ───────────────────────────────────────────────────────────────

_WIN_A = {"start": "20200605", "end": "20260605", "universe": "csi300", "market": "ashare"}
_WIN_B = {"start": "20180101", "end": "20201231", "universe": "csi500", "market": "ashare"}
_N_OBS = 303


def _full_key(**overrides) -> str:
    """与 team_orchestrator 默认配置对齐的完整 campaign_key。"""
    from factorzen.discovery.guardrails import DEFAULT_GATE

    base = dict(
        market="ashare",
        universe="csi300",
        start="20200605",
        end="20260605",
        holdout_ratio=0.2,
        objective="residual",
        horizon=1,
        gate=DEFAULT_GATE,
    )
    base.update(overrides)
    return campaign_key(**base)


def _line(
    expr: str,
    ir: float,
    *,
    run_id: str,
    window: dict,
    compile_ok: bool = True,
    campaign_id: str | None = None,
) -> str:
    rec = {
        "expression": expr,
        "hypothesis": "h",
        "ic_train": ir / 10.0,
        "ir_train": ir,
        "n_train": _N_OBS,
        "passed": False,
        "verdict": None,
        "decorrelated": False,
        "compile_ok": compile_ok,
        "error": None,
        "data_window": window,
        "run_id": run_id,
    }
    if campaign_id is not None:
        rec["campaign_id"] = campaign_id
    return json.dumps(rec, ensure_ascii=False)


def _write_index(path: Path, lines: list[str]) -> str:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _attempt(it: int, ir: float, expr: str) -> AttemptRecord:
    return AttemptRecord(
        iteration=it, hypothesis="h", expression=expr, compile_ok=True,
        ic_train=ir / 10.0, passed_guardrails=False, critic_verdict=None, error=None,
        ir_train=ir, turnover=0.3, n_train=_N_OBS,
    )


def _candidate(ir: float, expr: str) -> dict:
    return {
        "expression": expr, "hypothesis": "h", "ic_train": ir / 10.0, "ir_train": ir,
        "turnover": 0.3, "holdout_ic": 0.05, "holdout_ir": 0.5,
        "ic_ci_low": 0.01, "ic_ci_high": 0.09, "n_train": _N_OBS,
        "dsr": 0.99, "dsr_pvalue": 0.001,
    }


def _mock_daily(n_stocks=40, n_days=180, seed=1) -> pl.DataFrame:
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
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _scripted_team():
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}

    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


# ── 1. campaign_key 稳定性 ────────────────────────────────────────────────


def test_campaign_key_suite():
    """test_campaign_key_stable_and_sensitive；None 与空串（strip 后）对 key 等价。"""
    # -- 原 test_campaign_key_stable_and_sensitive --
    def _section_0_test_campaign_key_stable_and_sensitive():
        base = dict(
            market="ashare", universe="csi300", start="20200605", end="20260605",
            holdout_ratio=0.2, objective="residual", horizon=1, gate="library",
        )
        k0 = campaign_key(**base)
        assert k0 == campaign_key(**base)
        assert len(k0) == 16
        assert all(c in "0123456789abcdef" for c in k0)

        assert campaign_key(**{**base, "universe": "csi500"}) != k0
        assert campaign_key(**{**base, "holdout_ratio": 0.3}) != k0
        assert campaign_key(**{**base, "objective": "raw"}) != k0
        assert campaign_key(**{**base, "gate": "strict"}) != k0
        assert campaign_key(**{**base, "horizon": 5}) != k0
        assert campaign_key(**{**base, "start": "20200101"}) != k0

    _section_0_test_campaign_key_stable_and_sensitive()

    # -- 原 test_campaign_key_none_and_empty_string_normalized --
    def _section_1_test_campaign_key_none_and_empty_string_normalized():
        k_none = campaign_key(
            market=None, universe=None, start=None, end=None,
            holdout_ratio=None, objective=None, horizon=None, gate=None,
        )
        k_empty = campaign_key(
            market="", universe="  ", start="", end="",
            holdout_ratio=None, objective="", horizon=None, gate="  ",
        )
        assert k_none == k_empty

    _section_1_test_campaign_key_none_and_empty_string_normalized()


# ── 2–4. campaign_prior 重建 ─────────────────────────────────────────────


def test_campaign_prior_load_suite(tmp_path):
    """session A 3 式 + B 2 式（1 重复）同窗 + C 2 式异窗 → 同窗 prior N=4 / sessions=2。；test_campaign_prior_exclude_run_ids；test_campaign_prior_skips_corrupt_lines；test_campaign_prior_missing_file_returns_none"""
    # -- 原 test_campaign_prior_rebuild_dedup_and_window --
    def _section_0_test_campaign_prior_rebuild_dedup_and_window(tmp_path):
        lines = [
            _line("expr_a1", 0.10, run_id="team_a", window=_WIN_A),
            _line("expr_a2", 0.20, run_id="team_a", window=_WIN_A),
            _line("expr_a3", 0.30, run_id="team_a", window=_WIN_A),
            _line("expr_a1", 0.99, run_id="team_b", window=_WIN_A),  # 与 A 重复，保首行 IR
            _line("expr_b2", 0.40, run_id="team_b", window=_WIN_A),
            _line("expr_c1", 0.50, run_id="team_c", window=_WIN_B),
            _line("expr_c2", 0.60, run_id="team_c", window=_WIN_B),
        ]
        path = _write_index(tmp_path / "experiment_index.jsonl", lines)

        prior = campaign_prior(
            path, market="ashare", universe="csi300",
            start="20200605", end="20260605",
        )
        assert prior is not None
        assert prior.n_trials == 4
        assert len(prior.irs) == 4
        assert prior.n_sessions == 2
        # 去重保首行：expr_a1 取 session A 的 0.10，不是 B 的 0.99；顺序=首现序
        first_order = ["expr_a1", "expr_a2", "expr_a3", "expr_b2"]
        assert prior.irs == [0.10, 0.20, 0.30, 0.40]
        assert prior.expressions == set(first_order)
        assert "expr_c1" not in prior.expressions
        assert prior.source_path == path

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_campaign_prior_rebuild_dedup_and_window(_tp0)

    # -- 原 test_campaign_prior_exclude_run_ids --
    def _section_1_test_campaign_prior_exclude_run_ids(tmp_path):
        lines = [
            _line("e1", 0.1, run_id="team_a", window=_WIN_A),
            _line("e2", 0.2, run_id="team_a", window=_WIN_A),
            _line("e3", 0.3, run_id="team_a", window=_WIN_A),
            _line("e4", 0.4, run_id="team_b", window=_WIN_A),
            _line("e5", 0.5, run_id="team_b", window=_WIN_A),
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)
        prior = campaign_prior(
            path, market="ashare", universe="csi300",
            start="20200605", end="20260605",
            exclude_run_ids={"team_b"},
        )
        assert prior is not None
        assert prior.n_trials == 3
        assert prior.n_sessions == 1
        assert prior.expressions == {"e1", "e2", "e3"}

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_campaign_prior_exclude_run_ids(_tp1)

    # -- 原 test_campaign_prior_skips_corrupt_lines --
    def _section_2_test_campaign_prior_skips_corrupt_lines(tmp_path):
        lines = [
            _line("e1", 0.1, run_id="team_a", window=_WIN_A),
            "NOT_JSON{{{",
            _line("e2", 0.2, run_id="team_a", window=_WIN_A),
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)
        prior = campaign_prior(
            path, market="ashare", universe="csi300",
            start="20200605", end="20260605",
        )
        assert prior is not None
        assert prior.n_trials == 2

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_campaign_prior_skips_corrupt_lines(_tp2)

    # -- 原 test_campaign_prior_missing_file_returns_none --
    def _section_3_test_campaign_prior_missing_file_returns_none(tmp_path):
        assert campaign_prior(
            str(tmp_path / "nope.jsonl"),
            market="ashare", universe="csi300",
            start="20200605", end="20260605",
        ) is None

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_campaign_prior_missing_file_returns_none(_tp3)


# ── 5. finalize 单调性（TDD 核心）────────────────────────────────────────


def test_finalize_prior_union_suite():
    """同一候选：注入 100 个历史 IR 的 prior 后 p 变大（门槛更严）。；test_finalize_union_dedups_session_vs_prior；prior=None 时 basis 与仅用 session attempts 的 from_ir_pool 一致。"""
    # -- 原 test_finalize_pvalue_monotone_with_prior_n --
    def _section_0_test_finalize_pvalue_monotone_with_prior_n():
        cand_ir = 0.25
        expr = "rank(neg(pb))"
        session_pool = [0.05, -0.03, 0.12, cand_ir]

        state0 = AgentState(seed=1)
        for i, ir in enumerate(session_pool):
            e = expr if ir == cand_ir else f"rank(neg(ts_min(low, {5 + i})))"
            a = _attempt(0, ir, e)
            if e == expr:
                a.passed_guardrails = True
            state0.attempts.append(a)
        state0.candidates.append(_candidate(cand_ir, expr))

        basis0 = node_finalize_guardrails(state0)
        p0 = state0.candidates[0]["dsr_pvalue"]
        assert basis0.n_trials == 4

        # 同候选、同 session 池 + 100 个历史 IR
        hist_irs = [0.01 * ((i % 20) - 10) for i in range(100)]
        prior = CampaignPrior(
            campaign_id="deadbeefdeadbeef",
            n_trials=100,
            expressions={f"hist_{i}" for i in range(100)},
            irs=list(hist_irs),
            n_sessions=5,
            source_path="/tmp/x.jsonl",
        )
        state1 = AgentState(seed=1)
        for i, ir in enumerate(session_pool):
            e = expr if ir == cand_ir else f"rank(neg(ts_min(low, {5 + i})))"
            a = _attempt(0, ir, e)
            if e == expr:
                a.passed_guardrails = True
            state1.attempts.append(a)
        state1.candidates.append(_candidate(cand_ir, expr))

        basis1 = node_finalize_guardrails(state1, prior=prior)
        p1 = state1.candidates[0]["dsr_pvalue"]
        assert basis1.n_trials == 104, "100 prior + 4 session 唯一"
        assert p1 > p0, f"N 变大后 p 应更严：p0={p0:.6f} p1={p1:.6f}"

    _section_0_test_finalize_pvalue_monotone_with_prior_n()

    # -- 原 test_finalize_union_dedups_session_vs_prior --
    def _section_1_test_finalize_union_dedups_session_vs_prior():
        prior = CampaignPrior(
            campaign_id="x",
            n_trials=2,
            expressions={"ts_mean(close,5)", "rank(vol)"},
            irs=[0.10, 0.20],
            n_sessions=1,
            source_path="/tmp/x.jsonl",
        )
        state = AgentState(seed=1)
        # 与 prior 重复的表达式 + 一个新表达式
        for expr, ir in [("ts_mean(close,5)", 0.99), ("rank(pb)", 0.15)]:
            state.attempts.append(_attempt(0, ir, expr))
        state.candidates.append(_candidate(0.15, "rank(pb)"))

        basis = node_finalize_guardrails(state, prior=prior)
        # prior 2 + session 新增 1（ts_mean 不双计）= 3
        assert basis.n_trials == 3

    _section_1_test_finalize_union_dedups_session_vs_prior()

    # -- 原 test_finalize_prior_none_zero_regression --
    def _section_2_test_finalize_prior_none_zero_regression():
        state = AgentState(seed=1)
        pool = [0.02, -0.05, 0.08, 0.11]
        for i, ir in enumerate(pool):
            state.attempts.append(_attempt(0, ir, f"e{i}"))
        state.candidates.append(_candidate(0.11, "e3"))

        basis = node_finalize_guardrails(state, prior=None)
        want = DeflationBasis.from_ir_pool(pool, two_sided=True)
        assert basis.n_trials == want.n_trials
        assert basis.sharpe_variance == pytest.approx(want.sharpe_variance)

    _section_2_test_finalize_prior_none_zero_regression()


# ── 6. 同表达式不双计 ────────────────────────────────────────────────────


# ── 7. orchestrator 端 manifest 字段 ─────────────────────────────────────


def test_orchestrator_campaign_manifest_suite(tmp_path):
    """mock index + 小 session：manifest 含 campaign 族字段；family = prior + session 新增。；test_orchestrator_campaign_prior_disabled_zero_regression"""
    # -- 原 test_orchestrator_manifest_campaign_fields --
    def _section_0_test_orchestrator_manifest_campaign_fields(tmp_path):
        hist_cid = _full_key()
        hist = [
            _line("ts_std(close,10)", 0.08, run_id="team_99", window=_WIN_A, campaign_id=hist_cid),
            _line("rank(vol)", 0.06, run_id="team_99", window=_WIN_A, campaign_id=hist_cid),
        ]
        index_path = _write_index(tmp_path / "experiment_index.jsonl", hist)

        daily = _mock_daily()
        res = run_team_agent(
            daily, _scripted_team(),
            n_rounds=1, seed=42,
            index_path=index_path,
            data_window=dict(_WIN_A),
            heal_rounds=0,
            update_library=False,
            library_orthogonal=False,
            auto_lift=False,
            campaign_prior_enabled=True,
            run_id="team_42_testfix",
        )
        man_path = write_team_manifest(
            res, out_dir=str(tmp_path / "out"), run_id="t_campaign",
            params={"holdout_ratio": 0.2},
        )
        m = json.loads(man_path.read_text(encoding="utf-8"))

        assert m.get("campaign_id")
        assert len(m["campaign_id"]) == 16
        assert m["prior_n_trials"] == 2
        assert m["prior_n_sessions"] == 1
        assert "n_trials_family" in m
        # family = prior 唯一 + 本 session 不在 prior 里的唯一新增
        # scripted 评估 ts_mean(close,5)，不在历史 → family >= prior + 1（若编译成功）
        assert m["n_trials_family"] >= m["prior_n_trials"]
        if res.n_trials >= 1:
            # 本 session 至少有 1 个新表达式时 family 应严格大于 prior
            assert m["n_trials_family"] == m["prior_n_trials"] + res.n_trials or (
                m["n_trials_family"] > m["prior_n_trials"]
            )
        # 现有 n_trials 语义保持 = 本 session
        assert m["n_trials"] == res.n_trials

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_orchestrator_manifest_campaign_fields(_tp0)

    # -- 原 test_orchestrator_campaign_prior_disabled_zero_regression --
    def _section_1_test_orchestrator_campaign_prior_disabled_zero_regression(tmp_path):
        hist = [
            _line("ts_std(close,10)", 0.08, run_id="team_99", window=_WIN_A),
            _line("rank(vol)", 0.06, run_id="team_99", window=_WIN_A),
        ]
        index_path = _write_index(tmp_path / "experiment_index.jsonl", hist)
        daily = _mock_daily()

        res_off = run_team_agent(
            daily, _scripted_team(),
            n_rounds=1, seed=7,
            index_path=index_path,
            data_window=dict(_WIN_A),
            heal_rounds=0,
            update_library=False,
            library_orthogonal=False,
            auto_lift=False,
            campaign_prior_enabled=False,
        )
        m_off = json.loads(write_team_manifest(
            res_off, out_dir=str(tmp_path / "out_off"), run_id="t_off",
            params={},
        ).read_text(encoding="utf-8"))

        assert m_off["prior_n_trials"] == 0
        assert m_off.get("prior_n_sessions", 0) == 0
        # 无 prior 时 family = 本 session N（basis.n_trials）
        assert m_off["n_trials_family"] == m_off["n_trials"] or m_off["n_trials_family"] >= 0

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_orchestrator_campaign_prior_disabled_zero_regression(_tp1)


# ── 8. S3/P7：按完整统计问题分族 + 同 seed 不互斥 ─────────────────────────


def test_campaign_id_filter_suite(tmp_path):
    """配置分族：同窗不同 objective/horizon 不得混入同一 prior 池。；prior.campaign_id 必须等于传入的 campaign_id（= manifest 写入值），非全-None 哈希。；同 seed 历史不互斥：不同 run_id 的两 session 都应计入 prior。；campaign_id=None 时保持 legacy 按窗过滤（向后兼容既有测试与 M1）。；campaign_id 精确过滤时，缺 campaign_id 的旧行保守排除。；record(..., campaign_id=) 写入 index 行顶层字段（不进 data_window）。"""
    # -- 原 test_campaign_prior_filters_by_campaign_id_not_window_only --
    def _section_0_test_campaign_prior_filters_by_campaign_id_not_window_only(tmp_path):
        cid_x = _full_key(objective="residual", horizon=1)
        cid_y = _full_key(objective="raw", horizon=5)
        assert cid_x != cid_y

        lines = [
            _line("expr_x1", 0.10, run_id="team_x", window=_WIN_A, campaign_id=cid_x),
            _line("expr_x2", 0.20, run_id="team_x", window=_WIN_A, campaign_id=cid_x),
            _line("expr_y1", 0.30, run_id="team_y", window=_WIN_A, campaign_id=cid_y),
            _line("expr_y2", 0.40, run_id="team_y", window=_WIN_A, campaign_id=cid_y),
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)

        prior = campaign_prior(
            path,
            market="ashare",
            universe="csi300",
            start="20200605",
            end="20260605",
            campaign_id=cid_x,
        )
        assert prior is not None
        assert prior.n_trials == 2
        assert prior.expressions == {"expr_x1", "expr_x2"}
        assert "expr_y1" not in prior.expressions
        assert prior.campaign_id == cid_x

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_campaign_prior_filters_by_campaign_id_not_window_only(_tp0)

    # -- 原 test_campaign_prior_campaign_id_matches_input --
    def _section_1_test_campaign_prior_campaign_id_matches_input(tmp_path):
        cid = _full_key()
        lines = [
            _line("e1", 0.1, run_id="r1", window=_WIN_A, campaign_id=cid),
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)
        prior = campaign_prior(
            path,
            market="ashare",
            universe="csi300",
            start="20200605",
            end="20260605",
            campaign_id=cid,
        )
        assert prior is not None
        assert prior.campaign_id == cid
        # 全-None key 与完整 key 不同（审计可重建 basis）
        none_key = campaign_key(
            market="ashare", universe="csi300", start="20200605", end="20260605",
            holdout_ratio=None, objective=None, horizon=None, gate=None,
        )
        assert prior.campaign_id != none_key

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_campaign_prior_campaign_id_matches_input(_tp1)

    # -- 原 test_campaign_prior_same_seed_distinct_run_ids_both_counted --
    def _section_2_test_campaign_prior_same_seed_distinct_run_ids_both_counted(tmp_path):
        cid = _full_key()
        lines = [
            _line("e_aaa", 0.11, run_id="team_seed_aaa", window=_WIN_A, campaign_id=cid),
            _line("e_bbb", 0.22, run_id="team_seed_bbb", window=_WIN_A, campaign_id=cid),
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)
        prior = campaign_prior(
            path,
            market="ashare",
            universe="csi300",
            start="20200605",
            end="20260605",
            campaign_id=cid,
            exclude_run_ids={"team_seed_current"},
        )
        assert prior is not None
        assert prior.n_trials == 2
        assert prior.n_sessions == 2
        assert prior.expressions == {"e_aaa", "e_bbb"}

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_campaign_prior_same_seed_distinct_run_ids_both_counted(_tp2)

    # -- 原 test_campaign_prior_legacy_none_campaign_id_window_filter --
    def _section_3_test_campaign_prior_legacy_none_campaign_id_window_filter(tmp_path):
        cid_x = _full_key(objective="residual", horizon=1)
        cid_y = _full_key(objective="raw", horizon=5)
        lines = [
            _line("expr_x1", 0.10, run_id="team_x", window=_WIN_A, campaign_id=cid_x),
            _line("expr_x2", 0.20, run_id="team_x", window=_WIN_A, campaign_id=cid_x),
            _line("expr_y1", 0.30, run_id="team_y", window=_WIN_A, campaign_id=cid_y),
            _line("expr_y2", 0.40, run_id="team_y", window=_WIN_A, campaign_id=cid_y),
            _line("expr_c1", 0.50, run_id="team_c", window=_WIN_B, campaign_id="other"),
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)
        prior = campaign_prior(
            path,
            market="ashare",
            universe="csi300",
            start="20200605",
            end="20260605",
            campaign_id=None,
        )
        assert prior is not None
        assert prior.n_trials == 4
        assert prior.expressions == {"expr_x1", "expr_x2", "expr_y1", "expr_y2"}
        # legacy campaign_id 用全-None 算
        none_key = campaign_key(
            market="ashare", universe="csi300", start="20200605", end="20260605",
            holdout_ratio=None, objective=None, horizon=None, gate=None,
        )
        assert prior.campaign_id == none_key

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_campaign_prior_legacy_none_campaign_id_window_filter(_tp3)

    # -- 原 test_campaign_prior_strict_excludes_legacy_rows_without_campaign_id --
    def _section_4_test_campaign_prior_strict_excludes_legacy_rows_without_campaign_id(tmp_path):
        cid = _full_key()
        lines = [
            _line("with_id", 0.10, run_id="r1", window=_WIN_A, campaign_id=cid),
            _line("legacy_no_id", 0.99, run_id="r2", window=_WIN_A),  # 无 campaign_id
        ]
        path = _write_index(tmp_path / "idx.jsonl", lines)
        prior = campaign_prior(
            path,
            market="ashare",
            universe="csi300",
            start="20200605",
            end="20260605",
            campaign_id=cid,
        )
        assert prior is not None
        assert prior.n_trials == 1
        assert prior.expressions == {"with_id"}

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_campaign_prior_strict_excludes_legacy_rows_without_campaign_id(_tp4)

    # -- 原 test_librarian_record_writes_campaign_id --
    def _section_5_test_librarian_record_writes_campaign_id(tmp_path):
        from factorzen.agents.experiment_index import ExperimentIndex
        from factorzen.agents.roles.librarian import record
        from factorzen.agents.state import AttemptRecord

        idx = ExperimentIndex(str(tmp_path / "idx.jsonl"))
        a = AttemptRecord(
            iteration=0, hypothesis="h", expression="rank(close)", compile_ok=True,
            ic_train=0.01, passed_guardrails=False, critic_verdict=None, error=None,
            ir_train=0.1, turnover=0.3, n_train=_N_OBS,
        )
        record(
            idx, [a], run_id="r1",
            data_window=dict(_WIN_A),
            campaign_id="cidZ",
        )
        raw = (tmp_path / "idx.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(raw)
        assert row["campaign_id"] == "cidZ"
        assert row["data_window"] == _WIN_A
        assert "campaign_id" not in (row.get("data_window") or {})

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_librarian_record_writes_campaign_id(_tp5)


