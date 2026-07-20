"""合并自 agents 相关碎片测试（test_guardrail_parity.py）。

test_agent_guardrail_parity.py：Workstream B：护栏双路径对齐 + PBO；抽共享 guardrail_passed
test_agent_dsr_parity.py：P0: DSR 双路径 decision-parity —— Agent 护栏与 M1 同 deflation 口径
test_dsr_sidedness.py：DSR 的符号轴：统计量取 |IR| 时，deflation 基准必须按 2N 算
"""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.agents.nodes import node_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.evaluation import evaluate_expressions
from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue, guardrail_passed
from factorzen.discovery.scoring import DataBundle
from factorzen.validation.deflated_sharpe import deflated_sharpe, expected_max_sharpe
from factorzen.validation.multiple_testing import TrialLedger


# ==== 来自 test_agent_guardrail_parity.py ====
def test_guardrail_passed_positive_ic_significant():
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                            ci_low=0.01, ci_high=0.08) is True


def test_guardrail_passed_rejects_insignificant_dsr():
    """DSR p 值不显著（0.3>0.05）→ 拒。旧 M5/M6 口径 dsr>0.5(即 pval<0.5)会误放行。"""
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.3,
                            ci_low=0.01, ci_high=0.08) is False


def test_guardrail_passed_rejects_sign_mismatch():
    assert guardrail_passed(ic_train=0.05, holdout_ic=-0.04, dsr_pvalue=0.01,
                            ci_low=0.01, ci_high=0.08) is False


def test_guardrail_passed_ci_crosses_zero_now_allowed():
    """2026-07 松一档：移除 holdout CI 单边门。CI 跨零不再单独否决——
    holdout 方向仅由点估计同号把关（DSR 显著 + holdout 与 train 同号即过）。"""
    assert guardrail_passed(ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.01,
                            ci_low=-0.01, ci_high=0.08) is True


def test_guardrail_passed_negative_ic_bidirectional():
    assert guardrail_passed(ic_train=-0.05, holdout_ic=-0.04, dsr_pvalue=0.01,
                            ci_low=-0.08, ci_high=-0.01) is True


def test_guardrail_passed_none_and_nan_conservative():
    assert guardrail_passed(ic_train=None, holdout_ic=0.04, dsr_pvalue=0.01, ci_low=0.01) is False
    nan = float("nan")
    assert guardrail_passed(ic_train=0.05, holdout_ic=nan, dsr_pvalue=0.01, ci_low=0.01) is False


def test_guardrail_negative_ic_without_ci_high_falls_back_to_ci_low():
    assert guardrail_passed(ic_train=-0.05, holdout_ic=-0.04, dsr_pvalue=0.01, ci_low=0.01) is True


def test_cross_path_parity_m1_delegates_to_shared():
    """M1 的 _guard_passed 委托共享入口 acceptance_reasons，逐样本无漂移：
    strict 口径 == guardrail_passed(DSR)，library 口径(默认) == not library_reasons。"""
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


def test_node_guardrails_pbo_is_a_probability_when_pool_is_big_enough():
    """PBO 是「过拟合概率」，必须落在 [0, 1]。

    旧断言 `state.pbo is None or isinstance(state.pbo, float)` 是**恒真**的——nan 也是 float，
    且单候选时 `pool_pbo` 本就返回 nan，所以它连「PBO 是不是概率」都没验证。
    """
    state = _run_guardrails_with(n_candidates=3)

    assert len(state.candidates) >= 2, "需要 ≥2 个候选，PBO(CSCV) 才有定义"
    assert state.pbo == state.pbo, "候选足够时 PBO 不该是 nan"
    assert 0.0 <= state.pbo <= 1.0, f"PBO 是概率，必须 ∈ [0,1]，实得 {state.pbo}"


def test_node_guardrails_pbo_is_nan_when_pool_too_small():
    """候选 < 2 时 CSCV 无从切分 → nan（而非 0.0 之类会被误读为「无过拟合」的值）。"""
    import math

    state = _run_guardrails_with(n_candidates=1)

    assert len(state.candidates) == 1
    assert math.isnan(state.pbo), f"单候选时 PBO 应为 nan，实得 {state.pbo}"

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


def test_dead_expression_yields_none_ic_not_zero():
    """求值后无任何有效截面的表达式必须记 ic=None，而非 0.0。

    `quick_fitness` 空段返回 `ic_mean=0.0`（sentinel）。若照单全收，死表达式会以
    「IC 恰好为 0」的身份混进 `passed` → 膨胀 N，且把 0.0 灌进 IR 池拉低经验方差，
    使 P0 的 deflation 基准算在垃圾上。真实 run `agent_42_2r` 5/5 皆为此形态。
    """
    daily = _mk_daily__dsr_parity()
    bundle = DataBundle.build(daily)
    # 分母恒零 → 全 inf/nan → 过滤后空帧
    res = evaluate_expressions(["div(close, sub(close, close))"], daily, bundle)[0]

    assert res["compile_ok"] is True, "语法合法，compile 应成功"
    assert res["ic_train"] is None, "死表达式的 ic_train 必须是 None，不是 0.0"
    assert res["ir_train"] is None
    assert res["n_train"] == 0, "有效 IC 天数为 0"


def test_live_expression_reports_n_train():
    """正常表达式必须回报 train 段有效 IC 天数（供 DSR 的 n_obs 用，对齐 M1 的 c['n_train']）。"""
    daily = _mk_daily__dsr_parity()
    bundle = DataBundle.build(daily)
    res = evaluate_expressions(["rank(close)"], daily, bundle)[0]

    assert res["ic_train"] is not None
    assert res["n_train"] > 0
    # n_train 是「因子有效 IC 天数」，不是 train 段日历交易日数
    train_days = bundle._segment_mask(daily, "train")["trade_date"].n_unique()
    assert res["n_train"] <= train_days


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


def test_node_guardrails_dsr_pvalue_uses_pool_variance_not_h0_default(_stub_holdout):
    """核心断言：Agent 的 deflation 尺度必须是**池经验方差**，不是 `1/n_obs` 默认值。

    共享配方（`DeflationBasis` + `deflated_pvalue`）：
        ir_pool   = 全体评估过的唯一表达式的 signed train IR
        N         = len(ir_pool)          （Agent 双边 ⇒ effective_trials = 2N）
        sharpe_var= ir_pool.var()
        pval      = deflated_pvalue(signed_ir, basis, n_train)[1]
    """
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


def test_agent_rejects_factor_that_m1_rejects_regression(_stub_holdout):
    """回归：真实 run `agent_43_3r` 里 Agent 放行、M1 拒绝的那个因子，修复后必须被拒。

    实测参数：IR=0.1698，轮内累积 N=3，n_train(n_obs)=305，IR 池 signed。
    默认 1/n_obs 口径 → p=0.0181（放行）；池经验方差口径 → p>0.05（拒绝）。
    """
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


def test_node_guardrails_uses_factor_n_train_not_calendar_days(_stub_holdout):
    """n_obs 口径对齐：DSR 用因子自己的有效 IC 天数，不是 train 段日历交易日数。

    M1 注释（mining_session.py:304-306）明说不能用全段交易日数——后者更大，
    会系统性放大显著性（危险方向）。Agent 从前传的正是日历交易日数。

    构造一个只在「误用日历日数」时才会被放行的因子：n_train=40 时 DSR 不显著，
    但若拿 train 段的 ~210 个日历日去算，z 会被 sqrt(n_obs-1) 放大到显著。
    """
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

# ==== 来自 test_dsr_sidedness.py ====
_SRC = Path(__file__).resolve().parents[2] / "src" / "factorzen"

# team_51_6r 真实 run 反解出的 basis（6/6 候选的 dsr_pvalue 逐位复算吻合）
_REAL_VAR = 0.00127844
_REAL_N = 18
_REAL_NOBS = 303


# ── ground truth：取绝对值 ≡ 试验数翻倍 ────────────────────────────────────


@pytest.mark.parametrize("n", [5, 20, 100])
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


@pytest.mark.parametrize("n", [5, 20, 100])
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


def test_two_sided_basis_doubles_effective_trials_but_reports_honest_n():
    """`n_trials` 是诚实计数（写进 manifest 的那个），`effective_trials` 才是 deflation 用的。

    两者混用会让 manifest 声称「试了 36 个表达式」——那是谎报。
    """
    pool = [0.2, -0.1, 0.35, 0.05]
    one = DeflationBasis.from_ir_pool(pool)
    two = DeflationBasis.from_ir_pool(pool, two_sided=True)

    assert one.n_trials == two.n_trials == 4, "诚实计数不受 sidedness 影响"
    assert one.effective_trials == 4
    assert two.effective_trials == 8
    assert one.sharpe_variance == pytest.approx(two.sharpe_variance), "方差口径与 sidedness 无关"


def test_deflated_pvalue_applies_abs_only_when_two_sided():
    """调用方一律传**带符号** IR；取不取绝对值由 basis 决定。

    这样「统计量 abs 了没有」与「基准 N 还是 2N」不可能各说各话——两者由同一个字段决定。
    """
    var, n_obs = 0.01, 300
    two = DeflationBasis(n_trials=10, sharpe_variance=var, two_sided=True)
    one = DeflationBasis(n_trials=10, sharpe_variance=var)

    p_pos = deflated_pvalue(0.30, two, n_obs)[1]
    p_neg = deflated_pvalue(-0.30, two, n_obs)[1]
    assert p_neg == pytest.approx(p_pos), "双边下 ±0.30 必须同 p（反转因子只是符号相反）"

    p_neg_one = deflated_pvalue(-0.30, one, n_obs)[1]
    assert p_neg_one > 0.5, "单边下负 IR 不该显著——M1 的反转因子须以 neg(x) 形式出现"


def test_two_sided_basis_is_stricter_on_the_same_statistic():
    """同一个 |IR|，双边基准给出的 p 必须更大（门槛更高）。方向错了就是把漂移修反了。"""
    var, n_obs, ir = _REAL_VAR, _REAL_NOBS, 0.20
    p_one = deflated_pvalue(ir, DeflationBasis(_REAL_N, var), n_obs)[1]
    p_two = deflated_pvalue(ir, DeflationBasis(_REAL_N, var, two_sided=True), n_obs)[1]
    assert p_two > p_one


def test_sidedness_flips_a_real_run_candidate():
    """team_51_6r 的真实 basis 上，IR=0.172 的因子单边过关、双边被拒。

    数字来自真实 run 的 manifest 反解（复算 p 与记录 p 逐位吻合），不是合成的。
    """
    p_one = deflated_pvalue(0.172, DeflationBasis(_REAL_N, _REAL_VAR), _REAL_NOBS)[1]
    p_two = deflated_pvalue(0.172, DeflationBasis(_REAL_N, _REAL_VAR, two_sided=True),
                            _REAL_NOBS)[1]

    assert p_one < 0.05, f"单边口径下应过关（p={p_one:.4f}）"
    assert p_two >= 0.05, f"双边口径下应被拒（p={p_two:.4f}）"


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


def test_no_caller_feeds_an_abs_value_into_deflated_pvalue():
    """`abs()` 只许发生在 `deflated_pvalue` 内部。

    调用方一旦自己取绝对值，「统计量是 |IR|」这件事就对 deflation 基准不可见，
    基准 N/2N 的选择随即与统计量脱钩——这正是本次修复前的形态。
    """
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


