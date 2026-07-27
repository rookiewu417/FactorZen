"""`node_guardrails` 接线分年 IC 符号一致性门（raw 口径端到端）。

构造三整年数据：某因子在第 1 年强正向、第 2/3 年弱反向 —— 全窗 IC 仍为正、holdout 同号、
|IC| 远超地板，**改动前它会被无条件接纳**；分年门上线后应以「分年IC不稳」被拒。
对照组同样三年但方向恒定，必须仍然通过（零误杀）。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from factorzen.agents.nodes import node_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.guardrails import (
    REJECT_CATEGORY_LIFT_QUEUE,
    REJECT_CATEGORY_YEARLY_SIGN_UNSTABLE,
    classify_reject_category,
)
from factorzen.discovery.scoring import DataBundle
from factorzen.validation.holdout import HoldoutICResult
from factorzen.validation.multiple_testing import TrialLedger

N_STOCKS = 40


def _build_daily(beta_by_year: dict[int, float], *, seed: int = 11) -> pl.DataFrame:
    """三整年日线。股票 i 的 ``vol`` 恒为 (i+1)·1e5（截面秩恒定），
    次日收益 = beta(该日所属年) · z_i + 噪声，z_i = (i−(N−1)/2)/((N−1)/2) ∈ [−1, 1]。

    ⇒ 因子 ``vol`` 对 t→t+1 收益的截面 Rank IC 符号 = 该年 beta 的符号。
    """
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = dt.date(2021, 1, 1)
    while d < dt.date(2024, 1, 1):
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)

    half = (N_STOCKS - 1) / 2
    rows = []
    for i in range(N_STOCKS):
        z = (i - half) / half
        code = f"{600000 + i:06d}.SH"
        px = 10.0
        for t, dd in enumerate(days):
            rows.append({"trade_date": dd, "ts_code": code, "close": px,
                         "open": px, "high": px * 1.01, "low": px * 0.99,
                         "vol": (i + 1) * 1e5, "amount": px * (i + 1) * 1e5})
            if t + 1 < len(days):
                # t 日信号 → t→t+1 收益，beta 取**信号日**所属年
                r = beta_by_year[dd.year] * z + rng.normal(0.0, 0.005)
                px *= 1.0 + r
    return pl.DataFrame(rows)


def _run(daily: pl.DataFrame, *, ic_train: float) -> AgentState:
    """跑一次 node_guardrails（raw 口径、库空）。holdout 结果固定为正且覆盖充足，
    把变量隔离成「分年门」这一条。``train_ratio=1.0`` 让 train 段覆盖全部三年。"""
    bundle = DataBundle.build(daily, train_ratio=1.0)
    state = AgentState(seed=1)
    state.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="vol",
        compile_ok=True, ic_train=ic_train, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=ic_train * 10, turnover=0.1,
        n_train=daily["trade_date"].n_unique()))

    import factorzen.validation.holdout as hmod
    orig = hmod.holdout_ic_result
    hmod.holdout_ic_result = lambda *a, **k: HoldoutICResult(
        0.05, 0.5, (0.01, 0.09), n_days=200)
    try:
        node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                        ledger=TrialLedger(), top_k=5, objective="raw")
    finally:
        hmod.holdout_ic_result = orig
    return state


def _rank1(a: np.ndarray) -> np.ndarray:
    """1..n 的序（构造无并列：vol 互异、fwd 连续）。独立于 polars/项目实现。"""
    order = np.argsort(a, kind="stable")
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(1, a.size + 1, dtype=float)
    return r


def _yearly_ic_signs(daily: pl.DataFrame) -> dict[int, float]:
    """独立复算：逐年「逐日 Spearman(vol, 次日收益)」的年均 —— 纯 numpy，不经被测路径。"""
    df = daily.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close").shift(-1).over("ts_code") / pl.col("close") - 1).alias("fwd")
    ).drop_nulls("fwd")
    out: dict[int, list[float]] = {}
    for (d,), g in df.group_by(["trade_date"], maintain_order=True):
        rho = np.corrcoef(_rank1(g["vol"].to_numpy()), _rank1(g["fwd"].to_numpy()))[0, 1]
        out.setdefault(d.year, []).append(float(rho))
    return {y: float(np.mean(v)) for y, v in out.items()}


# ── 1. 前段正、后段负 → 被新 reason 拒 ─────────────────────────────────────
def test_node_guardrails_rejects_year_flipping_factor():
    daily = _build_daily({2021: 0.02, 2022: -0.0005, 2023: -0.0005})

    # 独立锚：确认构造真的是「2021 强正 / 2022、2023 负 / 全窗正」
    signs = _yearly_ic_signs(daily)
    assert set(signs) == {2021, 2022, 2023}, signs
    assert signs[2021] > 0.5, f"2021 应强正，实得 {signs[2021]:.4f}"
    assert signs[2022] < 0 and signs[2023] < 0, f"2022/2023 应为负，实得 {signs}"
    assert (signs[2021] + signs[2022] + signs[2023]) / 3 > 0, (
        f"全窗均值必须仍为正（否则同号年会变成 2 个）：{signs}"
    )

    state = _run(daily, ic_train=0.05)
    a = state.attempts[0]
    assert a.passed_guardrails is False, (
        f"三年里两年反向 → 占比 1/3 < 60%，应被拒；实得 reject_reason={a.reject_reason!r}"
    )
    assert a.reject_reason is not None and "分年IC不稳" in a.reject_reason, a.reject_reason
    assert "3年" in a.reject_reason, f"完整年数应为 3：{a.reject_reason}"
    assert "33%<60%" in a.reject_reason, f"同号年占比应为 1/3：{a.reject_reason}"
    assert not state.candidates, "被分年门拒的因子不得进候选池"

    # 类别：`classify_reject_category` 先给 yearly_sign_unstable，随后**既有**的 lift 队列
    # 第二通道（|IC|≥地板 + 覆盖足 + 非库重复）把它改写为 lift_queue —— 这与既有的
    # 方向性失败（holdout 反号，classify 返回 None）落在同一条通道上，行为一致。
    # 「新类别是方向性失败、可进 known_invalid」由
    # tests/discovery/test_yearly_sign_gate.py 在 classify + known_invalid 两层钉住。
    assert a.reject_category == REJECT_CATEGORY_LIFT_QUEUE, a.reject_category
    assert "(lift队列,待组合裁决)" in a.reject_reason, a.reject_reason
    assert classify_reject_category(
        [a.reject_reason.replace("(lift队列,待组合裁决)", "")]
    ) == REJECT_CATEGORY_YEARLY_SIGN_UNSTABLE, "改写前的类别应是分年不稳"


# ── 2. 方向恒定的同构因子 → 仍然通过（零误杀对照）───────────────────────────
def test_node_guardrails_keeps_stable_factor():
    daily = _build_daily({2021: 0.0008, 2022: 0.0008, 2023: 0.0008}, seed=23)
    signs = _yearly_ic_signs(daily)
    assert all(v > 0 for v in signs.values()), f"对照组三年应全正：{signs}"

    state = _run(daily, ic_train=0.05)
    a = state.attempts[0]
    assert a.passed_guardrails is True, (
        f"三年同号 → 占比 100%，不该被任何门拒；实得 {a.reject_reason!r}"
    )
    assert len(state.candidates) == 1


# ── 3. 完整年不足 → 门跳过（零误杀）─────────────────────────────────────────
def test_node_guardrails_skips_gate_when_years_insufficient(monkeypatch):
    """把完整年门槛抬到 10000 天 → 一个完整年都没有 → 分年门必须完全不发声。

    同一份「会被拒」的数据在门跳过后重新通过，证明拒绝确实来自分年门本身，
    而不是数据顺带触发了别的门。
    """
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.YEARLY_FULL_YEAR_MIN_DAYS", 10_000, raising=True,
    )
    daily = _build_daily({2021: 0.02, 2022: -0.0005, 2023: -0.0005})
    state = _run(daily, ic_train=0.05)
    a = state.attempts[0]
    assert a.passed_guardrails is True, (
        f"完整年门槛抬高后无完整年 → 门跳过 → 应恢复通过；实得 {a.reject_reason!r}"
    )
    assert len(state.candidates) == 1
