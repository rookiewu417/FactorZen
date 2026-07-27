"""分年度 IC 符号一致性门：纯函数 + 护栏接线 + reject 归类。

单因子轨此前只有 DSR + holdout 覆盖 + train/holdout 同号，**没有任何子期稳定性检验**：
2019–2021 IC=+0.05、2022–2024 IC≈0 的因子全窗 IC=+0.025 也能全过。本门按自然年
分组 train 段逐日 IC，要求「完整年的年均 IC 符号与全窗符号一致」达到占比阈值。
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from factorzen.discovery.guardrails import (
    DEFAULT_DSR_ALPHA,
    REJECT_CATEGORY_HOLDOUT_COVERAGE,
    REJECT_CATEGORY_IC_TOO_WEAK,
    REJECT_CATEGORY_YEARLY_SIGN_UNSTABLE,
    YEARLY_FULL_YEAR_MIN_DAYS,
    acceptance_reasons,
    classify_reject_category,
    guardrail_passed,
    guardrail_reasons,
    library_reasons,
    yearly_sign_consistency,
)


# ── 合成逐日 IC 帧 ──────────────────────────────────────────────────────────
def _ic_frame(spec: dict[int, tuple[int, float]], *, as_date: bool = False) -> pl.DataFrame:
    """{年: (天数, 该年恒定 IC)} → [trade_date, ic]。天数从该年 1/1 起逐日排。

    IC 在年内恒定 ⇒ 年均 = 该常数（期望值可手算，不依赖被测函数）。
    """
    dates: list = []
    ics: list[float | None] = []
    for year, (n_days, ic) in spec.items():
        d0 = dt.date(year, 1, 1)
        for i in range(n_days):
            d = d0 + dt.timedelta(days=i)
            assert d.year == year, "构造错误：天数溢出到下一年"
            dates.append(d if as_date else d.strftime("%Y%m%d"))
            ics.append(ic)
    return pl.DataFrame({"trade_date": dates, "ic": ics})


# ── 1. 纯函数：3 年（前 2 正、第 3 负）→ 占比 2/3 ────────────────────────────
def test_yearly_sign_three_years_two_agree():
    """2020 +0.05 / 2021 +0.04 / 2022 −0.03，各 150 天。

    手算全窗均值 = (150·0.05 + 150·0.04 + 150·(−0.03)) / 450 = 9.0/450 = +0.02 > 0
    ⇒ 同号年 = {2020, 2021} = 2；完整年 = 3；占比 = 2/3。
    """
    df = _ic_frame({2020: (150, 0.05), 2021: (150, 0.04), 2022: (150, -0.03)})
    ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 3, f"三年各 150 天 ≥{YEARLY_FULL_YEAR_MIN_DAYS} 都算完整年，实得 {n_full}"
    assert ratio is not None
    assert abs(ratio - 2 / 3) < 1e-12, f"手算占比 2/3，实得 {ratio}"


def test_yearly_sign_accepts_date_dtype():
    """trade_date 为 pl.Date（node_guardrails 的真实帧就是 date）时结论一致。"""
    spec = {2020: (150, 0.05), 2021: (150, 0.04), 2022: (150, -0.03)}
    r_str, n_str = yearly_sign_consistency(_ic_frame(spec))
    r_date, n_date = yearly_sign_consistency(_ic_frame(spec, as_date=True))
    assert (r_date, n_date) == (r_str, n_str), "Utf8 与 Date 两种 trade_date 结论必须一致"


# ── 2. 恰 1 个完整年 → None（样本不足，门自动跳过）──────────────────────────
def test_yearly_sign_single_full_year_returns_none():
    """2020 满 150 天、2021 仅 100 天（<120 → 残年）⇒ 完整年 = 1 < 2 → 占比 None。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (100, 0.04)})
    ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 1, f"仅 2020 是完整年，实得 {n_full}"
    assert ratio is None, "完整年数 < min_full_years 必须返回 None（不足以裁决 → 不误杀）"


def test_yearly_sign_min_full_years_is_configurable():
    """同一份数据在 min_full_years=1 下应给出占比 1.0（2020 与全窗同为正）。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (100, 0.04)})
    ratio, n_full = yearly_sign_consistency(df, min_full_years=1)
    assert n_full == 1
    assert ratio == 1.0, f"唯一完整年 2020 为正、全窗为正 → 占比 1.0，实得 {ratio}"


# ── 3. 残年剔除的判别用例 ───────────────────────────────────────────────────
def test_yearly_sign_partial_year_excluded_changes_verdict():
    """2019 仅 50 天且年均 −0.02（残年），2020/2021 各 200 天 +0.03。

    手算全窗均值 = (50·(−0.02) + 200·0.03 + 200·0.03)/450 = 11.0/450 ≈ +0.0244 > 0。
    残年被剔除 ⇒ 完整年 = 2、全部同号 ⇒ 占比 = 1.0。
    **判别力**：若残年被误计入，完整年 = 3、同号 2 ⇒ 占比 = 2/3 ≠ 1.0。
    """
    df = _ic_frame({2019: (50, -0.02), 2020: (200, 0.03), 2021: (200, 0.03)})
    ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 2, f"2019 只有 50 天 <{YEARLY_FULL_YEAR_MIN_DAYS}，不算完整年，实得 {n_full}"
    assert ratio == 1.0, f"两个完整年都与全窗（正）同号 → 1.0；实得 {ratio}（2/3 说明残年没剔）"


def test_yearly_sign_min_days_constant_is_monkeypatchable(monkeypatch):
    """把完整年门槛降到 40 天 → 2019 转正为完整年，占比从 1.0 落到 2/3。"""
    monkeypatch.setattr(
        "factorzen.discovery.guardrails.YEARLY_FULL_YEAR_MIN_DAYS", 40, raising=True,
    )
    df = _ic_frame({2019: (50, -0.02), 2020: (200, 0.03), 2021: (200, 0.03)})
    ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 3, f"门槛降到 40 后 2019 算完整年，实得 {n_full}"
    assert ratio is not None and abs(ratio - 2 / 3) < 1e-12, f"手算 2/3，实得 {ratio}"


def test_yearly_sign_counts_valid_days_not_rows():
    """2020 有 150 行但其中 60 行 IC 为 null ⇒ 有效天 90 < 120 → 残年（不是完整年）。"""
    dates: list[str] = []
    ics: list[float | None] = []
    for i in range(150):  # 2020：前 60 天 null，后 90 天 +0.05
        dates.append((dt.date(2020, 1, 1) + dt.timedelta(days=i)).strftime("%Y%m%d"))
        ics.append(None if i < 60 else 0.05)
    for year in (2021, 2022):
        for i in range(200):
            dates.append((dt.date(year, 1, 1) + dt.timedelta(days=i)).strftime("%Y%m%d"))
            ics.append(0.03)
    df = pl.DataFrame({"trade_date": dates, "ic": ics})
    assert 150 >= YEARLY_FULL_YEAR_MIN_DAYS > 90, "构造前提：行数够、有效天不够"
    _ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 2, f"完整年只应有 2021/2022（行数不等于有效天数），实得 {n_full}"


# ── 4. 退化：全 NaN / 空帧 / 全窗均值为 0 ───────────────────────────────────
def test_yearly_sign_all_nan_returns_none():
    df = pl.DataFrame(
        {"trade_date": [f"2020{m:02d}01" for m in range(1, 13)] * 20,
         "ic": [float("nan")] * 240}
    )
    ratio, n_full = yearly_sign_consistency(df)
    assert (ratio, n_full) == (None, 0), f"全 NaN → (None, 0)，实得 {(ratio, n_full)}"


def test_yearly_sign_empty_frame_returns_none():
    df = pl.DataFrame(schema={"trade_date": pl.Utf8, "ic": pl.Float64})
    assert yearly_sign_consistency(df) == (None, 0)
    assert yearly_sign_consistency(None) == (None, 0)


def test_yearly_sign_zero_overall_mean_returns_none():
    """2020 +0.25 / 2021 −0.25 各 150 天 → 全窗均值恰为 0（0.25 二进制可精确表示）。

    符号未定义 ⇒ 无从比对 ⇒ 占比 None，但完整年数仍诚实返回 2。
    """
    df = _ic_frame({2020: (150, 0.25), 2021: (150, -0.25)})
    ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 2
    assert ratio is None, "全窗均值恰为 0 → 无符号可比 → None（不误杀）"


def test_yearly_sign_year_mean_exactly_zero_counts_as_disagree():
    """年均恰为 0 的完整年既不同号也不算通过：2021 全 0 → 同号年只有 2020。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (150, 0.0), 2022: (150, 0.04)})
    ratio, n_full = yearly_sign_consistency(df)
    assert n_full == 3
    assert ratio is not None and abs(ratio - 2 / 3) < 1e-12, (
        f"2020/2022 同号、2021 年均为 0 不算同号 → 2/3，实得 {ratio}"
    )


# ── 5. 接进 guardrail_reasons：默认 None 零回归 ─────────────────────────────
def test_guardrail_reasons_none_series_is_byte_identical():
    """daily_ic_train 缺省 / 显式 None → 与不带该参数逐字节一致（旧调用方零回归）。"""
    kw = {"ic_train": 0.03, "holdout_ic": 0.02, "dsr_pvalue": 0.01}
    base = guardrail_reasons(**kw)
    assert base == [], f"DSR 显著 + 同号 → 全过，实得 {base}"
    assert guardrail_reasons(**kw, daily_ic_train=None) == base

    bad = {"ic_train": 0.03, "holdout_ic": -0.02, "dsr_pvalue": 0.5}
    expect = [
        f"DSR 不显著(p={0.5:.4f}≥{DEFAULT_DSR_ALPHA})",
        f"holdout 反号(train={0.03:.4f}/holdout={-0.02:.4f})",
    ]
    assert guardrail_reasons(**bad) == expect
    assert guardrail_reasons(**bad, daily_ic_train=None) == expect
    assert guardrail_passed(**kw) is True
    assert guardrail_passed(**kw, daily_ic_train=None) is True


def test_guardrail_reasons_yearly_gate_rejects_unstable():
    """2020 +0.05 / 2021 −0.02 / 2022 −0.02（各 150 天）：全窗 = 1.5/450 ≈ +0.0033 > 0
    ⇒ 同号年仅 2020 ⇒ 占比 1/3 = 33% < 60% → 追加 reason。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (150, -0.02), 2022: (150, -0.02)})
    reasons = guardrail_reasons(
        ic_train=0.03, holdout_ic=0.02, dsr_pvalue=0.01, daily_ic_train=df,
    )
    assert reasons == [f"分年IC不稳(同号年占比{1 / 3:.0%}<{0.6:.0%}, 3年)"], reasons
    assert guardrail_passed(
        ic_train=0.03, holdout_ic=0.02, dsr_pvalue=0.01, daily_ic_train=df,
    ) is False, "guardrail_passed 必须与 guardrail_reasons 同源（单一门源）"


def test_guardrail_reasons_yearly_gate_passes_at_threshold():
    """占比 2/3 ≈ 67% ≥ 60% → 不追加 reason（阈值边界不误杀）。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (150, 0.04), 2022: (150, -0.03)})
    assert guardrail_reasons(
        ic_train=0.03, holdout_ic=0.02, dsr_pvalue=0.01, daily_ic_train=df,
    ) == []
    # 阈值抬到 0.9 → 同一序列被拒（证明门确实读了阈值）
    assert guardrail_reasons(
        ic_train=0.03, holdout_ic=0.02, dsr_pvalue=0.01, daily_ic_train=df,
        yearly_min_agree_ratio=0.9,
    ) == [f"分年IC不稳(同号年占比{2 / 3:.0%}<{0.9:.0%}, 3年)"]


def test_guardrail_reasons_yearly_gate_skips_on_insufficient_years():
    """只有 1 个完整年 → 占比 None → 门跳过（零误杀）。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (100, -0.9)})
    assert guardrail_reasons(
        ic_train=0.03, holdout_ic=0.02, dsr_pvalue=0.01, daily_ic_train=df,
    ) == []


def test_guardrail_reasons_coverage_branch_untouched_by_yearly():
    """覆盖不足是数据性失败，走早退分支——不叠加分年门（口径不混）。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (150, -0.02), 2022: (150, -0.02)})
    reasons = guardrail_reasons(
        ic_train=0.03, holdout_ic=float("nan"), dsr_pvalue=0.01,
        holdout_n_days=10, daily_ic_train=df,
    )
    assert reasons == [f"holdout覆盖不足(days={10}/需{60})"], reasons


# ── 6. library 门（默认 gate）也咬合 ────────────────────────────────────────
def test_library_and_acceptance_reasons_wire_yearly_gate():
    """默认 gate='library' 路径必须同样接住分年门，否则整条挖掘链上门是死的。"""
    df = _ic_frame({2020: (150, 0.05), 2021: (150, -0.02), 2022: (150, -0.02)})
    expect = f"分年IC不稳(同号年占比{1 / 3:.0%}<{0.6:.0%}, 3年)"

    assert library_reasons(ic_train=0.03, holdout_ic=0.02) == []
    assert library_reasons(ic_train=0.03, holdout_ic=0.02, daily_ic_train=None) == []
    assert library_reasons(ic_train=0.03, holdout_ic=0.02, daily_ic_train=df) == [expect]

    assert acceptance_reasons(ic_train=0.03, holdout_ic=0.02) == []
    assert acceptance_reasons(ic_train=0.03, holdout_ic=0.02, daily_ic_train=df) == [expect]
    assert acceptance_reasons(
        gate="strict", ic_train=0.03, holdout_ic=0.02, dsr_pvalue=0.01,
        daily_ic_train=df,
    ) == [expect]


# ── 7. reject_category 归类 ────────────────────────────────────────────────
def test_classify_yearly_sign_unstable_category():
    r = f"分年IC不稳(同号年占比{1 / 3:.0%}<{0.6:.0%}, 3年)"
    assert classify_reject_category([r]) == REJECT_CATEGORY_YEARLY_SIGN_UNSTABLE
    assert REJECT_CATEGORY_YEARLY_SIGN_UNSTABLE == "yearly_sign_unstable"


def test_classify_yearly_priority_below_weak_and_coverage():
    """既弱又不稳 → ic_too_weak（强度不足时年度符号本就是噪声）；
    覆盖不足仍最高优先；反号仍返回 None（方向证据）。"""
    yearly = f"分年IC不稳(同号年占比{1 / 3:.0%}<{0.6:.0%}, 3年)"
    assert classify_reject_category(
        [f"train_IC 太弱(|{0.001:.4f}|<0.015)", yearly]
    ) == REJECT_CATEGORY_IC_TOO_WEAK
    assert classify_reject_category(
        ["holdout覆盖不足(days=10/需60)", yearly]
    ) == REJECT_CATEGORY_HOLDOUT_COVERAGE
    assert classify_reject_category(
        [f"holdout 反号(train={0.03:.4f}/holdout={-0.02:.4f})", yearly]
    ) is None


def test_classify_existing_reasons_unchanged():
    """既有 reason 集合的归类逐字节零回归（新关键词不与任何旧文案重叠）。"""
    assert classify_reject_category([]) is None
    assert classify_reject_category(["DSR 不显著(p=0.5000≥0.1)"]) is None
    assert classify_reject_category(["holdout无信号(train=0.0300/holdout=0.0000)"]) is None
    assert classify_reject_category(
        ["残差IC太弱(|0.0010|<0.01)"]
    ) == REJECT_CATEGORY_IC_TOO_WEAK


def test_yearly_sign_unstable_is_directional_evidence_for_known_invalid():
    """新类别是方向性失败 → 必须**不在** known_invalid 的排除表里（可进负例回灌）。

    与 holdout_coverage / ic_too_weak / library_correlated / lift_* 那类
    「非方向性」失败相反。这条断言把语义归类钉在 experiment_index 的真实过滤逻辑上。
    """
    from factorzen.agents.experiment_index import ExperimentIndex

    idx = ExperimentIndex.__new__(ExperimentIndex)
    recs = [
        {"expression": "yearly_bad", "passed": False, "compile_ok": True,
         "ic_train": 0.03,
         "reject_category": REJECT_CATEGORY_YEARLY_SIGN_UNSTABLE,
         "reject_reason": f"分年IC不稳(同号年占比{1 / 3:.0%}<{0.6:.0%}, 3年)"},
        {"expression": "coverage_bad", "passed": False, "compile_ok": True,
         "ic_train": 0.02,
         "reject_category": REJECT_CATEGORY_HOLDOUT_COVERAGE,
         "reject_reason": "holdout覆盖不足(days=10/需60)"},
    ]
    idx._scoped = lambda data_window=None: recs  # type: ignore[method-assign]
    invalid = idx.known_invalid(k=5)
    assert "yearly_bad" in invalid, f"分年不稳应进 known_invalid（方向性证据），实得 {invalid}"
    assert "coverage_bad" not in invalid, "对照组：覆盖失败仍被排除（证明过滤真的在跑）"
