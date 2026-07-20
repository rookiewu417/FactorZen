"""Merged discovery tests: test_lift_gates.py

test_lift_queue_gate.py：C1 护栏：lift_queue 无上界 + 库相关软/硬分层回归
test_gray_zone.py：lift 队列 is_lift_queue_candidate / is_gray_zone 判定
test_lift_null.py：lift 统计 null 校准（block sign-flip / AR(1) 日差分 H0）
test_lift_metric_provenance.py：lift_metric 落库 provenance：新旧口径可区分
"""

from __future__ import annotations

from datetime import (
    date,
    timedelta,
)

import numpy as np
import polars as pl

from factorzen.discovery.guardrails import (
    DEFAULT_DUPLICATE_CORR,
    DEFAULT_GRAY_IC_FLOOR,
    DEFAULT_HOLDOUT_MIN_DAYS,
    DEFAULT_IC_FLOOR,
    DEFAULT_RAW_GRAY_IC_FLOOR,
    DEFAULT_RESIDUAL_IC_FLOOR,
    REJECT_CATEGORY_LIBRARY_CORRELATED,
    REJECT_CATEGORY_LIFT_QUEUE,
    acceptance_reasons,
    is_gray_zone,
    is_lift_queue_candidate,
)
from factorzen.discovery.scoring import (
    DEFAULT_DECORR_THRESHOLD,
    library_orthogonal_check,
)

# ==== 来自 test_lift_queue_gate.py ====
# ── a. 过 floor 但 holdout 反号 → 入 lift 队列（修缝隙）──────────────────────

def test_residual_over_floor_holdout_flip_is_lift_queue():
    """残差 train IC=0.0113（≥0.010）+ holdout 反号 + 覆盖足 → 主门不过且可入队。

    这是「过 floor 但 holdout 跌倒掉缝隙」的实锤案例：旧灰区有上界 <0.010，
    会把已过 residual floor 却因反号被拒的候选排除在第二通道之外。
    """
    residual_ic_train = 0.0113
    residual_holdout_ic = -0.004  # 反号
    n_days = DEFAULT_HOLDOUT_MIN_DAYS

    # 主门：library residual 口径 → 反号 → not passed（独立构造期望）
    reasons = acceptance_reasons(
        gate="library",
        ic_train=residual_ic_train,
        holdout_ic=residual_holdout_ic,
        ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
        holdout_n_days=n_days,
        reason_style="residual",
    )
    assert reasons, "holdout 反号必须导致主门 reasons 非空"
    assert any("反号" in r for r in reasons)
    passed = not reasons
    assert passed is False

    cand = {
        "residual_ic_train": residual_ic_train,
        "residual_holdout_ic": residual_holdout_ic,
        "n_residual_holdout_days": n_days,
        "ic_train": 0.03,
        "n_holdout_days": n_days,
    }
    assert is_lift_queue_candidate(cand, objective="residual") is True
    # 调用方契约：not passed 且 is_lift_queue → 打标记
    if not passed and is_lift_queue_candidate(cand, objective="residual"):
        cat = REJECT_CATEGORY_LIFT_QUEUE
    else:
        cat = None
    assert cat == REJECT_CATEGORY_LIFT_QUEUE

# ── b. corr=0.72 软区不硬拒、可入队 ─────────────────────────────────────────

def test_soft_library_corr_072_not_hard_reject_is_lift_queue():
    """max_corr_library=0.72 + 残差达标 → 不被硬拒、落 lift_queue。

    修「两融 corr 0.72 被无声丢弃」：旧门在 0.7 硬拒。
    """
    mc = 0.72
    assert mc < DEFAULT_DUPLICATE_CORR  # 不触发重复硬门
    assert mc >= DEFAULT_DECORR_THRESHOLD  # 软区

    # 硬门度量：threshold=0.95 → ok
    # （无真实 factor_df 时直接用政策阈值语义）
    hard_ok = mc < DEFAULT_DUPLICATE_CORR
    assert hard_ok is True

    residual_ic_train = 0.008  # ≥ gray floor，< residual floor → 主门不过
    reasons = acceptance_reasons(
        gate="library",
        ic_train=residual_ic_train,
        holdout_ic=0.006,
        ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
        holdout_n_days=DEFAULT_HOLDOUT_MIN_DAYS,
        reason_style="residual",
    )
    # 软 reason 附加（与 call site 同文案）
    reasons = [*reasons, f"库相关持保留(corr={mc:.2f})"]
    passed = not reasons
    assert passed is False

    cand = {
        "residual_ic_train": residual_ic_train,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
        "max_corr_library": mc,
    }
    assert is_lift_queue_candidate(cand, objective="residual") is True
    assert REJECT_CATEGORY_LIBRARY_CORRELATED not in (
        cand.get("reject_category"),
    )

# ── c. corr=0.96 硬拒重复，不入队 ────────────────────────────────────────────

def test_duplicate_corr_096_hard_reject_not_lift_queue():
    """max_corr_library=0.96 → 硬拒 library_correlated，不入队列。"""
    mc = 0.96
    assert mc > DEFAULT_DUPLICATE_CORR
    hard_ok = mc < DEFAULT_DUPLICATE_CORR
    assert hard_ok is False

    cand = {
        "residual_ic_train": 0.012,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
        "max_corr_library": mc,
        "reject_category": REJECT_CATEGORY_LIBRARY_CORRELATED,
    }
    assert is_lift_queue_candidate(cand, objective="residual") is False

    # 仅靠 max_corr 也排除（无 reject_category 时）
    cand2 = {
        "residual_ic_train": 0.012,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
        "max_corr_library": mc,
    }
    assert is_lift_queue_candidate(cand2, objective="residual") is False

# ── d. corr=0.5 且现行门全过 → 零回归快速通道 ────────────────────────────────

def test_low_corr_full_pass_zero_regression_fast_path():
    """corr=0.5 且现行 library 门全过 → 通过路径与改动前完全一致。

    改动前语义手工构造期望：
    - library residual 门：|residual_ic|≥0.010、holdout 同号、覆盖≥60 → reasons=[]
    - 库相关 <0.7 → 无软 reason
    - passed=True，不打 lift_queue
    """
    mc = 0.5
    residual_ic_train = 0.015
    residual_holdout_ic = 0.012
    n_days = 100

    reasons = acceptance_reasons(
        gate="library",
        ic_train=residual_ic_train,
        holdout_ic=residual_holdout_ic,
        ic_floor=DEFAULT_RESIDUAL_IC_FLOOR,
        holdout_n_days=n_days,
        reason_style="residual",
    )
    assert reasons == []
    # 软 reason 条件：corr ≥ 0.7 才附加
    if abs(mc) >= DEFAULT_DECORR_THRESHOLD:
        reasons = [*reasons, f"库相关持保留(corr={mc:.2f})"]
    passed = not reasons
    assert passed is True

    cand = {
        "residual_ic_train": residual_ic_train,
        "residual_holdout_ic": residual_holdout_ic,
        "n_residual_holdout_days": n_days,
        "max_corr_library": mc,
    }
    # 函数本身不查 passed；call site 用 not passed 前置
    assert not (
        (not passed) and is_lift_queue_candidate(cand, objective="residual")
    )
    # 快速通道：corr 过 0.7 门
    assert mc < DEFAULT_DECORR_THRESHOLD

# ── 额外边界 ─────────────────────────────────────────────────────────────────

def test_no_upper_bound_residual_002_queues_when_not_passed():
    """残差 0.02 不过主门（反号）时可入队——无上界。"""
    cand = {
        "residual_ic_train": 0.02,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    assert DEFAULT_RESIDUAL_IC_FLOOR < 0.02  # 旧灰区上界之外
    assert is_lift_queue_candidate(cand, objective="residual") is True

def test_coverage_shortfall_not_queued():
    cand = {
        "residual_ic_train": 0.0113,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS - 1,
    }
    assert is_lift_queue_candidate(cand, objective="residual") is False

def test_raw_objective_floor():
    assert is_lift_queue_candidate(
        {"ic_train": DEFAULT_RAW_GRAY_IC_FLOOR, "n_holdout_days": 80},
        objective="raw",
    )
    assert not is_lift_queue_candidate(
        {"ic_train": DEFAULT_RAW_GRAY_IC_FLOOR - 0.001, "n_holdout_days": 80},
        objective="raw",
    )

def test_is_gray_zone_alias():
    c = {
        "residual_ic_train": 0.0113,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    assert is_gray_zone(c) is True
    assert is_gray_zone(c) is is_lift_queue_candidate(c)

def test_library_orthogonal_check_threshold_parameterized():
    """threshold 参数化；默认 0.7 向后兼容；硬拒用 0.95。"""
    # 空库零回归
    ok, mc, nearest = library_orthogonal_check(None, None)  # type: ignore[arg-type]
    assert ok is True and mc == 0.0 and nearest is None

    ok2, _, _ = library_orthogonal_check(None, {}, threshold=DEFAULT_DUPLICATE_CORR)  # type: ignore[arg-type]
    assert ok2 is True

    assert DEFAULT_DECORR_THRESHOLD == 0.7
    assert DEFAULT_DUPLICATE_CORR == 0.95
    assert REJECT_CATEGORY_LIFT_QUEUE == "lift_queue"

def test_soft_reason_does_not_classify_as_coverage():
    """软 reason 不触发 coverage 归类（不得污染 known_invalid 路径）。"""
    from factorzen.discovery.guardrails import classify_reject_category

    reasons = ["库相关持保留(corr=0.72)"]
    assert classify_reject_category(reasons) is None

# ── W1a/W1b：阈值收紧 + 非 top-K 统一门语义 ──────────────────────────────────

def test_old_noise_band_residual_no_longer_queued():
    """train residual 在 (0.003, 0.008) 旧噪声区 → 新 floor 下不入队（W1a 收紧）。"""
    from factorzen.discovery.guardrails import DEFAULT_GRAY_IC_FLOOR

    assert DEFAULT_GRAY_IC_FLOOR == 0.008
    for ric in (0.0035, 0.005, 0.0079):
        assert not is_lift_queue_candidate(
            {
                "residual_ic_train": ric,
                "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
            },
            objective="residual",
        )

def test_nontopk_holdout_coverage_gate_semantics():
    """W1b 统一门语义：train ≥ floor 但 holdout 覆盖不足 → 不入队；覆盖够 → 入队。

    非 top-K 旁路补算 holdout 后同样走 is_lift_queue_candidate，故此处用门函数
    直接验证两侧判别力（与 nodes 旁路契约对齐）。
    """
    from factorzen.discovery.guardrails import DEFAULT_GRAY_IC_FLOOR

    floor_ok = {
        "residual_ic_train": DEFAULT_GRAY_IC_FLOOR,
        "n_residual_holdout_days": 30,  # <60
    }
    assert not is_lift_queue_candidate(floor_ok, objective="residual")

    floor_and_cov = {
        "residual_ic_train": DEFAULT_GRAY_IC_FLOOR,
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    assert is_lift_queue_candidate(floor_and_cov, objective="residual") is True

# ==== 来自 test_gray_zone.py ====
def _base_residual(**kw):
    d = {
        "residual_ic_train": 0.009,  # ≥ DEFAULT_GRAY_IC_FLOOR (0.008)
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
        "ic_train": 0.02,
        "n_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    d.update(kw)
    return d

def _base_raw(**kw):
    d = {
        "ic_train": 0.012,  # ≥ DEFAULT_RAW_GRAY_IC_FLOOR (0.010)
        "n_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    d.update(kw)
    return d

# ── residual 模式 ────────────────────────────────────────────────────────────

def test_lift_queue_residual_at_and_above_floor():
    """残差 |IC| ≥ gray floor 即可入队（无上界）。"""
    assert is_lift_queue_candidate(
        _base_residual(residual_ic_train=DEFAULT_GRAY_IC_FLOOR)
    )
    assert is_lift_queue_candidate(_base_residual(residual_ic_train=0.0099))
    assert is_lift_queue_candidate(_base_residual(residual_ic_train=-0.009))
    # 过 residual floor / 更高：仍可入队（上界已取消）
    assert is_lift_queue_candidate(
        _base_residual(residual_ic_train=DEFAULT_RESIDUAL_IC_FLOOR)
    )
    assert is_lift_queue_candidate(_base_residual(residual_ic_train=0.02))

def test_lift_queue_residual_below_floor_rejected():
    """|IC| < gray floor → 纯噪声，不入队。"""
    assert not is_lift_queue_candidate(
        _base_residual(residual_ic_train=DEFAULT_GRAY_IC_FLOOR - 1e-6)
    )

def test_lift_queue_residual_coverage_insufficient():
    assert not is_lift_queue_candidate(
        _base_residual(n_residual_holdout_days=DEFAULT_HOLDOUT_MIN_DAYS - 1)
    )
    assert not is_lift_queue_candidate(_base_residual(n_residual_holdout_days=0))
    assert not is_lift_queue_candidate(_base_residual(n_residual_holdout_days=None))

def test_lift_queue_sign_flip_does_not_exclude():
    """弱因子 holdout 反号不在队列门重复征收——lift 实验本身是 OOS 裁决。"""
    assert is_lift_queue_candidate(
        _base_residual(
            residual_ic_train=0.009,
            residual_holdout_ic=-0.01,  # 反号
            holdout_ic=-0.02,
        )
    )

def test_lift_queue_library_duplicate_excluded():
    """仅 corr > 0.95 / library_correlated 排除；0.85 软区仍可入队。"""
    assert not is_lift_queue_candidate(
        _base_residual(reject_category=REJECT_CATEGORY_LIBRARY_CORRELATED)
    )
    assert not is_lift_queue_candidate(_base_residual(library_correlated=True))
    assert not is_lift_queue_candidate(
        _base_residual(max_corr_library=DEFAULT_DUPLICATE_CORR + 0.01)
    )
    # 软区 corr 不排除
    assert is_lift_queue_candidate(_base_residual(max_corr_library=0.85))
    assert is_lift_queue_candidate(_base_residual(max_corr_library=0.72))

# ── raw 模式 ─────────────────────────────────────────────────────────────────

def test_lift_queue_raw_floor_no_upper():
    assert is_lift_queue_candidate(_base_raw(ic_train=DEFAULT_RAW_GRAY_IC_FLOOR))
    assert is_lift_queue_candidate(_base_raw(ic_train=0.0149))
    assert is_lift_queue_candidate(_base_raw(ic_train=-0.012))
    assert is_lift_queue_candidate(_base_raw(ic_train=DEFAULT_IC_FLOOR))  # 无上界
    assert not is_lift_queue_candidate(
        _base_raw(ic_train=DEFAULT_RAW_GRAY_IC_FLOOR - 1e-4)
    )

def test_lift_queue_raw_coverage():
    assert not is_lift_queue_candidate(
        _base_raw(n_holdout_days=DEFAULT_HOLDOUT_MIN_DAYS - 1)
    )

def test_lift_queue_objective_override():
    """显式 objective=raw 即使有 residual 字段也走 raw 下界。"""
    # residual 字段 ≥0.008 且 raw ic=0.02 ≥ raw floor → 两口径均 True
    c = _base_residual(ic_train=0.02, residual_ic_train=0.009)
    assert is_lift_queue_candidate(c)  # 默认 residual
    assert is_lift_queue_candidate(c, objective="raw")
    # raw ic 低于 raw floor
    c2 = _base_residual(ic_train=0.004, residual_ic_train=0.009)
    assert is_lift_queue_candidate(c2)  # residual ok
    assert not is_lift_queue_candidate(c2, objective="raw")

def test_is_gray_zone_alias_equivalent():
    """is_gray_zone 薄别名 ≡ is_lift_queue_candidate（含无上界语义）。"""
    cases = [
        _base_residual(residual_ic_train=0.009),
        _base_residual(residual_ic_train=0.02),  # 过旧上界
        _base_residual(residual_ic_train=0.001),  # 低于 floor
        _base_residual(max_corr_library=0.85),
        _base_residual(max_corr_library=0.96),
        _base_raw(ic_train=0.012),
        _base_raw(ic_train=0.02),
    ]
    for c in cases:
        assert is_gray_zone(c) is is_lift_queue_candidate(c)
        assert is_gray_zone(c, objective="raw") is is_lift_queue_candidate(
            c, objective="raw"
        )

def test_lift_queue_constants_documented():
    assert DEFAULT_GRAY_IC_FLOOR == 0.008
    assert DEFAULT_GRAY_IC_FLOOR < DEFAULT_RESIDUAL_IC_FLOOR
    assert DEFAULT_RAW_GRAY_IC_FLOOR == 0.010
    assert DEFAULT_RAW_GRAY_IC_FLOOR < DEFAULT_IC_FLOOR
    assert DEFAULT_DUPLICATE_CORR == 0.95
    assert DEFAULT_IC_FLOOR > DEFAULT_RAW_GRAY_IC_FLOOR

# ==== 来自 test_lift_null.py ====
# ── 1. 管道一致性（防恒真） ──────────────────────────────────────────────────

def test_fixed_diff_matches_paired_lift_stats():
    """固定 diff 序列：手工 lift/SE/half 与 paired_lift_stats 一致。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    # 60 日，block_days=20 → 3 块；奇数中位归前半 → 前 2 块 vs 后 1 块
    n_days = 60
    block_days = 20
    diffs = np.array(
        [0.01] * 20 + [0.03] * 20 + [-0.01] * 20,
        dtype=float,
    )
    dates = [f"d{i:04d}" for i in range(n_days)]
    cand = pl.DataFrame(
        {"trade_date": dates, "ic": diffs.tolist()},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    base = pl.DataFrame(
        {"trade_date": dates, "ic": [0.0] * n_days},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    stats = paired_lift_stats(cand, base, block_days=block_days)

    # 手算
    lift = float(np.mean(diffs))
    block_means = np.array([
        float(np.mean(diffs[0:20])),
        float(np.mean(diffs[20:40])),
        float(np.mean(diffs[40:60])),
    ])
    se = float(block_means.std(ddof=1) / np.sqrt(3))
    first_half = float(np.mean(diffs[0:40]))  # 前 2 块
    second_half = float(np.mean(diffs[40:60]))  # 后 1 块

    assert stats["n_days"] == 60
    assert stats["n_blocks"] == 3
    assert abs(stats["lift"] - lift) < 1e-12
    assert abs(stats["lift_se"] - se) < 1e-12
    assert abs(stats["lift_first_half"] - first_half) < 1e-12
    assert abs(stats["lift_second_half"] - second_half) < 1e-12

def test_null_admission_rates_calls_lift_admission(monkeypatch):
    """null_admission_rates 内部必须走生产 lift_admission（探针计数）。"""
    import factorzen.discovery.lift_null as mod
    from factorzen.discovery.lift_test import lift_admission as real_adm

    calls = {"n": 0}

    def _probe(row, *, threshold=0.001, se_mult=1.0):
        calls["n"] += 1
        return real_adm(row, threshold=threshold, se_mult=se_mult)

    monkeypatch.setattr(mod, "lift_admission", _probe)

    n_sims = 30
    out = mod.null_admission_rates(
        n_days=80,
        block_days=20,
        daily_sigma=0.02,
        ar1=0.0,
        n_sims=n_sims,
        seed=7,
        n_candidates_batch=5,
    )
    assert calls["n"] == n_sims
    assert "p_active" in out
    assert 0.0 <= out["p_active"] <= 1.0

# ── 2. 方向性 ────────────────────────────────────────────────────────────────

def test_higher_se_mult_not_more_active():
    """同 seed 同序列集：se_mult=2.0 的 p_active ≤ se_mult=1.0。"""
    from factorzen.discovery.lift_null import null_admission_rates

    common = dict(
        n_days=400,
        block_days=20,
        daily_sigma=0.03,
        ar1=0.0,
        n_sims=800,
        seed=42,
        n_candidates_batch=10,
    )
    r1 = null_admission_rates(**common, se_mult=1.0)
    r2 = null_admission_rates(**common, se_mult=2.0)
    assert r2["p_active"] <= r1["p_active"]

def test_min_blocks_not_more_active():
    """校准附加规则：min_blocks=10 的 p_active ≤ min_blocks=0。

    n_days=100 → 恒有 n_blocks=5；min_blocks=10 前置 reject 全员，
    故 p_active(10)=0 ≤ p_active(0)。（固定 n_days 下 n_blocks 确定，
    该规则是「窗口够不够」的门槛，不是随机过滤。）
    """
    from factorzen.discovery.lift_null import calibration_table

    rows = calibration_table(
        n_days=100,  # block_days=20 → 5 块
        daily_sigma=0.03,
        ar1=0.0,
        se_mults=(1.0,),
        min_blocks_options=(0, 10),
        n_sims=600,
        seed=11,
    )
    by_mb = {r["min_blocks"]: r for r in rows}
    assert by_mb[10]["p_active"] <= by_mb[0]["p_active"]
    assert by_mb[10]["p_active"] == 0.0

# ── 3. 量级回归（复现审查 §7.2 ~14.8%） ────────────────────────────────────

def test_p_active_magnitude_92_blocks():
    """n_days≈1840（92×20）、ar1=0、se_mult=1.0 → p_active ∈ [0.10, 0.20]。

    宽区间防 seed 敏感；n_sims=2000 固定 seed。scale-free 区 daily_sigma
    取 0.05（SE 通常 > threshold，规则由 se_mult×SE 主导）。
    """
    from factorzen.discovery.lift_null import null_admission_rates

    out = null_admission_rates(
        n_days=1840,
        block_days=20,
        daily_sigma=0.05,
        ar1=0.0,
        se_mult=1.0,
        n_sims=2000,
        seed=0,
        n_candidates_batch=10,
    )
    assert 0.10 <= out["p_active"] <= 0.20, (
        f"p_active={out['p_active']:.4f} 不在 [0.10, 0.20]"
    )

# ── 4. 确定性 ────────────────────────────────────────────────────────────────

def test_null_admission_rates_deterministic():
    """同 seed 两次调用结果完全一致。"""
    from factorzen.discovery.lift_null import null_admission_rates

    kw = dict(
        n_days=120,
        daily_sigma=0.02,
        ar1=0.2,
        n_sims=200,
        seed=99,
        n_candidates_batch=10,
    )
    a = null_admission_rates(**kw)
    b = null_admission_rates(**kw)
    # 率与 FWER 完全一致
    assert a["p_active"] == b["p_active"]
    assert a["p_probation"] == b["p_probation"]
    assert a["p_pass"] == b["p_pass"]
    assert a["fwer_active"]["analytic"] == b["fwer_active"]["analytic"]
    assert a["fwer_active"]["simulated"] == b["fwer_active"]["simulated"]
    assert a["p_active_ci"] == b["p_active_ci"]

# ── 5. AR(1) 生效 ────────────────────────────────────────────────────────────

def test_ar1_changes_lift_se_distribution():
    """正自相关下块均值方差更大 → mean_lift_se(ar1=0.8) > mean_lift_se(ar1=0)。

    原因：AR(1) ρ>0 时块内日差分正相关，块均值的抽样方差升高，
    块 SE = std(block_means)/√n_blocks 随之抬高。
    """
    from factorzen.discovery.lift_null import null_admission_rates

    common = dict(
        n_days=400,
        block_days=20,
        daily_sigma=0.03,
        n_sims=500,
        seed=3,
        n_candidates_batch=10,
        se_mult=1.0,
    )
    r0 = null_admission_rates(**common, ar1=0.0)
    r8 = null_admission_rates(**common, ar1=0.8)
    assert r8["mean_lift_se"] > r0["mean_lift_se"], (
        f"期望 AR(1)=0.8 的 mean_lift_se 更大: "
        f"0.8→{r8['mean_lift_se']}, 0→{r0['mean_lift_se']}"
    )

# ── 6. Wilson CI 边界 ────────────────────────────────────────────────────────

def test_wilson_ci_extremes_no_crash():
    """p=0 / p=1 时 Wilson 区间不炸，且落在 [0,1]。"""
    from factorzen.discovery.lift_null import wilson_ci

    lo0, hi0 = wilson_ci(0, 100)
    assert 0.0 <= lo0 <= hi0 <= 1.0
    assert lo0 == 0.0 or lo0 < 0.05  # 下界贴 0 或极小

    lo1, hi1 = wilson_ci(100, 100)
    assert 0.0 <= lo1 <= hi1 <= 1.0
    assert hi1 == 1.0 or hi1 > 0.95

    lo_empty, hi_empty = wilson_ci(0, 0)
    assert 0.0 <= lo_empty <= hi_empty <= 1.0

# ── 辅助：校准表 / 经验参数 / markdown ───────────────────────────────────────

def test_calibration_table_and_markdown():
    from factorzen.discovery.lift_null import (
        calibration_table,
        format_calibration_markdown,
    )

    rows = calibration_table(
        n_days=100,
        daily_sigma=0.02,
        ar1=0.0,
        se_mults=(1.0, 2.0),
        min_blocks_options=(0, 6),
        n_sims=100,
        seed=1,
    )
    assert len(rows) == 4  # 2 × 2
    md = format_calibration_markdown(rows)
    assert "se_mult" in md
    assert "p_active" in md
    assert "|" in md

def test_estimate_daily_sigma_from_run():
    """粗估：lift_se ≈ σ_block/√n → σ_daily ≈ σ_block × √block_days。"""
    from factorzen.discovery.lift_null import estimate_daily_sigma_from_run

    out = estimate_daily_sigma_from_run(
        {"lift_se": 0.002, "n_blocks": 25, "n_days": 500},
    )
    assert out["n_days"] == 500
    assert out["daily_sigma"] > 0
    # 手算：block_mean_std = 0.002 * √25 = 0.01
    # avg_block_days = 500/25 = 20 → daily_sigma ≈ 0.01 * √20
    expected = 0.01 * np.sqrt(20)
    assert abs(out["daily_sigma"] - expected) < 1e-12

# ==== 来自 test_lift_metric_provenance.py ====
def _meta(**kw):
    base = {
        "session_dir": "sess/metric",
        "run_id": "run_metric",
        "universe": "csi300",
        "horizon": 5,
        "eval_start": "20200101",
        "eval_end": "20260101",
        "git_sha": "deadbeef",
        "now": "2026-07-18",
    }
    base.update(kw)
    return base

def test_upsert_lift_admissions_persists_lift_metric(tmp_path):
    """run_lift_tests → upsert_lift_admissions → FactorRecord.lift_metric == residual_ic_v1。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    dates: list[str] = []
    d = date(2024, 1, 2)
    while len(dates) < 50:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    n_stocks = 40  # residual 日守卫 max(30, k+10)
    active = {
        "lib_a": pl.DataFrame({
            "trade_date": [dd for dd in dates for _ in range(n_stocks)],
            "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
            "factor_value": [float(s) for _ in dates for s in range(n_stocks)],
        }),
    }
    ret = pl.DataFrame({
        "trade_date": [dd for dd in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "ret": [0.01 * s for _ in dates for s in range(n_stocks)],
    })
    cand = pl.DataFrame({
        "trade_date": [dd for dd in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "factor_value": [float(s) + 0.5 for _ in dates for s in range(n_stocks)],
    })

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame({
            "trade_date": ["x"], "ts_code": ["y"], "close": [1.0],
        }),
        leaf_map=None,
        horizon=5,
        admission_start="20240120",
        admission_end="20240315",
        profile_name="ashare_v1",
    )
    rows = run_lift_tests(
        [{"expression": "rank(close)", "residual_ic_train": 0.02, "ic_train": 0.03}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        block_days=12,
        threshold=0.001,
        ctx=ctx,
        lift_workers=1,
    )
    assert rows[0].get("lift_metric") == "residual_ic_v1"
    # 强制 passed 以便 upsert 写入（本测关心 provenance 落盘）
    rows[0]["lift"] = 0.05
    rows[0]["lift_se"] = 0.001
    rows[0]["lift_first_half"] = 0.04
    rows[0]["lift_second_half"] = 0.06
    rows[0]["passed"] = True

    upsert_lift_admissions(
        [rows[0]],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(),
        threshold=0.001,
        se_mult=1.0,
        allow_active=True,
    )
    rec = load_library("ashare", root=str(tmp_path))[0]
    assert rec.lift_metric == "residual_ic_v1"
    assert rec.lift_metric is not None
    assert rec.n_lib_factors == rows[0].get("n_lib_factors") == 1

def test_old_jsonl_missing_lift_metric_reads_as_none():
    """旧口径记录（无 lift_metric 键）读回为 None，不崩——新旧可区分。"""
    from factorzen.discovery.factor_library import FactorRecord

    old = {
        "expression": "rank(close)",
        "market": "ashare",
        "ic_train": 0.05,
        "status": "active",
        "admission_track": "lift",
        "lift": 0.01,
        "horizon": 5,
        "eval_start": "20200101",
        "eval_end": "20240101",
        # 故意无 lift_metric / n_lib_factors
    }
    rec = FactorRecord.from_dict(old)
    assert rec.lift_metric is None
    assert rec.n_lib_factors is None
    assert rec.lift == 0.01
    # 再 round-trip 不丢其它字段、不填假值
    again = FactorRecord.from_dict(rec.to_dict())
    assert again.lift_metric is None
    assert again.n_lib_factors is None

