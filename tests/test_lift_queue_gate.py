"""C1 护栏 gate 重组回归：lift_queue 无上界 + 库相关软/硬分层。

四个必测场景 + 边界。期望值独立构造，不用被测函数生成。
"""
from __future__ import annotations

from factorzen.discovery.guardrails import (
    DEFAULT_DUPLICATE_CORR,
    DEFAULT_HOLDOUT_MIN_DAYS,
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
