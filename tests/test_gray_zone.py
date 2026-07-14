"""lift 队列判定 is_lift_queue_candidate / is_gray_zone 别名。

语义（C1）：无 IC 上界；库重复阈 0.95；覆盖门保留。
独立构造期望，不读被测函数输出当期望（防恒真）。
"""
from __future__ import annotations

from factorzen.discovery.guardrails import (
    DEFAULT_DUPLICATE_CORR,
    DEFAULT_GRAY_IC_FLOOR,
    DEFAULT_HOLDOUT_MIN_DAYS,
    DEFAULT_IC_FLOOR,
    DEFAULT_RAW_GRAY_IC_FLOOR,
    DEFAULT_RESIDUAL_IC_FLOOR,
    REJECT_CATEGORY_LIBRARY_CORRELATED,
    is_gray_zone,
    is_lift_queue_candidate,
)


def _base_residual(**kw):
    d = {
        "residual_ic_train": 0.006,  # ≥ DEFAULT_GRAY_IC_FLOOR
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
        "ic_train": 0.02,
        "n_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    d.update(kw)
    return d


def _base_raw(**kw):
    d = {
        "ic_train": 0.008,  # ≥ DEFAULT_RAW_GRAY_IC_FLOOR
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
    assert is_lift_queue_candidate(_base_residual(residual_ic_train=-0.007))
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
            residual_ic_train=0.006,
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
    assert is_lift_queue_candidate(_base_raw(ic_train=-0.008))
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
    # residual 字段 ≥0.003 但 raw ic=0.02 也 ≥ raw floor → 两口径均 True
    c = _base_residual(ic_train=0.02, residual_ic_train=0.006)
    assert is_lift_queue_candidate(c)  # 默认 residual
    assert is_lift_queue_candidate(c, objective="raw")
    # raw ic 低于 raw floor
    c2 = _base_residual(ic_train=0.004, residual_ic_train=0.006)
    assert is_lift_queue_candidate(c2)  # residual ok
    assert not is_lift_queue_candidate(c2, objective="raw")


def test_is_gray_zone_alias_equivalent():
    """is_gray_zone 薄别名 ≡ is_lift_queue_candidate（含无上界语义）。"""
    cases = [
        _base_residual(residual_ic_train=0.006),
        _base_residual(residual_ic_train=0.02),  # 过旧上界
        _base_residual(residual_ic_train=0.001),  # 低于 floor
        _base_residual(max_corr_library=0.85),
        _base_residual(max_corr_library=0.96),
        _base_raw(ic_train=0.008),
        _base_raw(ic_train=0.02),
    ]
    for c in cases:
        assert is_gray_zone(c) is is_lift_queue_candidate(c)
        assert is_gray_zone(c, objective="raw") is is_lift_queue_candidate(
            c, objective="raw"
        )


def test_lift_queue_constants_documented():
    assert DEFAULT_GRAY_IC_FLOOR == 0.003
    assert DEFAULT_GRAY_IC_FLOOR < DEFAULT_RESIDUAL_IC_FLOOR
    assert DEFAULT_RAW_GRAY_IC_FLOOR == 0.005
    assert DEFAULT_DUPLICATE_CORR == 0.95
    assert DEFAULT_IC_FLOOR > DEFAULT_RAW_GRAY_IC_FLOOR
