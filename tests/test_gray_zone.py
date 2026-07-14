"""灰区判定 is_gray_zone：边界 / 覆盖不足 / 库相关排除 / 反号不排除。

独立构造期望，不读被测函数输出当期望（防恒真）。
"""
from __future__ import annotations

from factorzen.discovery.guardrails import (
    DEFAULT_GRAY_IC_FLOOR,
    DEFAULT_HOLDOUT_MIN_DAYS,
    DEFAULT_IC_FLOOR,
    DEFAULT_RESIDUAL_IC_FLOOR,
    REJECT_CATEGORY_LIBRARY_CORRELATED,
    is_gray_zone,
)


def _base_residual(**kw):
    d = {
        "residual_ic_train": 0.006,  # ∈ [0.003, 0.010)
        "n_residual_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
        "ic_train": 0.02,
        "n_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    d.update(kw)
    return d


def _base_raw(**kw):
    d = {
        "ic_train": 0.008,  # ∈ [0.005, 0.015)
        "n_holdout_days": DEFAULT_HOLDOUT_MIN_DAYS,
    }
    d.update(kw)
    return d


# ── residual 模式 ────────────────────────────────────────────────────────────


def test_gray_zone_residual_inside_band():
    assert is_gray_zone(_base_residual(residual_ic_train=DEFAULT_GRAY_IC_FLOOR))
    assert is_gray_zone(_base_residual(residual_ic_train=0.0099))
    assert is_gray_zone(_base_residual(residual_ic_train=-0.007))


def test_gray_zone_residual_below_floor_or_at_library_floor_rejected():
    """|IC| < gray floor → 纯噪声；|IC| ≥ residual floor → 本可走库门（非灰区）。"""
    assert not is_gray_zone(
        _base_residual(residual_ic_train=DEFAULT_GRAY_IC_FLOOR - 1e-6)
    )
    assert not is_gray_zone(
        _base_residual(residual_ic_train=DEFAULT_RESIDUAL_IC_FLOOR)
    )
    assert not is_gray_zone(
        _base_residual(residual_ic_train=DEFAULT_RESIDUAL_IC_FLOOR + 0.001)
    )


def test_gray_zone_residual_coverage_insufficient():
    assert not is_gray_zone(
        _base_residual(n_residual_holdout_days=DEFAULT_HOLDOUT_MIN_DAYS - 1)
    )
    assert not is_gray_zone(_base_residual(n_residual_holdout_days=0))
    assert not is_gray_zone(_base_residual(n_residual_holdout_days=None))


def test_gray_zone_sign_flip_does_not_exclude():
    """弱因子 holdout 反号不在灰区门重复征收——lift 实验本身是 OOS 裁决。"""
    assert is_gray_zone(
        _base_residual(
            residual_ic_train=0.006,
            residual_holdout_ic=-0.01,  # 反号
            holdout_ic=-0.02,
        )
    )


def test_gray_zone_library_correlated_excluded():
    assert not is_gray_zone(
        _base_residual(reject_category=REJECT_CATEGORY_LIBRARY_CORRELATED)
    )
    assert not is_gray_zone(_base_residual(library_correlated=True))
    assert not is_gray_zone(_base_residual(max_corr_library=0.85))


# ── raw 模式 ─────────────────────────────────────────────────────────────────


def test_gray_zone_raw_band():
    assert is_gray_zone(_base_raw(ic_train=0.005))
    assert is_gray_zone(_base_raw(ic_train=0.0149))
    assert is_gray_zone(_base_raw(ic_train=-0.008))
    assert not is_gray_zone(_base_raw(ic_train=0.0049))
    assert not is_gray_zone(_base_raw(ic_train=DEFAULT_IC_FLOOR))


def test_gray_zone_raw_coverage():
    assert not is_gray_zone(
        _base_raw(n_holdout_days=DEFAULT_HOLDOUT_MIN_DAYS - 1)
    )


def test_gray_zone_objective_override():
    """显式 objective=raw 即使有 residual 字段也走 raw 带。"""
    # residual 字段在 [0.003,0.010) 但 raw ic=0.02 不在 [0.005,0.015)
    c = _base_residual(ic_train=0.02, residual_ic_train=0.006)
    assert is_gray_zone(c)  # 默认 residual
    assert not is_gray_zone(c, objective="raw")


def test_gray_zone_constants_documented():
    assert DEFAULT_GRAY_IC_FLOOR == 0.003
    assert DEFAULT_GRAY_IC_FLOOR < DEFAULT_RESIDUAL_IC_FLOOR
    assert DEFAULT_IC_FLOOR > 0.005
