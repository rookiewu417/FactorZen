"""LibraryCorrPanel 与逐对 max_correlation_detail 完全等价（atol=1e-12）。

覆盖：null 行缺失不齐、值级 float NaN 毒化整日、某日 <30 只、池因子截面 std==0、
并列相关取后出现者、pool 空。panel=None 走原路径（零回归）。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def _dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _panel_df(
    days: list[date],
    stocks: list[str],
    values: np.ndarray,
    *,
    null_mask: np.ndarray | None = None,
) -> pl.DataFrame:
    """values shape (n_days, n_stocks). null_mask True → polars null（非 NaN）。"""
    rows = []
    for i, day in enumerate(days):
        for j, code in enumerate(stocks):
            if null_mask is not None and null_mask[i, j]:
                v = None
            else:
                v = float(values[i, j])
            rows.append({"trade_date": day, "ts_code": code, "factor_value": v})
    return pl.DataFrame(rows)


def _assert_same(a: tuple[float, str | None], b: tuple[float, str | None], *, atol: float = 1e-12):
    (mc_a, n_a), (mc_b, n_b) = a, b
    assert n_a == n_b, f"nearest mismatch: {n_a!r} vs {n_b!r}"
    assert mc_a == pytest.approx(mc_b, abs=atol), f"max_corr {mc_a} vs {mc_b}"


def test_panel_none_matches_pairwise_api_signature():
    """不传 panel 时 max_correlation_detail 行为与仅 pool 调用一致（API 零回归）。"""
    from factorzen.discovery.scoring import max_correlation_detail

    days = _dates(40)
    stocks = [f"{i:06d}.SH" for i in range(40)]
    rng = np.random.default_rng(0)
    base = rng.standard_normal((len(days), len(stocks)))
    cand = _panel_df(days, stocks, base)
    pool = {
        "lib_a": _panel_df(days, stocks, base + 0.01 * rng.standard_normal(base.shape)),
        "lib_b": _panel_df(days, stocks, rng.standard_normal(base.shape)),
    }
    r0 = max_correlation_detail(cand, pool)
    r1 = max_correlation_detail(cand, pool, panel=None)
    _assert_same(r0, r1)


def test_empty_pool_returns_zero_none():
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    days = _dates(5)
    stocks = [f"{i:06d}.SH" for i in range(5)]
    cand = _panel_df(days, stocks, np.ones((5, 5)))
    assert max_correlation_detail(cand, {}) == (0.0, None)
    assert build_library_corr_panel({}) is None
    assert build_library_corr_panel(None) is None
    panel = build_library_corr_panel({})
    assert max_correlation_detail(cand, {}, panel=panel) == (0.0, None)


def test_panel_matches_pairwise_random_coverage_and_nan():
    """随机 fixture：覆盖不齐 + float NaN 毒化 + 退化解。"""
    from factorzen.discovery.scoring import (
        build_library_corr_panel,
        library_orthogonal_check,
        max_correlation_detail,
    )

    rng = np.random.default_rng(42)
    n_days, n_stocks = 60, 50
    days = _dates(n_days)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]

    def _rand_vals():
        return rng.standard_normal((n_days, n_stocks))

    # 候选
    cand_v = _rand_vals()
    # 与候选高度相关
    lib_hi = cand_v + 0.05 * _rand_vals()
    # 近似正交
    lib_orth = _rand_vals()
    # 退化：截面常数
    lib_deg = np.ones((n_days, n_stocks))
    # 另一高相关（并列测试用，略低一点以免总赢）
    lib_hi2 = cand_v + 0.08 * _rand_vals()

    # null 掩码：各因子覆盖不同 (date, stock)
    null_cand = rng.random((n_days, n_stocks)) < 0.05
    null_hi = rng.random((n_days, n_stocks)) < 0.08
    null_orth = rng.random((n_days, n_stocks)) < 0.03
    null_deg = np.zeros((n_days, n_stocks), dtype=bool)
    null_hi2 = rng.random((n_days, n_stocks)) < 0.06

    # 值级 NaN：毒化若干整日（present 但 NaN）
    nan_hi = np.zeros((n_days, n_stocks), dtype=bool)
    nan_hi[10, :20] = True  # day 10 混入 NaN
    nan_orth = np.zeros((n_days, n_stocks), dtype=bool)
    nan_orth[20, 5:15] = True

    # 某日仅 25 只有效（<30）——对 lib_orth 在 day 5 大面积 null
    null_orth[5, 25:] = True

    def _mk(vals, null_m, nan_m=None):
        v = vals.copy()
        if nan_m is not None:
            v = v.astype(float)
            v[nan_m] = np.nan
        return _panel_df(days, stocks, v, null_mask=null_m)

    cand = _mk(cand_v, null_cand)
    pool = {
        "orth": _mk(lib_orth, null_orth, nan_orth),
        "degenerate": _mk(lib_deg, null_deg),
        "hi": _mk(lib_hi, null_hi, nan_hi),
        "hi2": _mk(lib_hi2, null_hi2),
    }

    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    assert panel is not None
    assert list(panel.names) == list(pool.keys())
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    _assert_same(pairwise, matrixed)

    # library_orthogonal_check 同步 panel
    ok_p, mc_p, n_p = library_orthogonal_check(cand, pool, threshold=0.7)
    ok_m, mc_m, n_m = library_orthogonal_check(cand, pool, threshold=0.7, panel=panel)
    assert ok_p == ok_m
    assert n_p == n_m
    assert mc_p == pytest.approx(mc_m, abs=1e-12)


def test_panel_tie_break_later_pool_entry():
    """并列 |corr| 取后出现者（c >= best）。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    n_days, n_stocks = 40, 40
    days = _dates(n_days)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rng = np.random.default_rng(7)
    base = rng.standard_normal((n_days, n_stocks))
    # 两个完全相同的池因子 → 与候选相关完全相同
    twin = base.copy()
    cand = _panel_df(days, stocks, base + 0.1 * rng.standard_normal(base.shape))
    pool = {
        "first": _panel_df(days, stocks, twin),
        "second": _panel_df(days, stocks, twin),
    }
    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    _assert_same(pairwise, matrixed)
    assert matrixed[1] == "second"


def test_degenerate_pool_factor_zeroizes_only_self():
    """退化池因子只零化自己，不污染与高相关因子的 max。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    n_days, n_stocks = 40, 40
    days = _dates(n_days)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rng = np.random.default_rng(1)
    good_v = rng.standard_normal((n_days, n_stocks))
    cand = _panel_df(days, stocks, good_v)
    good = _panel_df(days, stocks, good_v)  # corr ≈ 1
    deg = _panel_df(days, stocks, np.ones((n_days, n_stocks)))
    pool = {"good": good, "degenerate": deg}
    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    _assert_same(pairwise, matrixed)
    assert matrixed[0] > 0.99
    assert matrixed[1] == "good"


def test_nan_poisons_day_not_dropped_like_null():
    """值级 NaN 毒化该日（整对跳过该日）；null 只剔除该行。

    构造：仅 1 个交易日有足够样本；该日若 NaN 毒化 → 无幸存日 → corr=0；
    同位置若为 null 剔除后仍够 30 只 → corr 非零。
    """
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    n_stocks = 40
    # 两日：day0 有 40 只；day1 仅 10 只有效（两因子都有）→ day1 不够 30
    days = _dates(2)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rng = np.random.default_rng(99)
    v0 = rng.standard_normal(n_stocks)
    v1 = rng.standard_normal(n_stocks)

    # cand / lib 全日有值
    cand_vals = np.vstack([v0, v1[:n_stocks]])
    # day1 仅前 10 只有行（其余 null）——两因子一致
    null_d1 = np.zeros((2, n_stocks), dtype=bool)
    null_d1[1, 10:] = True

    cand = _panel_df(days, stocks, cand_vals, null_mask=null_d1)
    lib_clean = _panel_df(days, stocks, cand_vals + 0.01, null_mask=null_d1)

    # NaN 版：day0 第 0 只为 float NaN（present），毒化 day0 → 无幸存日
    nan_vals = cand_vals + 0.01
    nan_vals = nan_vals.astype(float)
    nan_vals[0, 0] = np.nan
    lib_nan = _panel_df(days, stocks, nan_vals, null_mask=null_d1)

    pool_clean = {"clean": lib_clean}
    pool_nan = {"nanlib": lib_nan}

    pw_clean = max_correlation_detail(cand, pool_clean)
    pw_nan = max_correlation_detail(cand, pool_nan)
    assert pw_clean[0] > 0.9
    assert pw_nan[0] == 0.0  # 毒化后无幸存日

    for pool in (pool_clean, pool_nan):
        panel = build_library_corr_panel(pool)
        _assert_same(
            max_correlation_detail(cand, pool),
            max_correlation_detail(cand, pool, panel=panel),
        )


def test_partial_coverage_independent_pairs():
    """两库因子覆盖不同股票集合：逐对独立 inner，不因「全池一次 join」互相污染。"""
    from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

    n_days, n_stocks = 35, 60
    days = _dates(n_days)
    stocks = [f"{i:06d}.SH" for i in range(n_stocks)]
    rng = np.random.default_rng(3)
    base = rng.standard_normal((n_days, n_stocks))

    # lib_a 只覆盖前 40 只；lib_b 只覆盖后 40 只
    null_a = np.zeros((n_days, n_stocks), dtype=bool)
    null_a[:, 40:] = True
    null_b = np.zeros((n_days, n_stocks), dtype=bool)
    null_b[:, :20] = True

    cand = _panel_df(days, stocks, base)
    pool = {
        "a": _panel_df(days, stocks, base + 0.02 * rng.standard_normal(base.shape), null_mask=null_a),
        "b": _panel_df(days, stocks, rng.standard_normal(base.shape), null_mask=null_b),
    }
    pairwise = max_correlation_detail(cand, pool)
    panel = build_library_corr_panel(pool)
    matrixed = max_correlation_detail(cand, pool, panel=panel)
    _assert_same(pairwise, matrixed)
    # a 与 cand 高相关应胜出（b 近似噪声）
    assert matrixed[1] == "a"
    assert matrixed[0] > 0.5
