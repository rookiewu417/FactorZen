"""Merged discovery tests: test_library_residual.py

test_residual_projector.py：ResidualProjector per-date QR 与 residualize_cross_section 契约
test_library_corr_panel_equiv.py：LibraryCorrPanel 与逐对 max_correlation_detail 完全等价
test_library_evidence_link.py：评估记录 → 库 evidence 链接（last_eval_run_id / last_eval_at）
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import time
from datetime import (
    date,
    timedelta,
)
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ==== 来自 test_residual_projector.py ====
# ── 合成工具 ────────────────────────────────────────────────────────────────

def _dates__residual_projector(n: int = 80) -> list[dt.date]:
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days

def _codes(n: int = 50) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]

def _panel_long(
    M: np.ndarray, dates: list, codes: list, *, col: str = "factor_value",
) -> pl.DataFrame:
    """M: (n_dates, n_stocks) → long panel；非有限 → null。"""
    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            v = float(M[i, j])
            rows.append({
                "trade_date": d,
                "ts_code": c,
                col: None if not np.isfinite(v) else v,
            })
    return pl.DataFrame(rows)

def _slow_residualize_panel(factor_df: pl.DataFrame, panel) -> pl.DataFrame:
    """独立慢路径：逐日 residualize_cross_section，镜像 ResidualProjector 语义。

    用于 golden 互证——**不**调用 ResidualProjector，只复用公开单日函数。
    """
    from factorzen.discovery.residual import (
        _day_min_samples,
        residualize_cross_section,
    )

    empty_schema = {
        "trade_date": pl.Date,
        "ts_code": pl.Utf8,
        "factor_value": pl.Float64,
    }
    if factor_df is None or factor_df.is_empty():
        return pl.DataFrame(schema=empty_schema)

    cand = factor_df.with_columns(pl.col("factor_value").fill_nan(None)).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    if cand.is_empty():
        return pl.DataFrame(schema=empty_schema)

    min_n = _day_min_samples(panel.k)
    out_rows: list[dict] = []

    for date_key, day_df in cand.group_by("trade_date", maintain_order=True):
        d = date_key[0] if isinstance(date_key, tuple) else date_key
        di = panel.date_idx.get(d)
        if di is None:
            continue
        codes = day_df["ts_code"].to_list()
        y = day_df["factor_value"].to_numpy().astype(np.float64, copy=False)
        si = np.fromiter(
            (panel.stock_idx.get(c, -1) for c in codes),
            dtype=np.int64,
            count=len(codes),
        )
        valid = si >= 0
        if int(valid.sum()) < min_n:
            continue
        si_v = si[valid]
        y_v = y[valid]
        codes_v = [c for c, ok in zip(codes, valid, strict=True) if ok]
        if y_v.shape[0] < min_n:
            continue
        X_day = panel.X[di, si_v, :]
        resid = residualize_cross_section(y_v, X_day)
        for c, r in zip(codes_v, resid, strict=True):
            out_rows.append({"trade_date": d, "ts_code": c, "factor_value": float(r)})

    if not out_rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.DataFrame(out_rows)

def _sort_panel(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["trade_date", "ts_code"])

def _build_synth_fixture(*, rank_def: bool = False, seed: int = 7):
    """≥3 库因子、≥80 日、含 NaN、含薄截面日。"""
    from factorzen.discovery.residual import build_library_panel

    rng = np.random.default_rng(seed)
    dates = _dates__residual_projector(85)
    codes = _codes(55)
    n_d, n_s = len(dates), len(codes)

    f1 = rng.normal(0, 1, size=(n_d, n_s))
    f2 = rng.normal(0, 1, size=(n_d, n_s))
    f3 = f1.copy() if rank_def else rng.normal(0, 1, size=(n_d, n_s))

    lib_pool = {
        "lib_a": _panel_long(f1, dates, codes),
        "lib_b": _panel_long(f2, dates, codes),
        "lib_c": _panel_long(f3, dates, codes),
    }
    panel = build_library_panel(lib_pool)
    assert panel is not None and panel.k == 3

    cand_m = 0.4 * f1 + 0.3 * f2 + rng.normal(0, 0.8, size=(n_d, n_s))
    cand_m = cand_m.copy()
    cand_m[rng.random(size=(n_d, n_s)) < 0.05] = np.nan
    # 最后 5 日只保留 12 只股票（min_n = max(30, k+10)=30 → 必掉）
    cand_m[-5:, 12:] = np.nan
    # 中间 3 日只保留 25 只（仍 < 30 → 掉）
    cand_m[40:43, 25:] = np.nan

    cand = _panel_long(cand_m, dates, codes).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )

    noise = rng.normal(0, 0.4, size=(n_d, n_s))
    fwd_m = np.where(np.isfinite(cand_m), cand_m + noise, 0.0)
    fwd = _panel_long(fwd_m, dates, codes, col="fwd_ret_1d")

    return dates, codes, panel, cand, fwd

# ── 1. golden parity ────────────────────────────────────────────────────────

def test_residual_projector_parity_suite():
    """ResidualProjector.residualize ≡ 逐日 residualize_cross_section（atol 1e-9）。；compute_residual_ic 快/慢路径 IC 与 n_days 一致。；库内两列完全相同 → QR 路径与 lstsq 残差仍逐值一致。"""
    # -- 原 test_residualize_parity_with_cross_section_slow_path --
    def _section_0_test_residualize_parity_with_cross_section_slow_path():
        from factorzen.discovery.residual import ResidualProjector

        _, _, panel, cand, _ = _build_synth_fixture()
        proj = ResidualProjector(panel)
        fast = _sort_panel(proj.residualize(cand))
        slow = _sort_panel(_slow_residualize_panel(cand, panel))

        assert set(fast.columns) == {"trade_date", "ts_code", "factor_value"}
        assert fast.shape == slow.shape, f"行数不一致 fast={fast.shape} slow={slow.shape}"
        assert fast.height > 0, "golden 应保留有效日"

        joined = fast.join(slow, on=["trade_date", "ts_code"], how="inner", suffix="_slow")
        assert joined.height == fast.height == slow.height
        a = joined["factor_value"].to_numpy()
        b = joined["factor_value_slow"].to_numpy()
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), (
            f"残差不对齐 max|Δ|={np.nanmax(np.abs(a - b))}"
        )

    _section_0_test_residualize_parity_with_cross_section_slow_path()

    # -- 原 test_compute_residual_ic_projector_matches_slow --
    def _section_1_test_compute_residual_ic_projector_matches_slow():
        from factorzen.discovery.residual import ResidualProjector, compute_residual_ic

        _, _, panel, cand, fwd = _build_synth_fixture()
        proj = ResidualProjector(panel)
        slow = compute_residual_ic(cand, panel, fwd)
        fast = compute_residual_ic(cand, panel, fwd, projector=proj)

        assert slow.n_days == fast.n_days
        assert slow.n_days > 0
        assert abs(slow.ic_mean - fast.ic_mean) < 1e-12

    _section_1_test_compute_residual_ic_projector_matches_slow()

    # -- 原 test_rank_deficient_library_parity --
    def _section_2_test_rank_deficient_library_parity():
        from factorzen.discovery.residual import ResidualProjector, compute_residual_ic

        _, _, panel, cand, fwd = _build_synth_fixture(rank_def=True, seed=99)
        X0 = panel.X[0]
        corr = float(np.corrcoef(X0[:, 0], X0[:, 2])[0, 1])
        assert abs(corr) > 0.999, f"秩亏夹具失效 corr={corr}"

        proj = ResidualProjector(panel)
        fast = _sort_panel(proj.residualize(cand))
        slow = _sort_panel(_slow_residualize_panel(cand, panel))
        assert fast.height == slow.height and fast.height > 0
        joined = fast.join(slow, on=["trade_date", "ts_code"], suffix="_slow")
        a = joined["factor_value"].to_numpy()
        b = joined["factor_value_slow"].to_numpy()
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), (
            f"秩亏残差不对齐 max|Δ|={np.nanmax(np.abs(a - b))}"
        )

        r_slow = compute_residual_ic(cand, panel, fwd)
        r_fast = compute_residual_ic(cand, panel, fwd, projector=proj)
        assert r_slow.n_days == r_fast.n_days
        if r_slow.n_days > 0:
            assert abs(r_slow.ic_mean - r_fast.ic_mean) < 1e-12

    _section_2_test_rank_deficient_library_parity()


# ── 2. 秩亏 parity ──────────────────────────────────────────────────────────


# ── 3. 对齐行为 ─────────────────────────────────────────────────────────────

def test_residual_alignment_guard_suite():
    """库外 ts_code 丢弃；库无缺日丢弃——快慢路径一致。；薄截面日两路径均掉日。；不传 projector 时行为与旧签名兼容。"""
    # -- 原 test_unknown_ts_code_and_missing_dates_alignment --
    def _section_0_test_unknown_ts_code_and_missing_dates_alignment():
        from factorzen.discovery.residual import ResidualProjector, build_library_panel

        rng = np.random.default_rng(3)
        dates = _dates__residual_projector(40)
        codes = _codes(45)
        n_d, n_s = len(dates), len(codes)
        lib_pool = {
            f"f{j}": _panel_long(rng.normal(0, 1, size=(n_d, n_s)), dates, codes)
            for j in range(3)
        }
        panel = build_library_panel(lib_pool)
        assert panel is not None

        base = rng.normal(0, 1, size=(n_d, n_s))
        cand = _panel_long(base, dates, codes)
        ghost_date = dt.date(2099, 1, 6)  # 工作日但不在库
        # 库外股票（同日）+ 库内股票但库无日；避免与 cand 已有 (date, code) 重复
        extra = pl.DataFrame({
            "trade_date": [
                dates[0], dates[0],
                ghost_date, ghost_date, ghost_date, ghost_date, ghost_date,
            ],
            "ts_code": [
                "999001.SH", "999002.SH",
                codes[0], codes[1], codes[2], codes[3], codes[4],
            ],
            "factor_value": [1.0, 2.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        })
        cand = pl.concat([cand, extra], how="vertical_relaxed")

        proj = ResidualProjector(panel)
        fast = _sort_panel(proj.residualize(cand))
        slow = _sort_panel(_slow_residualize_panel(cand, panel))

        assert fast.height == slow.height
        assert fast.filter(pl.col("ts_code").str.starts_with("999")).height == 0
        assert fast.filter(pl.col("trade_date") == ghost_date).height == 0
        joined = fast.join(slow, on=["trade_date", "ts_code"], suffix="_slow")
        assert joined.height == fast.height
        assert np.allclose(
            joined["factor_value"].to_numpy(),
            joined["factor_value_slow"].to_numpy(),
            atol=1e-9,
        )

    _section_0_test_unknown_ts_code_and_missing_dates_alignment()

    # -- 原 test_day_guard_drops_thin_days_both_paths --
    def _section_1_test_day_guard_drops_thin_days_both_paths():
        from factorzen.discovery.residual import ResidualProjector, _day_min_samples

        _, _, panel, cand, _ = _build_synth_fixture()
        min_n = _day_min_samples(panel.k)
        proj = ResidualProjector(panel)
        out = proj.residualize(cand)
        if out.height > 0:
            counts = out.group_by("trade_date").len()
            assert int(counts["len"].min()) >= min_n
        slow = _slow_residualize_panel(cand, panel)
        assert set(out["trade_date"].unique().to_list()) == set(
            slow["trade_date"].unique().to_list()
        )

    _section_1_test_day_guard_drops_thin_days_both_paths()

    # -- 原 test_projector_none_keeps_compute_residual_ic_signature --
    def _section_2_test_projector_none_keeps_compute_residual_ic_signature():
        from factorzen.discovery.residual import compute_residual_ic

        sig = inspect.signature(compute_residual_ic)
        assert "projector" in sig.parameters
        assert sig.parameters["projector"].default is None

        _, _, panel, cand, fwd = _build_synth_fixture(seed=1)
        res = compute_residual_ic(cand, panel, fwd)
        assert res.n_days >= 0

    _section_2_test_projector_none_keeps_compute_residual_ic_signature()


# ==== 来自 test_library_corr_panel_equiv.py ====
def _dates__library_corr_panel_equiv(n: int, start: date = date(2024, 1, 2)) -> list[date]:
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


def test_library_corr_panel_equiv_suite():
    """test_empty_pool_returns_zero_none；随机 fixture：覆盖不齐 + float NaN 毒化 + 退化解。；并列 |corr| 取后出现者（c >= best）。；退化池因子只零化自己，不污染与高相关因子的 max。；值级 NaN 毒化该日（整对跳过该日）；null 只剔除该行。；两库因子覆盖不同股票集合：逐对独立 inner，不因「全池一次 join」互相污染。"""
    # -- 原 test_empty_pool_returns_zero_none --
    def _section_0_test_empty_pool_returns_zero_none():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        days = _dates__library_corr_panel_equiv(5)
        stocks = [f"{i:06d}.SH" for i in range(5)]
        cand = _panel_df(days, stocks, np.ones((5, 5)))
        assert max_correlation_detail(cand, {}) == (0.0, None)
        assert build_library_corr_panel({}) is None
        assert build_library_corr_panel(None) is None
        panel = build_library_corr_panel({})
        assert max_correlation_detail(cand, {}, panel=panel) == (0.0, None)

    _section_0_test_empty_pool_returns_zero_none()

    # -- 原 test_panel_matches_pairwise_random_coverage_and_nan --
    def _section_1_test_panel_matches_pairwise_random_coverage_and_nan():
        from factorzen.discovery.scoring import (
            build_library_corr_panel,
            library_orthogonal_check,
            max_correlation_detail,
        )

        rng = np.random.default_rng(42)
        n_days, n_stocks = 60, 50
        days = _dates__library_corr_panel_equiv(n_days)
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

    _section_1_test_panel_matches_pairwise_random_coverage_and_nan()

    # -- 原 test_panel_tie_break_later_pool_entry --
    def _section_2_test_panel_tie_break_later_pool_entry():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        n_days, n_stocks = 40, 40
        days = _dates__library_corr_panel_equiv(n_days)
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

    _section_2_test_panel_tie_break_later_pool_entry()

    # -- 原 test_degenerate_pool_factor_zeroizes_only_self --
    def _section_3_test_degenerate_pool_factor_zeroizes_only_self():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        n_days, n_stocks = 40, 40
        days = _dates__library_corr_panel_equiv(n_days)
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

    _section_3_test_degenerate_pool_factor_zeroizes_only_self()

    # -- 原 test_nan_poisons_day_not_dropped_like_null --
    def _section_4_test_nan_poisons_day_not_dropped_like_null():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        n_stocks = 40
        # 两日：day0 有 40 只；day1 仅 10 只有效（两因子都有）→ day1 不够 30
        days = _dates__library_corr_panel_equiv(2)
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

    _section_4_test_nan_poisons_day_not_dropped_like_null()

    # -- 原 test_partial_coverage_independent_pairs --
    def _section_5_test_partial_coverage_independent_pairs():
        from factorzen.discovery.scoring import build_library_corr_panel, max_correlation_detail

        n_days, n_stocks = 35, 60
        days = _dates__library_corr_panel_equiv(n_days)
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

    _section_5_test_partial_coverage_independent_pairs()


# ==== 来自 test_library_evidence_link.py ====
def _write_lib(root: Path, market: str, records: list[dict]) -> Path:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path

def test_link_evaluation_suite(tmp_path):
    """python 型与 expression 型各一：命中写字段、round-trip、裁决字段不变。；找不到 name → False 且库文件未变（内容 + mtime）。；库文件损坏（乱字节）→ False 不崩。"""
    # -- 原 test_link_evaluation_to_library_python_and_expression --
    def _section_0_test_link_evaluation_to_library_python_and_expression(tmp_path):
        from factorzen.discovery.factor_library import (
            link_evaluation_to_library,
            load_library,
            python_identity,
        )

        py_key = python_identity("momentum_20d")
        recs = [
            {
                "expression": py_key,
                "market": "ashare",
                "kind": "python",
                "name": "momentum_20d",
                "impl": "momentum_20d",
                "ic_train": 0.042,
                "holdout_ic": 0.031,
                "lift": 0.005,
                "status": "active",
                "admission_track": "single",
            },
            {
                "expression": "rank(close)",
                "market": "ashare",
                "kind": "expression",
                "name": "mined_expr_close",
                "ic_train": 0.02,
                "holdout_ic": 0.015,
                "lift": None,
                "status": "probation",
                "admission_track": "lift",
            },
        ]
        _write_lib(tmp_path, "ashare", recs)

        assert link_evaluation_to_library(
            "momentum_20d", "momentum_20d_20260718_120000", "2026-07-18",
            market="ashare", root=str(tmp_path),
        ) is True
        assert link_evaluation_to_library(
            "mined_expr_close", "run_expr_001", "2026-07-18T12:00:00",
            market="ashare", root=str(tmp_path),
        ) is True

        lib = {r.name: r for r in load_library("ashare", root=str(tmp_path))}
        py = lib["momentum_20d"]
        assert py.last_eval_run_id == "momentum_20d_20260718_120000"
        assert py.last_eval_at == "2026-07-18"
        assert py.ic_train == 0.042
        assert py.holdout_ic == 0.031
        assert py.lift == 0.005
        assert py.status == "active"
        assert py.kind == "python"
        assert py.expression == py_key

        ex = lib["mined_expr_close"]
        assert ex.last_eval_run_id == "run_expr_001"
        assert ex.last_eval_at == "2026-07-18T12:00:00"
        assert ex.ic_train == 0.02
        assert ex.holdout_ic == 0.015
        assert ex.lift is None
        assert ex.status == "probation"
        assert ex.kind == "expression"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_link_evaluation_to_library_python_and_expression(_tp0)

    # -- 原 test_link_evaluation_to_library_miss_no_mutation --
    def _section_1_test_link_evaluation_to_library_miss_no_mutation(tmp_path):
        from factorzen.discovery.factor_library import link_evaluation_to_library

        path = _write_lib(
            tmp_path, "ashare",
            [{
                "expression": "rank(vol)",
                "market": "ashare",
                "name": "keep_me",
                "ic_train": 0.01,
                "status": "active",
            }],
        )
        before = path.read_bytes()
        mtime_before = path.stat().st_mtime_ns
        time.sleep(0.02)

        ok = link_evaluation_to_library(
            "does_not_exist", "run_x", "2026-07-18",
            market="ashare", root=str(tmp_path),
        )
        assert ok is False
        assert path.read_bytes() == before
        assert path.stat().st_mtime_ns == mtime_before

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_link_evaluation_to_library_miss_no_mutation(_tp1)

    # -- 原 test_link_evaluation_to_library_corrupt_library_no_crash --
    def _section_2_test_link_evaluation_to_library_corrupt_library_no_crash(tmp_path):
        from factorzen.discovery.factor_library import link_evaluation_to_library

        path = tmp_path / "ashare.jsonl"
        path.write_bytes(b"\xff\xfe\x00\x01\x80\x81 garbage not json\x00")

        ok = link_evaluation_to_library(
            "any", "run_y", "2026-07-18",
            market="ashare", root=str(tmp_path),
        )
        assert ok is False

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_link_evaluation_to_library_corrupt_library_no_crash(_tp2)


